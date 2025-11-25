#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// --- Configuration ---
#define VOCAB_SIZE 256
#define HIDDEN_DIM 512
#define N_LAYERS 4
#define SEQ_LEN 4096
#define POPULATION_SIZE 368
#define BATCH 8 // Streaming batch size per block
#define FIXED_POINT 4
#define SIGMA_SHIFT 4
#define UPDATE_THRESHOLD 270
#define MAX_VAL 127
#define MIN_VAL -127

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// --- Data Structures ---
typedef struct {
    uint8_t *data;
    long length;
} Dataset;

// Transposed Model for Coalesced Access and Shared Compatibility
// Note: In CUDA kernel, accessed such that weights are [Input_Dim * Output_Dim].
// Mapped as W[input_idx * output_dim + output_idx] (Column-Major effectively across threads)
typedef struct {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM]; 
    int8_t gru_weights[N_LAYERS][4][HIDDEN_DIM * HIDDEN_DIM]; 
    int8_t gru_biases[N_LAYERS][2][HIDDEN_DIM]; 
    int8_t mlp_weights[N_LAYERS][2][HIDDEN_DIM * (HIDDEN_DIM * 4)]; 
    int8_t head[HIDDEN_DIM * VOCAB_SIZE]; 
    int8_t ln_weights[N_LAYERS][2][HIDDEN_DIM];
    int8_t ln_out[HIDDEN_DIM];
} EggModel;

// CPU-side Recurrent State for Sampling
typedef struct {
    int8_t h[N_LAYERS][HIDDEN_DIM];
} RecurrentState;

// Global Tables
__device__ int32_t d_EXP2_TABLE[256];
int32_t h_EXP2_TABLE[256];

__device__ int32_t d_debug_updates[2]; // 0: Inc, 1: Dec

// --- Helpers (Host) ---

void init_tables() {
    for(int i=0; i<256; i++) 
        h_EXP2_TABLE[i] = (int32_t)(pow(2.0, (double)i / (1 << FIXED_POINT)) * (1 << FIXED_POINT));
}

static inline uint32_t xorshift32_host(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

static inline int8_t gen_noise_host(uint32_t *rng) {
    uint32_t r = xorshift32_host(rng);
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

void transpose_matrix(int8_t *dst, int8_t *src, int rows, int cols) {
    for(int r=0; r<rows; r++) {
        for(int c=0; c<cols; c++) {
            // src is Row-Major [r][c]. dst becomes [c][r] (Col-Major)
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

void init_model(EggModel *model) {
    uint32_t rng = 42;
    EggModel *temp = (EggModel*)malloc(sizeof(EggModel)); 
    if (!temp) { printf("Failed to allocate temp model\n"); exit(1); }

    // Embedding: Row-Major [Vocab][Hidden]
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) model->embedding[i] = gen_noise_host(&rng);

    // Head Generation (Move here to match C order)
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) temp->head[i] = gen_noise_host(&rng);
    transpose_matrix(model->head, temp->head, VOCAB_SIZE, HIDDEN_DIM);
    
    for(int l=0; l<N_LAYERS; l++) {
        for(int g=0; g<4; g++) {
            // Generate [512][512] logic
            for(int i=0; i<HIDDEN_DIM*HIDDEN_DIM; i++) temp->gru_weights[0][0][i] = gen_noise_host(&rng);
            // Transpose so threads (Output) are contiguous
            // Kernel accesses `w[k*HIDDEN_DIM + tid]`. `tid` is Output. `k` is Input.
            // So W is [Input][Output].
            // `init` generates [Output][Input] (Standard dense logic).
            transpose_matrix(model->gru_weights[l][g], temp->gru_weights[0][0], HIDDEN_DIM, HIDDEN_DIM);
        }
        for(int m=0; m<2; m++) for(int i=0; i<HIDDEN_DIM; i++) model->gru_biases[l][m][i] = 0; 
    }
    
    for(int l=0; l<N_LAYERS; l++) {
        // MLP 1: Expand. In: 512. Out: 2048.
        // Weights: [2048][512].
        // Transpose to [512][2048].
        int dim_expand = HIDDEN_DIM * (HIDDEN_DIM * 4);
        for(int i=0; i<dim_expand; i++) temp->mlp_weights[0][0][i] = gen_noise_host(&rng);
            transpose_matrix(model->mlp_weights[l][0], temp->mlp_weights[0][0], HIDDEN_DIM*4, HIDDEN_DIM);

        // MLP 2: Project. In: 2048. Out: 512.
        // Weights: [512][2048].
        // Transpose to [2048][512].
        for(int i=0; i<dim_expand; i++) temp->mlp_weights[0][0][i] = gen_noise_host(&rng);
        transpose_matrix(model->mlp_weights[l][1], temp->mlp_weights[0][0], HIDDEN_DIM, HIDDEN_DIM*4);
        
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][0][i] = 16;
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][1][i] = 16;
    }
    
    for(int i=0; i<HIDDEN_DIM; i++) model->ln_out[i] = 16;
    
    free(temp);
}

// --- HOST CPU SAMPLING LOGIC (Scalar) ---

int8_t clip_h(int32_t a) {
    return (a > MAX_VAL) ? MAX_VAL : ((a < MIN_VAL) ? MIN_VAL : (int8_t)a);
}

void cpu_ln(const int8_t *x, const int8_t *w, int8_t *out) {
    int32_t sum = 0;
    for(int i=0; i<HIDDEN_DIM; i++) sum += abs(x[i]);
    if(sum == 0) sum = 1;
    int32_t mean = sum / HIDDEN_DIM;
    if(mean == 0) mean = 1;
    for(int i=0; i<HIDDEN_DIM; i++) {
        out[i] = clip_h(((int32_t)x[i] * w[i]) / mean);
    }
}

// Standard Matrix Mul on Transposed Weights (as in Model struct)
// W is [Input][Output].
// In: [Input]. Out: [Output].
void cpu_matmul(const int8_t *x, const int8_t *W, int8_t *out, int in_dim, int out_dim) {
    for(int r=0; r<out_dim; r++) {
        int32_t acc = 0;
        // W is Col-Major wrt output? W index: k * out_dim + r
        for(int k=0; k<in_dim; k++) {
            acc += x[k] * W[k*out_dim + r];
        }
        out[r] = clip_h(acc >> 8);
    }
}

// Different shift for MLP Project (9)
void cpu_matmul_shift(const int8_t *x, const int8_t *W, int8_t *out, int in_dim, int out_dim, int shift) {
    for(int r=0; r<out_dim; r++) {
        int32_t acc = 0;
        for(int k=0; k<in_dim; k++) {
            acc += x[k] * W[k*out_dim + r];
        }
        out[r] = clip_h(acc >> shift);
    }
}

void forward_pass_cpu(EggModel *model, int token, int8_t *logits, RecurrentState *h) {
    int8_t x[HIDDEN_DIM];
    int8_t buf1[HIDDEN_DIM]; // Temp for gates
    int8_t buf2[HIDDEN_DIM]; // Temp for gates
    // Embed
    for(int i=0; i<HIDDEN_DIM; i++) x[i] = model->embedding[token * HIDDEN_DIM + i];
    
    int8_t expanded[2048];

    for(int l=0; l<N_LAYERS; l++) {
        int8_t residual[HIDDEN_DIM];
        for(int i=0; i<HIDDEN_DIM; i++) residual[i] = x[i];
        
        cpu_ln(x, model->ln_weights[l][0], x);
        
        // GRU
        // W0: Input->Buf1. W1: Hidden->Buf2.
        cpu_matmul(x, model->gru_weights[l][0], buf1, HIDDEN_DIM, HIDDEN_DIM);
        cpu_matmul(h->h[l], model->gru_weights[l][1], buf2, HIDDEN_DIM, HIDDEN_DIM);
        
        int8_t ft[HIDDEN_DIM]; // Combined gate
        for(int i=0; i<HIDDEN_DIM; i++) 
            ft[i] = clip_h((int32_t)buf1[i] + buf2[i] + model->gru_biases[l][0][i]);
            
        int8_t gated_past[HIDDEN_DIM];
        for(int i=0; i<HIDDEN_DIM; i++) 
            gated_past[i] = (int8_t)(((int32_t)(ft[i] + 127) * h->h[l][i]) >> 8);
            
        cpu_matmul(x, model->gru_weights[l][2], buf1, HIDDEN_DIM, HIDDEN_DIM);
        cpu_matmul(gated_past, model->gru_weights[l][3], buf2, HIDDEN_DIM, HIDDEN_DIM);
        
        int8_t ht[HIDDEN_DIM];
        for(int i=0; i<HIDDEN_DIM; i++)
            ht[i] = clip_h((int32_t)buf1[i] + buf2[i] + model->gru_biases[l][1][i]);
            
        for(int i=0; i<HIDDEN_DIM; i++) {
            int32_t diff = ht[i] - h->h[l][i];
            int32_t update = ((int32_t)(ft[i] + 127) * diff) >> 8;
            h->h[l][i] = clip_h(h->h[l][i] + update);
            x[i] = h->h[l][i];
        }
        
        for(int i=0; i<HIDDEN_DIM; i++) x[i] = clip_h((int32_t)x[i] + residual[i]);
        
        for(int i=0; i<HIDDEN_DIM; i++) residual[i] = x[i];
        cpu_ln(x, model->ln_weights[l][1], x);
        
        // MLP Expand 512->2048
        cpu_matmul(x, model->mlp_weights[l][0], expanded, HIDDEN_DIM, 2048);
        // MLP Project 2048->512 (shift 9)
        cpu_matmul_shift(expanded, model->mlp_weights[l][1], x, 2048, HIDDEN_DIM, 9);
        
        for(int i=0; i<HIDDEN_DIM; i++) x[i] = clip_h((int32_t)x[i] + residual[i]);
    }
    
    cpu_ln(x, model->ln_out, x);
    cpu_matmul(x, model->head, logits, HIDDEN_DIM, VOCAB_SIZE);
}

int sample_logits(int8_t *logits) {
    int32_t probs[VOCAB_SIZE];
    int32_t sum = 0;
    for(int i=0; i<VOCAB_SIZE; i++) {
        int idx = (int32_t)logits[i] + 128;
        idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
        probs[i] = h_EXP2_TABLE[idx];
        sum += probs[i];
    }
    if(sum == 0) return 0;
    int32_t r = rand() % sum;
    int32_t acc = 0;
    for(int i=0; i<VOCAB_SIZE; i++) {
        acc += probs[i];
        if(r < acc) return i;
    }
    return VOCAB_SIZE - 1;
}

void sample_model_host(EggModel *model, const uint8_t *seed_text, int seed_len, int gen_len) {
    RecurrentState state;
    for(int l=0; l<N_LAYERS; l++) for(int i=0; i<HIDDEN_DIM; i++) state.h[l][i] = 0;
    
    int8_t logits[VOCAB_SIZE];
    
    printf("\033[32m");
    int input_token = 0;

    for(int t=0; t < seed_len + gen_len; t++) {
         if (t < seed_len) {
            input_token = seed_text[t];
            if(input_token >= 32 && input_token <= 126) printf("%c", input_token);
            else printf(".");
        } else {
            if (t == seed_len) printf("\033[36m");
            if(input_token >= 32 && input_token <= 126) printf("%c", input_token);
            else printf(".");
        }
        
        forward_pass_cpu(model, input_token, logits, &state);
        
        if(t >= seed_len - 1) {
            input_token = sample_logits(logits);
        }
    }
    printf("\033[0m\n");
}

// --- DEVICE ---

__device__ __forceinline__ uint32_t hash_rng(uint32_t s, uint32_t idx) {
    uint32_t x = s + idx * 0x9e3779b9u; 
    x ^= x >> 16; x *= 0x85ebca6b; x ^= x >> 13; x *= 0xc2b2ae35; x ^= x >> 16;
    return x;
}

__device__ __forceinline__ int8_t noise_from_hash(uint32_t s, uint32_t idx) {
    uint32_t r = hash_rng(s, idx);
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

__device__ __forceinline__ int8_t clip(int32_t a) {
    return (a > MAX_VAL) ? MAX_VAL : ((a < MIN_VAL) ? MIN_VAL : (int8_t)a);
}

__device__ __forceinline__ int32_t warpReduceSum(int32_t val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ int32_t blockReduceSum(int32_t val) {
    __syncthreads(); // Prevent fast warp entry race
    static __shared__ int32_t shared[32]; 
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // Only first warp computes final sum
    val = (threadIdx.x < (blockDim.x / 32)) ? shared[threadIdx.x] : 0;
    
    if (wid == 0) val = warpReduceSum(val);
    
    // Broadcast result to all threads
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    
    return shared[0];
}

extern __shared__ int8_t s_mem[]; // [BATCH][2048] flattened

__global__ void __launch_bounds__(HIDDEN_DIM) train_sequence_kernel(
    const uint8_t * __restrict__ dataset,
    long data_len,
    int start_idx,
    const EggModel * __restrict__ model,
    int8_t * __restrict__ pop_states, 
    int32_t *accum_loss,
    uint32_t step_seed
) {
    int tid = threadIdx.x;
    int block_p_idx = blockIdx.x; 
    
    int8_t *s_ptr = s_mem;
    
    // Local State (Register)
    int8_t h[BATCH][N_LAYERS];
    
    // Load State
    for(int b=0; b<BATCH; b++) {
        int p_idx = block_p_idx * BATCH + b;
        for(int l=0; l<N_LAYERS; l++) {
             // Index: p_idx * (layers * hidden) + l * hidden + tid
             if(tid < HIDDEN_DIM) 
                 h[b][l] = pop_states[p_idx * (N_LAYERS * HIDDEN_DIM) + l * HIDDEN_DIM + tid];
        }
    }
    
    int32_t my_loss[BATCH]; 
    for(int b=0; b<BATCH; b++) my_loss[b] = 0;
    
    // Batched Loop
    for (int t = 0; t < SEQ_LEN; t++) {
        __syncthreads(); 
        
        // 1. Load Inputs
        for(int b=0; b<BATCH; b++) {
            int p_idx = block_p_idx * BATCH + b;
            long pair_idx = p_idx / 2;
            long stride = data_len / (POPULATION_SIZE / 2);
            long stream_pos = (start_idx + (pair_idx * stride)) % (data_len - SEQ_LEN);
            uint8_t input_token = dataset[stream_pos + t];
            
            if (tid < HIDDEN_DIM) {
                s_ptr[b * 2048 + tid] = model->embedding[input_token * HIDDEN_DIM + tid];
            }
        }
        __syncthreads();
        
        // 2. Layers
        for (int l = 0; l < N_LAYERS; l++) {
            int8_t gru_resid[BATCH];

            // LN 1
            for(int b=0; b<BATCH; b++) {
                int8_t *sx_b = &s_ptr[b * 2048];
                int32_t val = sx_b[tid];
                gru_resid[b] = (int8_t)val; // Save Pre-LN residual
                
                int32_t sum = blockReduceSum(abs(val));
                if(!sum) sum=1; int32_t mean = sum/HIDDEN_DIM; if(!mean) mean=1;
                val = clip(((int32_t)val * model->ln_weights[l][0][tid]) / mean);
                sx_b[tid] = val; // x_in
            }
            __syncthreads();
            
            // Save H to shared (Offset 512) for MatMul2
            for(int b=0; b<BATCH; b++) {
                 if(tid < HIDDEN_DIM) s_ptr[b*2048 + 512 + tid] = h[b][l];
            }
            __syncthreads();

            // Rank-1 Precalc (xB)
            int32_t xB_m1[BATCH], xB_m2[BATCH], xB_m3[BATCH], xB_m4[BATCH];
            
            for(int b=0; b<BATCH; b++) {
                int p_idx = block_p_idx * BATCH + b;
                int pair_idx = p_idx / 2;
                // SEED FIX: Use pair_idx, not p_idx, for noise generation consistency
                uint32_t seed = (step_seed + pair_idx) + (l * 100); 
                
                // M1 (In=x)
                int8_t b1 = noise_from_hash(seed + 1 + HIDDEN_DIM, tid);
                xB_m1[b] = blockReduceSum((int32_t)s_ptr[b*2048+tid] * b1);
                
                // M2 (In=h)
                int8_t b2 = noise_from_hash(seed + 2 + HIDDEN_DIM, tid);
                xB_m2[b] = blockReduceSum((int32_t)s_ptr[b*2048+512+tid] * b2);
            }

            // MatMul 1 & 2 (Fused Loop)
            int32_t dot1[BATCH]; for(int b=0; b<BATCH; b++) dot1[b]=0;
            int32_t dot2[BATCH]; for(int b=0; b<BATCH; b++) dot2[b]=0;
            
            const int8_t *w1 = &model->gru_weights[l][0][0];
            const int8_t *w2 = &model->gru_weights[l][1][0];
            
            for(int k=0; k<HIDDEN_DIM; k++) {
                int8_t v1 = w1[k*HIDDEN_DIM + tid];
                int8_t v2 = w2[k*HIDDEN_DIM + tid];
                for(int b=0; b<BATCH; b++) {
                    dot1[b] += (int32_t)s_ptr[b*2048 + k] * v1;
                    dot2[b] += (int32_t)s_ptr[b*2048 + 512 + k] * v2;
                }
            }
            
            int8_t ft_reg[BATCH];
            int8_t gated_reg[BATCH];
            
            for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int ns = (p_idx % 2 == 0) ? 1 : -1;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed + pair_idx) + (l * 100);
                 
                 int8_t a1 = noise_from_hash(seed + 1, tid);
                 if(ns!=0) dot1[b] += ((xB_m1[b] * (int32_t)a1) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int8_t a2 = noise_from_hash(seed + 2, tid);
                 if(ns!=0) dot2[b] += ((xB_m2[b] * (int32_t)a2) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int8_t r1 = dot1[b] >> 8;
                 int8_t r2 = dot2[b] >> 8;
                 ft_reg[b] = clip(r1 + r2 + model->gru_biases[l][0][tid]);
                 gated_reg[b] = (int8_t)(((int32_t)(ft_reg[b] + 127) * h[b][l]) >> 8);
            }
            
            // Save Gated to Shared (Offset 1024) for M4
            __syncthreads();
            for(int b=0; b<BATCH; b++) {
                if(tid < HIDDEN_DIM) s_ptr[b*2048 + 1024 + tid] = gated_reg[b];
            }
            __syncthreads();
            
            // Rank-1 M3 & M4
             for(int b=0; b<BATCH; b++) {
                int p_idx = block_p_idx * BATCH + b;
                int pair_idx = p_idx / 2;
                uint32_t seed = (step_seed + pair_idx) + (l * 100);
                
                // M3 (In=x)
                int8_t b3 = noise_from_hash(seed + 3 + HIDDEN_DIM, tid);
                xB_m3[b] = blockReduceSum((int32_t)s_ptr[b*2048+tid] * b3);
                
                // M4 (In=gated)
                int8_t b4 = noise_from_hash(seed + 4 + HIDDEN_DIM, tid);
                xB_m4[b] = blockReduceSum((int32_t)s_ptr[b*2048+1024+tid] * b4);
            }
            
            for(int b=0; b<BATCH; b++) { dot1[b]=0; dot2[b]=0; } // reuse regs
            const int8_t *w3 = &model->gru_weights[l][2][0];
            const int8_t *w4 = &model->gru_weights[l][3][0];
            
            for(int k=0; k<HIDDEN_DIM; k++) {
                int8_t v3 = w3[k*HIDDEN_DIM + tid];
                int8_t v4 = w4[k*HIDDEN_DIM + tid];
                for(int b=0; b<BATCH; b++) {
                    dot1[b] += (int32_t)s_ptr[b*2048 + k] * v3;
                    dot2[b] += (int32_t)s_ptr[b*2048 + 1024 + k] * v4;
                }
            }
            
            // Update H
            for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int ns = (p_idx % 2 == 0) ? 1 : -1;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed + pair_idx) + (l * 100);
                 
                 int8_t a3 = noise_from_hash(seed + 3, tid);
                 if(ns!=0) dot1[b] += ((xB_m3[b] * (int32_t)a3) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int8_t a4 = noise_from_hash(seed + 4, tid);
                 if(ns!=0) dot2[b] += ((xB_m4[b] * (int32_t)a4) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int8_t ht = clip((dot1[b] >> 8) + (dot2[b] >> 8) + model->gru_biases[l][1][tid]);
                 
                 int32_t diff = ht - h[b][l];
                 int32_t update = ((int32_t)(ft_reg[b] + 127) * diff) >> 8;
                 h[b][l] = clip(h[b][l] + update);
            }
            
            // Residual & New X
            for(int b=0; b<BATCH; b++) {
                // Use stored Pre-LN residual
                int8_t new_x = clip((int32_t)h[b][l] + gru_resid[b]); 
                s_ptr[b*2048 + tid] = new_x; // Update s_x for MLP
            }
            __syncthreads(); 
            
            // Save Pre-LN Residual for MLP
            int8_t mlp_resid[BATCH];
            for(int b=0; b<BATCH; b++) mlp_resid[b] = s_ptr[b*2048 + tid];

            // MLP LN 2
            for(int b=0; b<BATCH; b++) {
                int32_t val = s_ptr[b*2048 + tid];
                int32_t sum = blockReduceSum(abs(val));
                if(!sum) sum=1; int32_t mean = sum/HIDDEN_DIM; if(!mean) mean=1;
                val = clip(((int32_t)val * model->ln_weights[l][1][tid]) / mean);
                s_ptr[b*2048 + tid] = val;
            }
            __syncthreads(); 

            // MLP Expand (512 -> 2048)
            int8_t mlp_res[BATCH][4];
            
            // 1. Rank-1 xB for MLP1
            int32_t xB_mlp1[BATCH];
            for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed+pair_idx) + (l * 100) + 5;
                 int8_t b_val = noise_from_hash(seed + 2048, tid); // Offset A size 2048
                 xB_mlp1[b] = blockReduceSum((int32_t)s_ptr[b*2048+tid] * b_val);
            }
            
            // 2. Init accums
            int32_t acc_mlp[BATCH][4]; 
            for(int b=0; b<BATCH; b++) for(int s=0; s<4; s++) acc_mlp[b][s] = 0;
            
            const int8_t *w_mlp1 = &model->mlp_weights[l][0][0];
             for(int k=0; k<HIDDEN_DIM; k++) {
                 for(int sub=0; sub<4; sub++) {
                      // Access: W[k * 2048 + (tid + sub*512)]
                      int out_idx = tid + sub * HIDDEN_DIM;
                      int8_t w_val = w_mlp1[k*2048 + out_idx];
                      for(int b=0; b<BATCH; b++) {
                          acc_mlp[b][sub] += (int32_t)s_ptr[b*2048 + k] * w_val;
                      }
                 }
             }
             
             // 3. Noise & Store
             for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int ns = (p_idx % 2 == 0) ? 1 : -1;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed+pair_idx) + (l * 100) + 5;
                 
                 for(int sub=0; sub<4; sub++) {
                     int out_idx = tid + sub * HIDDEN_DIM;
                     int8_t a_val = noise_from_hash(seed, out_idx);
                     if(ns!=0) acc_mlp[b][sub] += ((xB_mlp1[b] * (int32_t)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                     mlp_res[b][sub] = clip(acc_mlp[b][sub] >> 8);
                 }
             }
             
             // MLP Project (2048 -> 512)
             // Input is `mlp_res` (Distributed).
             // We MUST write to Shared to broadcast for next MatMul.
             __syncthreads();
             for(int b=0; b<BATCH; b++) {
                 for(int sub=0; sub<4; sub++) {
                     s_ptr[b*2048 + tid + sub*HIDDEN_DIM] = mlp_res[b][sub];
                 }
             }
             __syncthreads();
             
             // xB for M2 (Input 2048)
             int32_t xB_mlp2[BATCH];
             for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed+pair_idx) + (l * 100) + 6;
                 
                 int32_t local_sum = 0;
                 for(int sub=0; sub<4; sub++) {
                     int in_idx = tid + sub*HIDDEN_DIM;
                     int8_t b_val = noise_from_hash(seed + 512, in_idx); // Offset B=512
                     local_sum += (int32_t)s_ptr[b*2048 + in_idx] * b_val;
                 }
                 xB_mlp2[b] = blockReduceSum(local_sum);
             }
             
             int32_t acc_proj[BATCH]; for(int b=0; b<BATCH; b++) acc_proj[b] = 0;
             const int8_t *w_mlp2 = &model->mlp_weights[l][1][0];
             // K=2048. Loop 2048.
             for(int k=0; k<2048; k++) {
                 int8_t w_val = w_mlp2[k*HIDDEN_DIM + tid];
                 for(int b=0; b<BATCH; b++) {
                     acc_proj[b] += (int32_t)s_ptr[b*2048 + k] * w_val;
                 }
             }

            // Apply Result
             for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int ns = (p_idx % 2 == 0) ? 1 : -1;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed+pair_idx) + (l * 100) + 6;
                 
                 int8_t a_val = noise_from_hash(seed, tid);
                 if(ns!=0) acc_proj[b] += ((xB_mlp2[b] * (int32_t)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int32_t res = acc_proj[b] >> 9; // Shift 9 for MLP project
                 s_ptr[b*2048 + tid] = clip(res + mlp_resid[b]);
             }
             __syncthreads();
         
        } // l Loop
        
        // 3. Head LN
        for(int b=0; b<BATCH; b++) {
            int32_t val = s_ptr[b * 2048 + tid];
            int32_t sum = blockReduceSum(abs(val));
            if(!sum) sum=1; int32_t mean = sum/HIDDEN_DIM; if(!mean) mean=1;
            s_ptr[b * 2048 + tid] = clip(((int32_t)val * model->ln_out[tid]) / mean);
        }
        __syncthreads();
        
        // 4. Head Dense
        int32_t logits[BATCH]; 
        // xB
         int32_t xB_head[BATCH];
         for(int b=0; b<BATCH; b++) {
             int p_idx = block_p_idx * BATCH + b;
             int pair_idx = p_idx / 2;
             uint32_t seed = (step_seed+pair_idx) + 999;
             int8_t b_val = noise_from_hash(seed + VOCAB_SIZE, tid); // B off 256
             xB_head[b] = blockReduceSum((int32_t)s_ptr[b*2048+tid] * b_val);
         }
         
         const int8_t *w_head = &model->head[0];
         if(tid < VOCAB_SIZE) {
            int32_t acc[BATCH]; for(int b=0; b<BATCH; b++) acc[b]=0;
             
            for(int k=0; k<HIDDEN_DIM; k++) {
                int8_t w = w_head[k*VOCAB_SIZE + tid];
                for(int b=0; b<BATCH; b++) acc[b] += (int32_t)s_ptr[b*2048 + k] * w;
            }
            
            for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int ns = (p_idx % 2 == 0) ? 1 : -1;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed+pair_idx) + 999;
                 
                 int8_t a_val = noise_from_hash(seed, tid);
                 if(ns!=0) acc[b] += ((xB_head[b] * (int32_t)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 logits[b] = acc[b] >> 8;
            }
         }
         
         // 5. Softmax Loss
          uint8_t targets[BATCH];
          if(tid==0) {
              for(int b=0; b<BATCH; b++) {
                  int p_idx = block_p_idx * BATCH + b;
                  long pair_idx = p_idx / 2;
                  long stride = data_len / (POPULATION_SIZE / 2);
                  long stream_pos = (start_idx + (pair_idx * stride)) % (data_len - SEQ_LEN);
                  targets[b] = dataset[stream_pos + t + 1];
              }
          }
          
          for(int b=0; b<BATCH; b++) {
              int32_t l_val = (tid < VOCAB_SIZE) ? logits[b] : -9999;
              int32_t idx = l_val + 128;
              int32_t exp_val = (tid < VOCAB_SIZE) ? d_EXP2_TABLE[(idx<0)?0:(idx>255?255:idx)] : 0;
              int32_t sum_exp = blockReduceSum(exp_val);
              
              int32_t log_sum = 0;
              if(tid==0) {
                 int32_t x = sum_exp;
                 if (x > 0) {
                     int pos = 0;
                     if (x >= 1<<16) { x >>= 16; pos += 16; }
                     if (x >= 1<<8)  { x >>= 8;  pos += 8; }
                     if (x >= 1<<4)  { x >>= 4;  pos += 4; }
                     if (x >= 1<<2)  { x >>= 2;  pos += 2; }
                     if (x >= 1<<1)  {           pos += 1; }
                     int32_t fraction = (pos>=4) ? (sum_exp-(1<<pos))>>(pos-4) : (sum_exp-(1<<pos))<<(4-pos);
                     log_sum = (pos<<4) + fraction - 64;
                 }
              }
              
              __syncthreads();
              if(tid < VOCAB_SIZE) s_ptr[tid] = (int8_t)clip(l_val); 
              __syncthreads();
              
              if(tid==0) {
                  int32_t target_l = (int32_t)s_ptr[targets[b]] + 128;
                  my_loss[b] += (log_sum - target_l);
              }
          }

    } // Seq Loop
    
    if (tid == 0) {
        for(int b=0; b<BATCH; b++) accum_loss[block_p_idx * BATCH + b] = my_loss[b];
    }
    
    // Store State
     for(int b=0; b<BATCH; b++) {
        int p_idx = block_p_idx * BATCH + b;
        for(int l=0; l<N_LAYERS; l++) {
             if(tid < HIDDEN_DIM) 
                 pop_states[p_idx * (N_LAYERS * HIDDEN_DIM) + l * HIDDEN_DIM + tid] = h[b][l];
        }
    }
}

__global__ void update_matrix_kernel(
    int8_t * __restrict__ W,
    int rows,
    int cols,
    int offset_A,
    int offset_B,
    int seed_base_add,
    const int32_t * __restrict__ fitnesses,
    uint32_t step_seed
) {
    // W is stored as [Input][Output] = [cols][rows] for coalesced access
    // Thread idx maps to: c * rows + r
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    
    int r = idx % rows; // Output Index (contiguous in memory)
    int c = idx / rows; // Input Index
    
    int vote = 0;
    
    for(int p=0; p < POPULATION_SIZE/2; p++) {
        int fit = fitnesses[p];
        
        // Use pair_idx for seed consistency with forward pass
        uint32_t s = step_seed + p + seed_base_add;
        int8_t a = noise_from_hash(s + offset_A, r);
        int8_t b = noise_from_hash(s + offset_B, c);
        
        if (fit == 0) continue;
        
        vote += fit * (int)a * (int)b;
    }
    
    int8_t w_curr = W[c * rows + r]; // Correct indexing for transposed layout

    if(vote > UPDATE_THRESHOLD && w_curr < MAX_VAL) {
        w_curr++;
        atomicAdd(&d_debug_updates[0], 1);
    }
    else if(vote < -UPDATE_THRESHOLD && w_curr > MIN_VAL) {
        w_curr--;
        atomicAdd(&d_debug_updates[1], 1);
    }
    W[c * rows + r] = w_curr; // Correct indexing for transposed layout
}

int main() {
    srand(time(NULL));
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    
    init_tables();
    cudaMemcpyToSymbol(d_EXP2_TABLE, h_EXP2_TABLE, 256*sizeof(int32_t));
    
    Dataset ds = {0,0};
    FILE *f = fopen("input.txt", "rb");
    if(!f) { printf("Error: input.txt not found!\n"); exit(1); }
    fseek(f,0,SEEK_END); ds.length=ftell(f); fseek(f,0,SEEK_SET); 
    ds.data=(uint8_t*)malloc(ds.length); 
    if(!fread(ds.data,1,ds.length,f)) { printf("Error reading input.txt\n"); exit(1); }
    fclose(f); 

    EggModel *h_model = (EggModel*)malloc(sizeof(EggModel));
    init_model(h_model); 
    
    EggModel *d_model;
    CHECK_CUDA(cudaMalloc(&d_model, sizeof(EggModel)));
    CHECK_CUDA(cudaMemcpy(d_model, h_model, sizeof(EggModel), cudaMemcpyHostToDevice));
    
    uint8_t *d_dataset;
    CHECK_CUDA(cudaMalloc(&d_dataset, ds.length));
    CHECK_CUDA(cudaMemcpy(d_dataset, ds.data, ds.length, cudaMemcpyHostToDevice));
    
    int32_t *d_accum_loss;
    CHECK_CUDA(cudaMalloc(&d_accum_loss, POPULATION_SIZE * sizeof(int32_t)));
    int32_t *h_accum_loss = (int32_t*)malloc(POPULATION_SIZE * sizeof(int32_t));
    
    int32_t *h_fitnesses = (int32_t*)malloc((POPULATION_SIZE/2) * sizeof(int32_t));
    int32_t *d_fitnesses;
    CHECK_CUDA(cudaMalloc(&d_fitnesses, (POPULATION_SIZE/2) * sizeof(int32_t)));
    
    // Persistent Population States
    int8_t *d_pop_states;
    size_t state_size = POPULATION_SIZE * N_LAYERS * HIDDEN_DIM;
    CHECK_CUDA(cudaMalloc(&d_pop_states, state_size));
    CHECK_CUDA(cudaMemset(d_pop_states, 0, state_size));
    
    printf("Starting EGGROLL CUDA Training (Batch=%d)...\n", BATCH);
    long max_steps = (ds.length - 1) / SEQ_LEN;
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    unsigned long total_tokens = 0;

    for(long step=0; step<max_steps; step++) {
        uint32_t seed = (uint32_t)time(NULL) ^ (step * 0x9e3779b9);
        int start_idx = step * SEQ_LEN;
        
        // 1. Forward Pass (Population)
        // Grid: POP_SIZE/BATCH (2 blocks if POP=16, BATCH=8)
        train_sequence_kernel<<<POPULATION_SIZE / BATCH, HIDDEN_DIM, BATCH * 2048>>>(
            d_dataset, ds.length, start_idx, d_model, d_pop_states, d_accum_loss, seed
        );
        cudaDeviceSynchronize();
        
        // 2. Compute Fitness on Host
        CHECK_CUDA(cudaMemcpy(h_accum_loss, d_accum_loss, POPULATION_SIZE * sizeof(int32_t), cudaMemcpyDeviceToHost));
        
        int fit_sum = 0;
        for(int p=0; p < POPULATION_SIZE/2; p++) {
             int32_t loss_pos = h_accum_loss[2*p];
             int32_t loss_neg = h_accum_loss[2*p+1];
             if (loss_pos < loss_neg) h_fitnesses[p] = 1;
             else if (loss_neg < loss_pos) h_fitnesses[p] = -1;
             else h_fitnesses[p] = 0;
             fit_sum += abs(h_fitnesses[p]);
        }
        CHECK_CUDA(cudaMemcpy(d_fitnesses, h_fitnesses, (POPULATION_SIZE/2) * sizeof(int32_t), cudaMemcpyHostToDevice));

        if (step % 10 == 0) {
            printf("[Debug] Step %ld Pair 0: Pos=%d Neg=%d Fit=%d | Total NonZero Fits: %d\n", 
                step, h_accum_loss[0], h_accum_loss[1], h_fitnesses[0], fit_sum);
        }
        
        // Reset debug counters
        int32_t zeros[2] = {0, 0};
        CHECK_CUDA(cudaMemcpyToSymbol(d_debug_updates, zeros, 2*sizeof(int32_t)));

        // 3. Update Weights (Kernels)
        for(int l=0; l<N_LAYERS; l++) {
             int seed_base = l * 100;
             // GRU 0..3
             size_t gru_size = HIDDEN_DIM * HIDDEN_DIM * sizeof(int8_t);
             long off_g = offsetof(EggModel, gru_weights) + (l * 4 * HIDDEN_DIM * HIDDEN_DIM);
             
             update_matrix_kernel<<< (HIDDEN_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                 (int8_t*)d_model + off_g + 0*gru_size, HIDDEN_DIM, HIDDEN_DIM, 1, 1+HIDDEN_DIM, seed_base, d_fitnesses, seed
             );
             update_matrix_kernel<<< (HIDDEN_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                 (int8_t*)d_model + off_g + 1*gru_size, HIDDEN_DIM, HIDDEN_DIM, 2, 2+HIDDEN_DIM, seed_base, d_fitnesses, seed
             );
             update_matrix_kernel<<< (HIDDEN_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                 (int8_t*)d_model + off_g + 2*gru_size, HIDDEN_DIM, HIDDEN_DIM, 3, 3+HIDDEN_DIM, seed_base, d_fitnesses, seed
             );
             update_matrix_kernel<<< (HIDDEN_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                 (int8_t*)d_model + off_g + 3*gru_size, HIDDEN_DIM, HIDDEN_DIM, 4, 4+HIDDEN_DIM, seed_base, d_fitnesses, seed
             );
             
             size_t mlp_exp_size = HIDDEN_DIM * (HIDDEN_DIM*4) * sizeof(int8_t);
             long off_m = offsetof(EggModel, mlp_weights) + (l * 2 * mlp_exp_size);
             
             // MLP 1 (Expand): 2048x512. A_off=0, B_off=2048. Seed=Base+5.
             update_matrix_kernel<<< (2048*512 + 511)/512, 512 >>>(
                 (int8_t*)d_model + off_m + 0, 2048, 512, 0, 2048, seed_base+5, d_fitnesses, seed
             );
             
             // MLP 2 (Project): 512x2048. A_off=0, B_off=512. Seed=Base+6.
             update_matrix_kernel<<< (512*2048 + 511)/512, 512 >>>(
                 (int8_t*)d_model + off_m + mlp_exp_size, 512, 2048, 0, 512, seed_base+6, d_fitnesses, seed
             );
        }
        
        // Head
        // Weights: [512][256]. A_off=0, B_off=256. Seed=999.
        long off_head = offsetof(EggModel, head);
        update_matrix_kernel<<< (256*512 + 511)/512, 512 >>>(
            (int8_t*)d_model + off_head, 256, 512, 0, 256, 999, d_fitnesses, seed
        );
        cudaDeviceSynchronize();

        // Reporting
        clock_gettime(CLOCK_MONOTONIC, &end);
        double t = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1e9;
        total_tokens += POPULATION_SIZE * SEQ_LEN;
        
        if (step % 1 == 0) {
            int32_t h_updates[2];
            CHECK_CUDA(cudaMemcpyFromSymbol(h_updates, d_debug_updates, 2*sizeof(int32_t), 0, cudaMemcpyDeviceToHost));
            
            // Measure Model Transfer
            struct timespec t_tr_start, t_tr_end;
            clock_gettime(CLOCK_MONOTONIC, &t_tr_start);
            
            CHECK_CUDA(cudaMemcpy(h_model, d_model, sizeof(EggModel), cudaMemcpyDeviceToHost));
            
            clock_gettime(CLOCK_MONOTONIC, &t_tr_end);
            double transfer_ms = ((t_tr_end.tv_sec - t_tr_start.tv_sec) * 1000.0) + 
                                 ((t_tr_end.tv_nsec - t_tr_start.tv_nsec) / 1e6);
            double transfer_mb = (double)sizeof(EggModel) / (1024.0 * 1024.0);

            // Weight Stats
            double w_sum = 0;
            long w_count = HIDDEN_DIM * HIDDEN_DIM;
            for(int i=0; i<w_count; i++) w_sum += abs(h_model->gru_weights[0][0][i]);
            double w_mean = w_sum / w_count;

            double sample_ms = 0.0;
            if (step % 1 == 0) {
               // Sampling
               struct timespec t_samp_start, t_samp_end;
               clock_gettime(CLOCK_MONOTONIC, &t_samp_start);
               
               sample_model_host(h_model, &ds.data[start_idx], 30, 50);
               
               clock_gettime(CLOCK_MONOTONIC, &t_samp_end);
               sample_ms = ((t_samp_end.tv_sec - t_samp_start.tv_sec) * 1000.0) + 
                           ((t_samp_end.tv_nsec - t_samp_start.tv_nsec) / 1e6);
            }
            
            double avg_loss = 0;
            for(int i=0; i<POPULATION_SIZE; i++) avg_loss += h_accum_loss[i];
            avg_loss /= (double)(POPULATION_SIZE); 
            double loss_per_token = avg_loss / (SEQ_LEN * (1 << FIXED_POINT));
            
            printf("\nStep %ld | Loss: %.4f | Up+: %d Up-: %d | W_Mean: %.4f | Tok/s: %.2f\n", 
               step, loss_per_token, h_updates[0], h_updates[1], w_mean, (double)(POPULATION_SIZE * SEQ_LEN * 1) / t); 
            
            printf("[Transfer: %.2f MB in %.2f ms] [CPU Sample: %.2f ms]\n", 
                   transfer_mb, transfer_ms, sample_ms);

            clock_gettime(CLOCK_MONOTONIC, &start); 
        }
    }
    return 0;
}
