#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// --- Configuration ---
#define SM_CORES 46
#define VOCAB_SIZE 256
#define HIDDEN_DIM (SM_CORES * 8)
#define N_LAYERS 4
#define SEQ_LEN 4096
#define BATCH 8 // Streaming batch size per block
#define POPULATION_SIZE (SM_CORES * BATCH)
#define SHARED_STRIDE (HIDDEN_DIM * 4)
#define FIXED_POINT 4
#define SIGMA_SHIFT 4
#define UPDATE_THRESHOLD 270
#define MAX_VAL 127
#define MIN_VAL -127

// --- Seed Offsets for RNG Consistency ---
#define SEED_OFF_GRU_M1 1
#define SEED_OFF_GRU_M2 2
#define SEED_OFF_GRU_M3 3
#define SEED_OFF_GRU_M4 4
#define SEED_OFF_MLP_EXP 5
#define SEED_OFF_MLP_PROJ 6
#define SEED_OFF_HEAD 999

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

// Transposed Model for Coalesced Access.
// Mapped as W[input_idx * output_dim + output_idx] (effectively Column-Major across threads).
typedef struct {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM]; 
    int8_t gru_weights[N_LAYERS][4][HIDDEN_DIM * HIDDEN_DIM]; 
    int8_t gru_biases[N_LAYERS][2][HIDDEN_DIM]; 
    int8_t mlp_weights[N_LAYERS][2][HIDDEN_DIM * (HIDDEN_DIM * 4)]; 
    int8_t head[HIDDEN_DIM * VOCAB_SIZE]; 
    int8_t ln_weights[N_LAYERS][2][HIDDEN_DIM];
    int8_t ln_out[HIDDEN_DIM];
} EggModel;

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

    // Head Generation
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) temp->head[i] = gen_noise_host(&rng);
    transpose_matrix(model->head, temp->head, VOCAB_SIZE, HIDDEN_DIM);
    
    for(int l=0; l<N_LAYERS; l++) {
        for(int g=0; g<4; g++) {
            // Generate [HIDDEN][HIDDEN] logic
            for(int i=0; i<HIDDEN_DIM*HIDDEN_DIM; i++) temp->gru_weights[0][0][i] = gen_noise_host(&rng);
            // Transpose to [Input][Output]
            transpose_matrix(model->gru_weights[l][g], temp->gru_weights[0][0], HIDDEN_DIM, HIDDEN_DIM);
        }
        for(int m=0; m<2; m++) for(int i=0; i<HIDDEN_DIM; i++) model->gru_biases[l][m][i] = 0; 
    }
    
    for(int l=0; l<N_LAYERS; l++) {
        // MLP 1: Expand. In: HIDDEN. Out: 4*HIDDEN.
        // Weights: [4*HIDDEN][HIDDEN] -> Transpose [HIDDEN][4*HIDDEN]
        int dim_expand = HIDDEN_DIM * (HIDDEN_DIM * 4);
        for(int i=0; i<dim_expand; i++) temp->mlp_weights[0][0][i] = gen_noise_host(&rng);
        transpose_matrix(model->mlp_weights[l][0], temp->mlp_weights[0][0], HIDDEN_DIM*4, HIDDEN_DIM);

        // MLP 2: Project. In: 4*HIDDEN. Out: HIDDEN.
        // Weights: [HIDDEN][4*HIDDEN] -> Transpose [4*HIDDEN][HIDDEN]
        for(int i=0; i<dim_expand; i++) temp->mlp_weights[0][0][i] = gen_noise_host(&rng);
        transpose_matrix(model->mlp_weights[l][1], temp->mlp_weights[0][0], HIDDEN_DIM, HIDDEN_DIM*4);
        
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][0][i] = 16;
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][1][i] = 16;
    }
    
    for(int i=0; i<HIDDEN_DIM; i++) model->ln_out[i] = 16;
    
    free(temp);
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
    __syncthreads(); 
    static __shared__ int32_t shared[32]; 
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < (blockDim.x / 32)) ? shared[threadIdx.x] : 0;
    
    if (wid == 0) val = warpReduceSum(val);
    
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    
    return shared[0];
}

extern __shared__ int8_t s_mem[]; // Shared memory: [BATCH][SHARED_STRIDE]

__global__ void generate_sequence_kernel(
    const EggModel * __restrict__ model,
    uint32_t seed,
    const uint8_t *seed_text,
    int seed_len,
    int gen_len,
    uint8_t *output
) {
    int tid = threadIdx.x;
    int8_t *s_ptr = s_mem;
    
    // Local hidden state
    int8_t h[N_LAYERS]; 
    for(int l=0; l<N_LAYERS; l++) h[l] = 0;
    
    int total_len = seed_len + gen_len;
    uint8_t current_token = 0; 
    
    for(int t=0; t < total_len; t++) {
        if (t < seed_len) current_token = seed_text[t];
        
        // 1. Embed
        if(tid < HIDDEN_DIM) s_ptr[tid] = model->embedding[current_token * HIDDEN_DIM + tid];
        __syncthreads();
        
        // 2. Layers
        for(int l=0; l<N_LAYERS; l++) {
             int8_t residual = s_ptr[tid];
             
             // LN 1
             int32_t sum = blockReduceSum(abs((int32_t)s_ptr[tid]));
             if(!sum) sum=1; int32_t mean = sum/HIDDEN_DIM; if(!mean) mean=1;
             s_ptr[tid] = clip(((int32_t)s_ptr[tid] * model->ln_weights[l][0][tid]) / mean);
             __syncthreads();
             
             // GRU Phase 1: W0*x -> Buf1, W1*h -> Buf2
             s_ptr[HIDDEN_DIM + tid] = h[l]; // Save H to Shared
             __syncthreads();
             
             const int8_t *w0 = &model->gru_weights[l][0][0];
             const int8_t *w1 = &model->gru_weights[l][1][0];
             int32_t acc0 = 0, acc1 = 0;
             
             for(int k=0; k<HIDDEN_DIM; k++) {
                 acc0 += (int32_t)s_ptr[k] * w0[k*HIDDEN_DIM + tid];
                 acc1 += (int32_t)s_ptr[HIDDEN_DIM + k] * w1[k*HIDDEN_DIM + tid];
             }
             __syncthreads(); 
             s_ptr[HIDDEN_DIM + tid] = acc0 >> 8;
             s_ptr[2*HIDDEN_DIM + tid] = acc1 >> 8;
             __syncthreads();
             
             // Gates
             int8_t b1 = s_ptr[HIDDEN_DIM + tid];
             int8_t b2 = s_ptr[2*HIDDEN_DIM + tid];
             int8_t ft = clip(b1 + b2 + model->gru_biases[l][0][tid]);
             
             // Store Gated to Buf2
             s_ptr[2*HIDDEN_DIM + tid] = (int8_t)(((int32_t)(ft + 127) * h[l]) >> 8);
             __syncthreads();
             
             // GRU Phase 2: W2*x -> Buf1, W3*gated -> Buf3
             const int8_t *w2 = &model->gru_weights[l][2][0];
             const int8_t *w3 = &model->gru_weights[l][3][0];
             
             acc0 = 0; acc1 = 0;
             for(int k=0; k<HIDDEN_DIM; k++) {
                 acc0 += (int32_t)s_ptr[k] * w2[k*HIDDEN_DIM + tid];
                 acc1 += (int32_t)s_ptr[2*HIDDEN_DIM + k] * w3[k*HIDDEN_DIM + tid];
             }
             __syncthreads(); 
             s_ptr[HIDDEN_DIM + tid] = acc0 >> 8;
             s_ptr[3*HIDDEN_DIM + tid] = acc1 >> 8;
             __syncthreads(); 
             
             // Update H
             b1 = s_ptr[HIDDEN_DIM + tid];
             b2 = s_ptr[3*HIDDEN_DIM + tid];
             int8_t ht = clip(b1 + b2 + model->gru_biases[l][1][tid]);
             int32_t diff = ht - h[l];
             int32_t update = ((int32_t)(ft + 127) * diff) >> 8;
             h[l] = clip(h[l] + update);
             
             // New X (Pre-LN for MLP residual)
             s_ptr[tid] = clip((int32_t)h[l] + residual);
             __syncthreads();
             
             int8_t mlp_in_resid = s_ptr[tid];
             
             // MLP LN
             sum = blockReduceSum(abs((int32_t)s_ptr[tid]));
             if(!sum) sum=1; mean = sum/HIDDEN_DIM; if(!mean) mean=1;
             s_ptr[tid] = clip(((int32_t)s_ptr[tid] * model->ln_weights[l][1][tid]) / mean);
             __syncthreads();
             
             // MLP Expand: HIDDEN->4*HIDDEN.
             const int8_t *w_exp = &model->mlp_weights[l][0][0];
             int8_t exp_res[4];
             for(int i=0; i<4; i++) {
                 int out_idx = tid + i*HIDDEN_DIM;
                 int32_t acc = 0;
                 for(int k=0; k<HIDDEN_DIM; k++) {
                     acc += (int32_t)s_ptr[k] * w_exp[k*(4*HIDDEN_DIM) + out_idx];
                 }
                 exp_res[i] = clip(acc >> 8);
             }
             __syncthreads();
             // Store safely
             for(int i=0; i<4; i++) s_ptr[tid + i*HIDDEN_DIM] = exp_res[i];
             __syncthreads();
             
             // MLP Project: 4*HIDDEN->HIDDEN
             const int8_t *w_proj = &model->mlp_weights[l][1][0];
             int32_t acc = 0;
             for(int k=0; k<(4*HIDDEN_DIM); k++) {
                 acc += (int32_t)s_ptr[k] * w_proj[k*HIDDEN_DIM + tid];
             }
             s_ptr[tid] = clip((acc >> 9) + mlp_in_resid);
             __syncthreads();
        }
        
        // 3. Head
        int32_t sum = blockReduceSum(abs((int32_t)s_ptr[tid]));
        if(!sum) sum=1; int32_t mean = sum/HIDDEN_DIM; if(!mean) mean=1;
        s_ptr[tid] = clip(((int32_t)s_ptr[tid] * model->ln_out[tid]) / mean);
        __syncthreads();
        
        // Logits: HIDDEN->VOCAB
        const int8_t *w_head = &model->head[0];
        if (tid < VOCAB_SIZE) {
            int32_t acc = 0;
            for(int k=0; k<HIDDEN_DIM; k++) {
                acc += (int32_t)s_ptr[k] * w_head[k*VOCAB_SIZE + tid]; 
            }
            s_ptr[HIDDEN_DIM + tid] = acc >> 8; 
        }
        __syncthreads();
        
        // 4. Sample
        if (t >= seed_len) {
            int32_t val = 0;
            if (tid < VOCAB_SIZE) {
                int idx = (int32_t)s_ptr[HIDDEN_DIM + tid] + 128;
                idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
                val = d_EXP2_TABLE[idx];
            }
            int32_t sum_exp = blockReduceSum(val); 
            
            uint32_t s = seed + t * 222;
            uint32_t r_val = hash_rng(s, 0);
            int32_t r_threshold = (sum_exp > 0) ? (r_val % sum_exp) : 0;
            
            if(tid == 0) {
                int32_t running = 0;
                int selected = VOCAB_SIZE - 1;
                for(int i=0; i<VOCAB_SIZE; i++) {
                    int idx = (int32_t)s_ptr[HIDDEN_DIM + i] + 128;
                    idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
                    running += d_EXP2_TABLE[idx];
                    if (running > r_threshold) {
                        selected = i;
                        break;
                    }
                }
                current_token = (uint8_t)selected;
                if (t >= seed_len) output[t - seed_len] = current_token;
                s_ptr[0] = current_token;
            }
             __syncthreads();
             current_token = (uint8_t)s_ptr[0];
        }
    }
}

__global__ void __launch_bounds__(1024) train_sequence_kernel(
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
    
    // Local State
    int8_t h[BATCH][N_LAYERS];
    
    // Load State
    for(int b=0; b<BATCH; b++) {
        int p_idx = block_p_idx * BATCH + b;
        for(int l=0; l<N_LAYERS; l++) {
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
                s_ptr[b * SHARED_STRIDE + tid] = model->embedding[input_token * HIDDEN_DIM + tid];
            }
        }
        __syncthreads();
        
        // 2. Layers
        for (int l = 0; l < N_LAYERS; l++) {
            int8_t gru_resid[BATCH];

            // LN 1
            for(int b=0; b<BATCH; b++) {
                int8_t *sx_b = &s_ptr[b * SHARED_STRIDE];
                int32_t val = sx_b[tid];
                gru_resid[b] = (int8_t)val; // Save Pre-LN residual
                
                int32_t sum = blockReduceSum(abs(val));
                if(!sum) sum=1; int32_t mean = sum/HIDDEN_DIM; if(!mean) mean=1;
                val = clip(((int32_t)val * model->ln_weights[l][0][tid]) / mean);
                sx_b[tid] = val; // x_in
            }
            __syncthreads();
            
            // Save H to shared (Offset HIDDEN_DIM) for MatMul2
            for(int b=0; b<BATCH; b++) {
                 if(tid < HIDDEN_DIM) s_ptr[b*SHARED_STRIDE + HIDDEN_DIM + tid] = h[b][l];
            }
            __syncthreads();

            // Rank-1 Precalc (xB)
            int32_t xB_m1[BATCH], xB_m2[BATCH], xB_m3[BATCH], xB_m4[BATCH];
            
            for(int b=0; b<BATCH; b++) {
                int p_idx = block_p_idx * BATCH + b;
                int pair_idx = p_idx / 2;
                uint32_t seed = (step_seed + pair_idx) + (l * 100); 
                
                int8_t b1 = noise_from_hash(seed + SEED_OFF_GRU_M1 + HIDDEN_DIM, tid);
                xB_m1[b] = blockReduceSum((int32_t)s_ptr[b*SHARED_STRIDE+tid] * b1);
                
                int8_t b2 = noise_from_hash(seed + SEED_OFF_GRU_M2 + HIDDEN_DIM, tid);
                xB_m2[b] = blockReduceSum((int32_t)s_ptr[b*SHARED_STRIDE+HIDDEN_DIM+tid] * b2);
            }

            // MatMul 1 & 2
            int32_t dot1[BATCH]; for(int b=0; b<BATCH; b++) dot1[b]=0;
            int32_t dot2[BATCH]; for(int b=0; b<BATCH; b++) dot2[b]=0;
            
            const int8_t *w1 = &model->gru_weights[l][0][0];
            const int8_t *w2 = &model->gru_weights[l][1][0];
            
            for(int k=0; k<HIDDEN_DIM; k++) {
                int8_t v1 = w1[k*HIDDEN_DIM + tid];
                int8_t v2 = w2[k*HIDDEN_DIM + tid];
                for(int b=0; b<BATCH; b++) {
                    dot1[b] += (int32_t)s_ptr[b*SHARED_STRIDE + k] * v1;
                    dot2[b] += (int32_t)s_ptr[b*SHARED_STRIDE + HIDDEN_DIM + k] * v2;
                }
            }
            
            int8_t ft_reg[BATCH];
            int8_t gated_reg[BATCH];
            
            for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int ns = (p_idx % 2 == 0) ? 1 : -1;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed + pair_idx) + (l * 100);
                 
                 int8_t a1 = noise_from_hash(seed + SEED_OFF_GRU_M1, tid);
                 if(ns!=0) dot1[b] += ((xB_m1[b] * (int32_t)a1) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int8_t a2 = noise_from_hash(seed + SEED_OFF_GRU_M2, tid);
                 if(ns!=0) dot2[b] += ((xB_m2[b] * (int32_t)a2) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int8_t r1 = dot1[b] >> 8;
                 int8_t r2 = dot2[b] >> 8;
                 ft_reg[b] = clip(r1 + r2 + model->gru_biases[l][0][tid]);
                 gated_reg[b] = (int8_t)(((int32_t)(ft_reg[b] + 127) * h[b][l]) >> 8);
            }
            
            // Save Gated to Shared for M4
            __syncthreads();
            for(int b=0; b<BATCH; b++) {
                if(tid < HIDDEN_DIM) s_ptr[b*SHARED_STRIDE + 2*HIDDEN_DIM + tid] = gated_reg[b];
            }
            __syncthreads();
            
            // Rank-1 M3 & M4
             for(int b=0; b<BATCH; b++) {
                int p_idx = block_p_idx * BATCH + b;
                int pair_idx = p_idx / 2;
                uint32_t seed = (step_seed + pair_idx) + (l * 100);
                
                int8_t b3 = noise_from_hash(seed + SEED_OFF_GRU_M3 + HIDDEN_DIM, tid);
                xB_m3[b] = blockReduceSum((int32_t)s_ptr[b*SHARED_STRIDE+tid] * b3);
                
                int8_t b4 = noise_from_hash(seed + SEED_OFF_GRU_M4 + HIDDEN_DIM, tid);
                xB_m4[b] = blockReduceSum((int32_t)s_ptr[b*SHARED_STRIDE + 2*HIDDEN_DIM + tid] * b4);
            }
            
            for(int b=0; b<BATCH; b++) { dot1[b]=0; dot2[b]=0; } 
            const int8_t *w3 = &model->gru_weights[l][2][0];
            const int8_t *w4 = &model->gru_weights[l][3][0];
            
            for(int k=0; k<HIDDEN_DIM; k++) {
                int8_t v3 = w3[k*HIDDEN_DIM + tid];
                int8_t v4 = w4[k*HIDDEN_DIM + tid];
                for(int b=0; b<BATCH; b++) {
                    dot1[b] += (int32_t)s_ptr[b*SHARED_STRIDE + k] * v3;
                    dot2[b] += (int32_t)s_ptr[b*SHARED_STRIDE + 2*HIDDEN_DIM + k] * v4;
                }
            }
            
            // Update H
            for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int ns = (p_idx % 2 == 0) ? 1 : -1;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed + pair_idx) + (l * 100);
                 
                 int8_t a3 = noise_from_hash(seed + SEED_OFF_GRU_M3, tid);
                 if(ns!=0) dot1[b] += ((xB_m3[b] * (int32_t)a3) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int8_t a4 = noise_from_hash(seed + SEED_OFF_GRU_M4, tid);
                 if(ns!=0) dot2[b] += ((xB_m4[b] * (int32_t)a4) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int8_t ht = clip((dot1[b] >> 8) + (dot2[b] >> 8) + model->gru_biases[l][1][tid]);
                 
                 int32_t diff = ht - h[b][l];
                 int32_t update = ((int32_t)(ft_reg[b] + 127) * diff) >> 8;
                 h[b][l] = clip(h[b][l] + update);
            }
            
            // Residual & New X
            for(int b=0; b<BATCH; b++) {
                int8_t new_x = clip((int32_t)h[b][l] + gru_resid[b]); 
                s_ptr[b*SHARED_STRIDE + tid] = new_x; 
            }
            __syncthreads(); 
            
            // Save Pre-LN Residual for MLP
            int8_t mlp_resid[BATCH];
            for(int b=0; b<BATCH; b++) mlp_resid[b] = s_ptr[b*SHARED_STRIDE + tid];

            // MLP LN 2
            for(int b=0; b<BATCH; b++) {
                int32_t val = s_ptr[b*SHARED_STRIDE + tid];
                int32_t sum = blockReduceSum(abs(val));
                if(!sum) sum=1; int32_t mean = sum/HIDDEN_DIM; if(!mean) mean=1;
                val = clip(((int32_t)val * model->ln_weights[l][1][tid]) / mean);
                s_ptr[b*SHARED_STRIDE + tid] = val;
            }
            __syncthreads(); 

            // MLP Expand (HIDDEN -> 4*HIDDEN)
            int8_t mlp_res[BATCH][4];
            
            // 1. Rank-1 xB for MLP1
            int32_t xB_mlp1[BATCH];
            for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed+pair_idx) + (l * 100) + SEED_OFF_MLP_EXP;
                 int8_t b_val = noise_from_hash(seed + SHARED_STRIDE, tid); // Offset B
                 xB_mlp1[b] = blockReduceSum((int32_t)s_ptr[b*SHARED_STRIDE+tid] * b_val);
            }
            
            // 2. Init accums
            int32_t acc_mlp[BATCH][4]; 
            for(int b=0; b<BATCH; b++) for(int s=0; s<4; s++) acc_mlp[b][s] = 0;
            
            const int8_t *w_mlp1 = &model->mlp_weights[l][0][0];
             for(int k=0; k<HIDDEN_DIM; k++) {
                 for(int sub=0; sub<4; sub++) {
                      int out_idx = tid + sub * HIDDEN_DIM;
                      int8_t w_val = w_mlp1[k*(HIDDEN_DIM*4) + out_idx];
                      for(int b=0; b<BATCH; b++) {
                          acc_mlp[b][sub] += (int32_t)s_ptr[b*SHARED_STRIDE + k] * w_val;
                      }
                 }
             }
             
             // 3. Noise & Store
             for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int ns = (p_idx % 2 == 0) ? 1 : -1;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed+pair_idx) + (l * 100) + SEED_OFF_MLP_EXP;
                 
                 for(int sub=0; sub<4; sub++) {
                     int out_idx = tid + sub * HIDDEN_DIM;
                     int8_t a_val = noise_from_hash(seed, out_idx);
                     if(ns!=0) acc_mlp[b][sub] += ((xB_mlp1[b] * (int32_t)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                     mlp_res[b][sub] = clip(acc_mlp[b][sub] >> 8);
                 }
             }
             
             // MLP Project Input (Distributed)
             __syncthreads();
             for(int b=0; b<BATCH; b++) {
                 for(int sub=0; sub<4; sub++) {
                     s_ptr[b*SHARED_STRIDE + tid + sub*HIDDEN_DIM] = mlp_res[b][sub];
                 }
             }
             __syncthreads();
             
             // xB for M2 (Input 4*HIDDEN)
             int32_t xB_mlp2[BATCH];
             for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed+pair_idx) + (l * 100) + SEED_OFF_MLP_PROJ;
                 
                 int32_t local_sum = 0;
                 for(int sub=0; sub<4; sub++) {
                     int in_idx = tid + sub*HIDDEN_DIM;
                     int8_t b_val = noise_from_hash(seed + HIDDEN_DIM, in_idx); // Offset B
                     local_sum += (int32_t)s_ptr[b*SHARED_STRIDE + in_idx] * b_val;
                 }
                 xB_mlp2[b] = blockReduceSum(local_sum);
             }
             
             int32_t acc_proj[BATCH]; for(int b=0; b<BATCH; b++) acc_proj[b] = 0;
             const int8_t *w_mlp2 = &model->mlp_weights[l][1][0];
             for(int k=0; k<HIDDEN_DIM*4; k++) {
                 int8_t w_val = w_mlp2[k*HIDDEN_DIM + tid];
                 for(int b=0; b<BATCH; b++) {
                     acc_proj[b] += (int32_t)s_ptr[b*SHARED_STRIDE + k] * w_val;
                 }
             }

            // Apply Result
             for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int ns = (p_idx % 2 == 0) ? 1 : -1;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed+pair_idx) + (l * 100) + SEED_OFF_MLP_PROJ;
                 
                 int8_t a_val = noise_from_hash(seed, tid);
                 if(ns!=0) acc_proj[b] += ((xB_mlp2[b] * (int32_t)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int32_t res = acc_proj[b] >> 9; // Shift 9 for MLP project
                 s_ptr[b*SHARED_STRIDE + tid] = clip(res + mlp_resid[b]);
             }
             __syncthreads();
         
        } // Layers Loop
        
        // 3. Head LN
        for(int b=0; b<BATCH; b++) {
            int32_t val = s_ptr[b * SHARED_STRIDE + tid];
            int32_t sum = blockReduceSum(abs(val));
            if(!sum) sum=1; int32_t mean = sum/HIDDEN_DIM; if(!mean) mean=1;
            s_ptr[b * SHARED_STRIDE + tid] = clip(((int32_t)val * model->ln_out[tid]) / mean);
        }
        __syncthreads();
        
        // 4. Head Dense
        int32_t logits[BATCH]; 
         int32_t xB_head[BATCH];
         for(int b=0; b<BATCH; b++) {
             int p_idx = block_p_idx * BATCH + b;
             int pair_idx = p_idx / 2;
             uint32_t seed = (step_seed+pair_idx) + SEED_OFF_HEAD;
             int8_t b_val = noise_from_hash(seed + VOCAB_SIZE, tid); 
             xB_head[b] = blockReduceSum((int32_t)s_ptr[b*SHARED_STRIDE+tid] * b_val);
         }
         
         const int8_t *w_head = &model->head[0];
         if(tid < VOCAB_SIZE) {
            int32_t acc[BATCH]; for(int b=0; b<BATCH; b++) acc[b]=0;
             
            for(int k=0; k<HIDDEN_DIM; k++) {
                int8_t w = w_head[k*VOCAB_SIZE + tid];
                for(int b=0; b<BATCH; b++) acc[b] += (int32_t)s_ptr[b*SHARED_STRIDE + k] * w;
            }
            
            for(int b=0; b<BATCH; b++) {
                 int p_idx = block_p_idx * BATCH + b;
                 int ns = (p_idx % 2 == 0) ? 1 : -1;
                 int pair_idx = p_idx / 2;
                 uint32_t seed = (step_seed+pair_idx) + SEED_OFF_HEAD;
                 
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
        
        // Use pair_idx for seed consistency
        uint32_t s = step_seed + p + seed_base_add;
        int8_t a = noise_from_hash(s + offset_A, r);
        int8_t b = noise_from_hash(s + offset_B, c);
        
        if (fit == 0) continue;
        
        vote += fit * (int)a * (int)b;
    }
    
    int8_t w_curr = W[c * rows + r]; 

    if(vote > UPDATE_THRESHOLD && w_curr < MAX_VAL) {
        w_curr++;
        atomicAdd(&d_debug_updates[0], 1);
    }
    else if(vote < -UPDATE_THRESHOLD && w_curr > MIN_VAL) {
        w_curr--;
        atomicAdd(&d_debug_updates[1], 1);
    }
    W[c * rows + r] = w_curr; 
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
        train_sequence_kernel<<<POPULATION_SIZE / BATCH, HIDDEN_DIM, BATCH * SHARED_STRIDE>>>(
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
        dim3 block(512);
        for(int l=0; l<N_LAYERS; l++) {
             int seed_base = l * 100;
             
             // GRU 0..3
             size_t gru_size = HIDDEN_DIM * HIDDEN_DIM * sizeof(int8_t);
             long off_g = offsetof(EggModel, gru_weights) + (l * 4 * HIDDEN_DIM * HIDDEN_DIM);
             
             for(int g=0; g<4; g++) {
                 int seed_off = SEED_OFF_GRU_M1 + g; // M1..M4 are 1..4
                 update_matrix_kernel<<< (HIDDEN_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                     (int8_t*)d_model + off_g + g*gru_size, 
                     HIDDEN_DIM, HIDDEN_DIM, 
                     seed_off, seed_off + HIDDEN_DIM, // Offset A (Out), Offset B (In)
                     seed_base, d_fitnesses, seed
                 );
             }
             
             size_t mlp_exp_size = HIDDEN_DIM * (HIDDEN_DIM*4) * sizeof(int8_t);
             long off_m = offsetof(EggModel, mlp_weights) + (l * 2 * mlp_exp_size);
             
             // MLP 1 (Expand): Input=HIDDEN, Output=4*HIDDEN
             // Rows (Out) = 4*H, Cols (In) = H
             int exp_rows = 4*HIDDEN_DIM;
             int exp_cols = HIDDEN_DIM;
             update_matrix_kernel<<< (exp_rows*exp_cols + 511)/512, 512 >>>(
                 (int8_t*)d_model + off_m + 0, exp_rows, exp_cols, 
                 0, SHARED_STRIDE, // Offset A (Out 0), Offset B (In 4H=SHARED_STRIDE)
                 seed_base + SEED_OFF_MLP_EXP, d_fitnesses, seed
             );
             
             // MLP 2 (Project): Input=4*HIDDEN, Output=HIDDEN
             // Rows (Out) = H, Cols (In) = 4*H
             update_matrix_kernel<<< (HIDDEN_DIM*(4*HIDDEN_DIM) + 511)/512, 512 >>>(
                 (int8_t*)d_model + off_m + mlp_exp_size, HIDDEN_DIM, 4*HIDDEN_DIM, 
                 0, HIDDEN_DIM, // Offset A (Out 0), Offset B (In HIDDEN)
                 seed_base + SEED_OFF_MLP_PROJ, d_fitnesses, seed
             );
        }
        
        // Head
        long off_head = offsetof(EggModel, head);
        update_matrix_kernel<<< (VOCAB_SIZE*HIDDEN_DIM + 511)/512, 512 >>>(
            (int8_t*)d_model + off_head, VOCAB_SIZE, HIDDEN_DIM, 
            0, VOCAB_SIZE, // Offset A (Out 0), Offset B (In V)
            SEED_OFF_HEAD, d_fitnesses, seed
        );
        cudaDeviceSynchronize();

        // Reporting
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_tokens += POPULATION_SIZE * SEQ_LEN;
        
        if (step % 1 == 0) { 
            int32_t h_updates[2];
            CHECK_CUDA(cudaMemcpyFromSymbol(h_updates, d_debug_updates, 2*sizeof(int32_t), 0, cudaMemcpyDeviceToHost));
            
            // --- GPU Sample ---
            static uint8_t *d_output = NULL;
            static uint8_t *h_output = NULL;
            int seed_len = 30; 
            int gen_len = 50;
            if(!d_output) {
                CHECK_CUDA(cudaMalloc(&d_output, gen_len));
                h_output = (uint8_t*)malloc(gen_len);
            }
            
            struct timespec t_samp_start, t_samp_end;
            clock_gettime(CLOCK_MONOTONIC, &t_samp_start);
            
            generate_sequence_kernel<<<1, HIDDEN_DIM, 5 * HIDDEN_DIM * sizeof(int8_t)>>>(
                d_model, seed + 12345, d_dataset + start_idx, seed_len, gen_len, d_output
            );
            CHECK_CUDA(cudaDeviceSynchronize()); 
            CHECK_CUDA(cudaMemcpy(h_output, d_output, gen_len, cudaMemcpyDeviceToHost));
            
            clock_gettime(CLOCK_MONOTONIC, &t_samp_end);
            double sample_ms = ((t_samp_end.tv_sec - t_samp_start.tv_sec) * 1000.0) + 
                               ((t_samp_end.tv_nsec - t_samp_start.tv_nsec) / 1e6);
            
            // Print result
            printf("\033[32m");
            for(int i=0; i<seed_len; i++) {
                char c = ds.data[start_idx+i];
                if(c>=32 && c<=126) printf("%c", c); else printf(".");
            }
            printf("\033[36m");
            for(int i=0; i<gen_len; i++) {
                char c = h_output[i];
                if(c>=32 && c<=126) printf("%c", c); else printf(".");
            }
            printf("\033[0m\n");

            double avg_loss = 0;
            for(int i=0; i<POPULATION_SIZE; i++) avg_loss += h_accum_loss[i];
            avg_loss /= (double)(POPULATION_SIZE); 
            double loss_per_token = avg_loss / (SEQ_LEN * (1 << FIXED_POINT));
            
            double total_t = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1e9;
            
            printf("\nStep %ld | Loss: %.4f | Up+: %d Up-: %d | GPU Sample: %.2f ms | Tok/s: %.2f\n", 
               step, loss_per_token, h_updates[0], h_updates[1], sample_ms,  total_tokens / total_t); 
        }
    }
    return 0;
}
