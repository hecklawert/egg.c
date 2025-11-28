#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <cub/cub.cuh>

volatile sig_atomic_t keep_running = 1;

void handle_sigint(int sig) {
    const char msg[] = "\n[SIGINT] Interrupt received. Stopping after current step...\n";
    write(STDOUT_FILENO, msg, sizeof(msg)-1);
    keep_running = 0;
}

// --- CONFIGURATION ---
#define HIDDEN_DIM 768
#define HEAD_DIM 64
#define N_LAYERS 4
#define SEQ_LEN 64     
#define VOCAB_SIZE 256
#define WARP_SIZE 32
#define MAX_BLOCK_THREADS 1024 
#define ALIGNED_DIM ((HIDDEN_DIM + 31) & ~31)
#define BLOCK_THREADS (ALIGNED_DIM > MAX_BLOCK_THREADS ? MAX_BLOCK_THREADS : ALIGNED_DIM)
#define N_HEADS (HIDDEN_DIM / HEAD_DIM)

// Population Sizing (Target 20GB)
#define POPULATION_SIZE 8192 

#define FIXED_POINT 4
#define SIGMA_SHIFT 4
#define SIGMA_SHIFT_VECTOR (SIGMA_SHIFT - 2)
#define MAX_VAL 127
#define MIN_VAL -127

#define SEED_OFF_EMB 0
#define SEED_OFF_POS 1
#define SEED_OFF_LN_1 100
#define SEED_OFF_Q_A 200
#define SEED_OFF_Q_B 201
#define SEED_OFF_K_A 202
#define SEED_OFF_K_B 203
#define SEED_OFF_V_A 204
#define SEED_OFF_V_B 205
#define SEED_OFF_O_A 206
#define SEED_OFF_O_B 207
#define SEED_OFF_LN_2 300
#define SEED_OFF_MLP_UP_A 400
#define SEED_OFF_MLP_UP_B 401
#define SEED_OFF_MLP_DOWN_A 402
#define SEED_OFF_MLP_DOWN_B 403
#define SEED_OFF_LN_F 900
#define SEED_OFF_HEAD 999

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA Error: %s:%d\n", cudaGetErrorString(err), __LINE__); exit(1); } }

typedef struct { uint8_t *data; long length; } Dataset;

typedef struct {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM];
    int8_t pos_emb[SEQ_LEN * HIDDEN_DIM];
    int8_t ln_1[N_LAYERS][HIDDEN_DIM];
    int8_t w_q[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    int8_t w_k[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    int8_t w_v[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    int8_t w_o[N_LAYERS][HIDDEN_DIM * HIDDEN_DIM];
    int8_t ln_2[N_LAYERS][HIDDEN_DIM];
    int8_t w_up[N_LAYERS][HIDDEN_DIM * (HIDDEN_DIM * 4)];
    int8_t w_down[N_LAYERS][(HIDDEN_DIM * 4) * HIDDEN_DIM];
    int8_t ln_f[HIDDEN_DIM];
    int8_t head[HIDDEN_DIM * VOCAB_SIZE];
} TransformerModel;

int8_t *d_kv_cache = NULL;

__constant__ int32_t d_EXP2_TABLE[256];
int32_t h_EXP2_TABLE[256];
__device__ int32_t d_debug_updates[2];

// --- HOST HELPER ---
int get_update_threshold(double loss) { return (loss > 2.0) ? 5000 : 50000; } 

double get_time_diff_ms(struct timespec start, struct timespec end) {
    return ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_nsec - start.tv_nsec) / 1e6);
}

void init_tables() {
    for(int i=0; i<256; i++) h_EXP2_TABLE[i] = (int32_t)(pow(2.0, (double)i / (1 << FIXED_POINT)) * (1 << FIXED_POINT));
}

static inline uint32_t xorshift32_host(uint32_t *state) {
    uint32_t x = *state; x ^= x << 13; x ^= x >> 17; x ^= x << 5; *state = x; return x;
}
static inline int8_t gen_noise_host(uint32_t *rng) { return (int8_t)((xorshift32_host(rng) & 1 ? 1 : -1) * ((xorshift32_host(rng) >> 1) & 15)); }
void transpose_matrix(int8_t *dst, int8_t *src, int rows, int cols) {
    for(int r=0; r<rows; r++) for(int c=0; c<cols; c++) dst[c * rows + r] = src[r * cols + c];
}
void init_model(TransformerModel *model) {
    uint32_t rng = 42;
    TransformerModel *temp = (TransformerModel*)calloc(1, sizeof(TransformerModel));
    if(!temp) exit(1);
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) temp->embedding[i] = gen_noise_host(&rng);
    transpose_matrix(model->embedding, temp->embedding, VOCAB_SIZE, HIDDEN_DIM);
    for(int i=0; i<SEQ_LEN*HIDDEN_DIM; i++) model->pos_emb[i] = gen_noise_host(&rng);
    for(int i=0; i<HIDDEN_DIM*VOCAB_SIZE; i++) temp->head[i] = gen_noise_host(&rng);
    transpose_matrix(model->head, temp->head, HIDDEN_DIM, VOCAB_SIZE);
    for(int l=0; l<N_LAYERS; l++) {
        for(int i=0; i<HIDDEN_DIM; i++) { model->ln_1[l][i]=16; model->ln_2[l][i]=16; }
        int d2 = HIDDEN_DIM*HIDDEN_DIM;
        for(int i=0; i<d2; i++) temp->w_q[l][i] = gen_noise_host(&rng); transpose_matrix(model->w_q[l], temp->w_q[l], HIDDEN_DIM, HIDDEN_DIM);
        for(int i=0; i<d2; i++) temp->w_k[l][i] = gen_noise_host(&rng); transpose_matrix(model->w_k[l], temp->w_k[l], HIDDEN_DIM, HIDDEN_DIM);
        for(int i=0; i<d2; i++) temp->w_v[l][i] = gen_noise_host(&rng); transpose_matrix(model->w_v[l], temp->w_v[l], HIDDEN_DIM, HIDDEN_DIM);
        for(int i=0; i<d2; i++) temp->w_o[l][i] = gen_noise_host(&rng); transpose_matrix(model->w_o[l], temp->w_o[l], HIDDEN_DIM, HIDDEN_DIM);
        
        for(int i=0; i<HIDDEN_DIM*(HIDDEN_DIM*4); i++) temp->w_up[l][i] = gen_noise_host(&rng); transpose_matrix(model->w_up[l], temp->w_up[l], HIDDEN_DIM, 4*HIDDEN_DIM);
        for(int i=0; i<HIDDEN_DIM*(HIDDEN_DIM*4); i++) temp->w_down[l][i] = gen_noise_host(&rng); transpose_matrix(model->w_down[l], temp->w_down[l], 4*HIDDEN_DIM, HIDDEN_DIM);
    }
    for(int i=0; i<HIDDEN_DIM; i++) model->ln_f[i]=16;
    free(temp);
}

// --- DEVICE KERNELS ---
__device__ __forceinline__ uint32_t hash_rng(uint32_t s, uint32_t idx) {
    uint32_t x = s + idx * 0x9e3779b9u; x ^= x >> 16; x *= 0x85ebca6b; x ^= x >> 13; x *= 0xc2b2ae35; x ^= x >> 16; return x;
}
__device__ __forceinline__ int8_t noise_from_hash(uint32_t s, uint32_t idx) {
    uint32_t r = hash_rng(s, idx); return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 15));
}
__device__ __forceinline__ int8_t clip(long long a) { return (a > MAX_VAL) ? MAX_VAL : ((a < MIN_VAL) ? MIN_VAL : (int8_t)a); }

extern __shared__ int8_t s_mem[]; 

__global__ void __launch_bounds__(MAX_BLOCK_THREADS) train_sequence_kernel(
    const uint8_t * __restrict__ dataset, long data_len, int start_idx,
    const TransformerModel * __restrict__ model,
    int8_t * __restrict__ global_kv_cache,
    int32_t *accum_loss, uint32_t step_seed
) {
    int p_idx = blockIdx.x; 
    if (p_idx >= POPULATION_SIZE) return;
    int tid = threadIdx.x;
    if (tid >= HIDDEN_DIM) return;

    int8_t *s_x = s_mem; 
    typedef cub::BlockReduce<long long, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ long long shared_scalar;

    long pair_idx = p_idx / 2;
    long stride = data_len / (POPULATION_SIZE / 2);
    long stream_pos = (start_idx + (pair_idx * stride)) % (data_len - SEQ_LEN);
    int ns = (p_idx % 2 == 0) ? 1 : -1;
    size_t kv_layer_stride = 2ULL * SEQ_LEN * HIDDEN_DIM;
    size_t kv_ind_offset = (size_t)p_idx * N_LAYERS * kv_layer_stride;

    long long my_loss = 0;

    for (int t = 0; t < SEQ_LEN; t++) {
        
        // 1. Embedding
        uint8_t input_token = dataset[stream_pos + t];
        uint32_t seed_emb = (step_seed + pair_idx) + SEED_OFF_EMB;
        int8_t emb = model->embedding[tid * VOCAB_SIZE + input_token];
        int8_t pos = model->pos_emb[t * HIDDEN_DIM + tid];
        int8_t a_tok = noise_from_hash(seed_emb, input_token);
        int8_t b_dim = noise_from_hash(seed_emb + HIDDEN_DIM, tid);
        long long perturb = ((long long)a_tok * b_dim * ns) >> (FIXED_POINT + SIGMA_SHIFT);
        s_x[tid] = clip((long long)emb + pos + perturb);
        __syncthreads();

        // 2. Stack
        for (int l = 0; l < N_LAYERS; l++) {
            uint32_t seed_base = (step_seed + pair_idx) + (l * 1000);
            int8_t *s_norm = &s_mem[HIDDEN_DIM]; // Normalized buf

            // LN 1
            long long local_sum = abs(s_x[tid]);
            long long total_sum = BlockReduce(temp_storage).Sum(local_sum);
            if (tid == 0) shared_scalar = total_sum; __syncthreads();
            long long mean = shared_scalar / HIDDEN_DIM; if(!mean) mean=1;
            int8_t ln_w = model->ln_1[l][tid];
            int8_t ln_n = noise_from_hash(seed_base + SEED_OFF_LN_1, tid);
            int8_t r_in = clip(((long long)s_x[tid] * (ln_w + (((long long)ln_n * ns) >> SIGMA_SHIFT_VECTOR))) / mean);
            s_norm[tid] = r_in; __syncthreads();

            // QKV Projection (Rank-1 Fast)
            long long pbq = (long long)r_in * noise_from_hash(seed_base + SEED_OFF_Q_B, tid);
            long long sbq = BlockReduce(temp_storage).Sum(pbq); if(tid==0) shared_scalar=sbq; __syncthreads(); sbq=shared_scalar;
            
            long long pbk = (long long)r_in * noise_from_hash(seed_base + SEED_OFF_K_B, tid);
            long long sbk = BlockReduce(temp_storage).Sum(pbk); if(tid==0) shared_scalar=sbk; __syncthreads(); sbk=shared_scalar;
            
            long long pbv = (long long)r_in * noise_from_hash(seed_base + SEED_OFF_V_B, tid);
            long long sbv = BlockReduce(temp_storage).Sum(pbv); if(tid==0) shared_scalar=sbv; __syncthreads(); sbv=shared_scalar;

            long long aq=0, ak=0, av=0;
            const int8_t *wq = &model->w_q[l][0];
            const int8_t *wk = &model->w_k[l][0];
            const int8_t *wv = &model->w_v[l][0];
            
            for(int k=0; k<HIDDEN_DIM; k++) {
                int8_t v = s_norm[k];
                aq += (long long)v * wq[k*HIDDEN_DIM + tid];
                ak += (long long)v * wk[k*HIDDEN_DIM + tid];
                av += (long long)v * wv[k*HIDDEN_DIM + tid];
            }
            if(ns!=0) {
                aq += ((sbq * (long long)noise_from_hash(seed_base + SEED_OFF_Q_A, tid)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                ak += ((sbk * (long long)noise_from_hash(seed_base + SEED_OFF_K_A, tid)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                av += ((sbv * (long long)noise_from_hash(seed_base + SEED_OFF_V_A, tid)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
            }
            int8_t qv = clip(aq>>8), kv = clip(ak>>8), vv = clip(av>>8);
            
            // Store KV
            int8_t *lkv = global_kv_cache + kv_ind_offset + (l * kv_layer_stride);
            lkv[t*HIDDEN_DIM + tid] = kv;
            lkv[SEQ_LEN*HIDDEN_DIM + t*HIDDEN_DIM + tid] = vv;
            __syncthreads();

            // Attention (Per Head)
            int h = tid / HEAD_DIM;
            int8_t *s_scores = &s_mem[2*HIDDEN_DIM]; // Accumulator
            if(tid < N_HEADS) ((int32_t*)s_scores)[tid] = 0;
            __syncthreads();

            long long w_v_sum = 0;
            long long tot_sc = 0;
            
            for(int ctx=0; ctx <= t; ctx++) {
                 int8_t k_ctx = lkv[ctx*HIDDEN_DIM + tid];
                 long long df = (long long)qv * k_ctx;
                 
                 // Reduce across head (Head Size 64, Warp 32)
                 // Need shuffle to reduce 64 to 1.
                 // tid 0..31 (Part1), tid 32..63 (Part2)
                 for (int off = 16; off > 0; off /= 2) df += __shfl_down_sync(0xFFFFFFFF, df, off);
                 
                 // Lane 0 of each warp has a partial sum for that warp's head-part.
                 // Lane 0 (tid % 32 == 0).
                 if ((tid % 32) == 0) atomicAdd((int32_t*)&s_scores[h*4], (int32_t)df);
                 __syncthreads();
                 
                 int32_t sc = ((int32_t*)s_scores)[h*4/4]; // Head Score
                 int idx = (sc >> 3) + 128; idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
                 int32_t wt = d_EXP2_TABLE[idx];
                 
                 int8_t v_ctx = lkv[SEQ_LEN*HIDDEN_DIM + ctx*HIDDEN_DIM + tid];
                 w_v_sum += (long long)wt * v_ctx;
                 tot_sc += wt;
                 
                 if(tid < N_HEADS) ((int32_t*)s_scores)[tid] = 0; // Reset
                 __syncthreads();
            }
            if(tot_sc==0) tot_sc=1;
            int8_t ao = clip(w_v_sum / tot_sc);
            s_norm[tid] = ao; __syncthreads();

            // Output Proj
            long long pbo = (long long)ao * noise_from_hash(seed_base + SEED_OFF_O_B, tid);
            long long sbo = BlockReduce(temp_storage).Sum(pbo); if(tid==0) shared_scalar=sbo; __syncthreads(); sbo=shared_scalar;

            const int8_t *wo = &model->w_o[l][0];
            long long aco = 0;
            for(int k=0; k<HIDDEN_DIM; k++) aco += (long long)s_norm[k] * wo[k*HIDDEN_DIM + tid];
            if(ns!=0) aco += ((sbo * (long long)noise_from_hash(seed_base + SEED_OFF_O_A, tid)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
            s_x[tid] = clip((long long)s_x[tid] + (aco >> 8)); __syncthreads();

            // MLP Block
            // Norm 2
            long long msum = abs(s_x[tid]);
            long long mtot = BlockReduce(temp_storage).Sum(msum); if(tid==0) shared_scalar=mtot; __syncthreads();
            long long mmn = shared_scalar/HIDDEN_DIM; if(!mmn) mmn=1;
            int8_t l2w = model->ln_2[l][tid];
            int8_t l2n = noise_from_hash(seed_base + SEED_OFF_LN_2, tid);
            int8_t n2x = clip(((long long)s_x[tid] * (l2w + (((long long)l2n*ns)>>SIGMA_SHIFT_VECTOR))) / mmn);
            s_norm[tid] = n2x; __syncthreads();

            // Expand
            long long pbup = (long long)n2x * noise_from_hash(seed_base + SEED_OFF_MLP_UP_B, tid);
            long long sbup = BlockReduce(temp_storage).Sum(pbup); if(tid==0) shared_scalar=sbup; __syncthreads(); sbup=shared_scalar;

            int8_t *s_mlp = &s_mem[2*HIDDEN_DIM + 256]; // Storage for huge activation? Wait.
            // HIDDEN=768. MLP=3072. 3KB. Fits in Shared (48KB typically).
            // Offset: 2*768 (1.5K) + 256 (Scores) = ~1.8K. +3K = 4.8K. Safe.

            const int8_t *wup = &model->w_up[l][0];
            // Need to compute 4x output dims. Loop.
            for(int sub=0; sub<4; sub++) {
                int oidx = tid + sub*HIDDEN_DIM;
                long long aup = 0;
                for(int k=0; k<HIDDEN_DIM; k++) aup += (long long)s_norm[k] * wup[k*(4*HIDDEN_DIM) + oidx];
                if(ns!=0) aup += ((sbup * (long long)noise_from_hash(seed_base + SEED_OFF_MLP_UP_A, oidx)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                int8_t v = clip(aup>>8); if(v<0) v=0; // ReLU
                s_mlp[oidx] = v; // Requires synchronization if reused across threads?
                // Each thread computes its own 'oidx' slices? 
                // No. Thread tid computes out [tid, tid+H, tid+2H...].
                // 's_mlp' is a shared buffer.
            }
            __syncthreads();

            // Down Project
            // Input is s_mlp (4H).
            // Need scalar B projection of s_mlp.
            long long pbdn = 0;
            for(int sub=0; sub<4; sub++) pbdn += (long long)s_mlp[tid + sub*HIDDEN_DIM] * noise_from_hash(seed_base + SEED_OFF_MLP_DOWN_B, tid + sub*HIDDEN_DIM);
            long long sbdn = BlockReduce(temp_storage).Sum(pbdn); if(tid==0) shared_scalar=sbdn; __syncthreads(); sbdn=shared_scalar;

            const int8_t *wdn = &model->w_down[l][0];
            long long adn = 0;
            for(int k=0; k<(4*HIDDEN_DIM); k++) adn += (long long)s_mlp[k] * wdn[k*HIDDEN_DIM + tid];
            if(ns!=0) adn += ((sbdn * (long long)noise_from_hash(seed_base + SEED_OFF_MLP_DOWN_A, tid)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
            s_x[tid] = clip((long long)s_x[tid] + (adn >> 9)); __syncthreads();
        }

        // 3. Final Head
        long long fsum = abs(s_x[tid]);
        long long ftot = BlockReduce(temp_storage).Sum(fsum); if(tid==0) shared_scalar=ftot; __syncthreads();
        long long fmn = shared_scalar/HIDDEN_DIM; if(!fmn) fmn=1;
        
        int8_t lfw = model->ln_f[tid];
        int8_t lfn = noise_from_hash(step_seed + pair_idx + SEED_OFF_LN_F, tid);
        int8_t nf = clip(((long long)s_x[tid] * (lfw + (((long long)lfn*ns)>>SIGMA_SHIFT_VECTOR))) / fmn);
        
        // Reuse s_mem[HD] for normed
        int8_t *s_norm = &s_mem[HIDDEN_DIM];
        s_norm[tid] = nf; __syncthreads();

        long long pbh = (long long)nf * noise_from_hash(step_seed + pair_idx + SEED_OFF_HEAD + VOCAB_SIZE, tid);
        long long sbh = BlockReduce(temp_storage).Sum(pbh); if(tid==0) shared_scalar=sbh; __syncthreads(); sbh=shared_scalar;

        // Compute Logits (only first VOCAB_SIZE threads)
        if(tid < VOCAB_SIZE) {
            long long ah = 0;
            const int8_t *wh = &model->head[0];
            for(int k=0; k<HIDDEN_DIM; k++) ah += (long long)s_norm[k] * wh[k*VOCAB_SIZE + tid];
            if(ns!=0) ah += ((sbh * (long long)noise_from_hash(step_seed + pair_idx + SEED_OFF_HEAD, tid)) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
            int8_t lgt = clip(ah >> 8);
            
            int idx = (int32_t)lgt + 128;
            long long ex = d_EXP2_TABLE[idx];
            
            // Store logits in s_x (safe reuse, we are done with resid for this step)
            // Cast to int32 buffer at start
             ((int32_t*)s_x)[tid] = (int32_t)ex; 
             ((int8_t*)s_x)[HIDDEN_DIM*4 + tid] = lgt; // Keep raw logit at offset
        }
        __syncthreads();

        // Loss Calc (Thread 0)
        if (tid == 0) {
             uint8_t target_token = dataset[stream_pos + t + 1];
             long long sum_ex = 0;
             for(int i=0; i<VOCAB_SIZE; i++) sum_ex += ((int32_t*)s_x)[i];
             
             long long log_sum = 0;
             if (sum_ex > 0) {
                 long long x = sum_ex; int pos=0;
                 while(x>=256){x>>=8; pos+=8;} if(x>=16){x>>=4; pos+=4;} if(x>=2){pos+=1;}
                 log_sum = (pos<<4) + ((sum_ex-(1LL<<pos))<<(4-pos)) - 64; 
             }
             int8_t tgt_lgt = ((int8_t*)s_x)[HIDDEN_DIM*4 + target_token];
             int32_t tgt_val = (int32_t)tgt_lgt + 128;
             my_loss += (log_sum - tgt_val);
        }
        __syncthreads(); // Sync before next step overwrites s_x
    }

    if (tid == 0) accum_loss[p_idx] = (int32_t)my_loss;
}

__global__ void compute_fitness_kernel(const int32_t *accum_loss, int32_t *fitnesses, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int32_t p = accum_loss[2*idx]; int32_t n = accum_loss[2*idx+1];
    fitnesses[idx] = (p < n) ? 1 : ((n < p) ? -1 : 0);
}

__global__ void update_matrix_kernel(int8_t *W, int rows, int cols, int off_A, int off_B, int seed_base, const int32_t *fitnesses, uint32_t step_seed, int thres) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int r = idx % rows; int c = idx / rows;
    long long vote = 0;
    for(int p=0; p < POPULATION_SIZE/2; p++) {
        int fit = fitnesses[p]; if(fit==0) continue;
        uint32_t s = step_seed + p + seed_base;
        vote += (long long)fit * noise_from_hash(s + off_A, r) * noise_from_hash(s + off_B, c);
    }
    int8_t w = W[idx];
    if(vote > thres && w < MAX_VAL) w++; else if(vote < -thres && w > MIN_VAL) w--;
    W[idx] = w;
}

__global__ void update_vector_kernel(int8_t *V, int len, int off_A, int seed_base, const int32_t *fitnesses, uint32_t step_seed, int thres) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    long long vote = 0;
    for(int p=0; p < POPULATION_SIZE/2; p++) {
        int fit = fitnesses[p]; if(fit==0) continue;
        vote += (long long)fit * noise_from_hash(step_seed + p + seed_base + off_A, idx);
    }
    int8_t v = V[idx];
    if(vote > thres && v < MAX_VAL) v++; else if(vote < -thres && v > MIN_VAL) v--;
    V[idx] = v;
}

int main() {
    signal(SIGINT, handle_sigint);
    init_tables();
    cudaMemcpyToSymbol(d_EXP2_TABLE, h_EXP2_TABLE, 256*sizeof(int32_t));

    Dataset ds = {0,0};
    FILE *f = fopen("input.txt", "rb");
    if(!f) { printf("No input.txt\n"); exit(1); }
    fseek(f,0,SEEK_END); ds.length=ftell(f); fseek(f,0,SEEK_SET);
    ds.data=(uint8_t*)malloc(ds.length); fread(ds.data,1,ds.length,f); fclose(f);

    TransformerModel *h_model = (TransformerModel*)calloc(1, sizeof(TransformerModel));
    init_model(h_model);
    TransformerModel *d_model; CHECK_CUDA(cudaMalloc(&d_model, sizeof(TransformerModel)));
    CHECK_CUDA(cudaMemcpy(d_model, h_model, sizeof(TransformerModel), cudaMemcpyHostToDevice));

    uint8_t *d_dataset; CHECK_CUDA(cudaMalloc(&d_dataset, ds.length));
    CHECK_CUDA(cudaMemcpy(d_dataset, ds.data, ds.length, cudaMemcpyHostToDevice));

    int32_t *d_loss, *d_fit;
    CHECK_CUDA(cudaMalloc(&d_loss, POPULATION_SIZE * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_fit, (POPULATION_SIZE/2) * sizeof(int32_t)));
    
    size_t kv_size = (size_t)POPULATION_SIZE * N_LAYERS * 2 * SEQ_LEN * HIDDEN_DIM;
    printf("Allocating KV Cache: %.2f GB\n", kv_size / (1024.0*1024*1024));
    CHECK_CUDA(cudaMalloc(&d_kv_cache, kv_size));

    printf("Starting Transformer Training (Pop=%d, Dim=%d)...\n", POPULATION_SIZE, HIDDEN_DIM);
    long max_steps = (ds.length - 1) / SEQ_LEN;

    for(long step=0; step<max_steps && keep_running; step++) {
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        uint32_t seed = (uint32_t)time(NULL) ^ (step * 0x12345678);
        
        // 6KB Shared Mem (2*Hidden + 256 + MLP Buffer)
        size_t sm_size = 2 * HIDDEN_DIM + 512 + (4*HIDDEN_DIM); 
        train_sequence_kernel<<<POPULATION_SIZE, BLOCK_THREADS, sm_size>>>(
            d_dataset, ds.length, step*SEQ_LEN, d_model, d_kv_cache, d_loss, seed
        );
        CHECK_CUDA(cudaDeviceSynchronize());
        
        compute_fitness_kernel<<< (POPULATION_SIZE/2 + 255)/256, 256 >>>(d_loss, d_fit, POPULATION_SIZE/2);
        
        thrust::device_ptr<int32_t> t_loss(d_loss);
        long long total_loss = thrust::reduce(t_loss, t_loss + POPULATION_SIZE);
        double avg_loss = (double)total_loss / (POPULATION_SIZE * SEQ_LEN * 16.0); // Normalization factor approx
        int thres = get_update_threshold(avg_loss);

        for(int l=0; l<N_LAYERS; l++) {
            int s_base = l * 1000;
            int d2 = HIDDEN_DIM*HIDDEN_DIM;
            
            // Layers
            update_matrix_kernel<<< (d2+511)/512, 512 >>>( (int8_t*)d_model->w_q[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_Q_A, SEED_OFF_Q_B, s_base, d_fit, seed, thres);
            update_matrix_kernel<<< (d2+511)/512, 512 >>>( (int8_t*)d_model->w_k[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_K_A, SEED_OFF_K_B, s_base, d_fit, seed, thres);
            update_matrix_kernel<<< (d2+511)/512, 512 >>>( (int8_t*)d_model->w_v[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_V_A, SEED_OFF_V_B, s_base, d_fit, seed, thres);
            update_matrix_kernel<<< (d2+511)/512, 512 >>>( (int8_t*)d_model->w_o[l], HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_O_A, SEED_OFF_O_B, s_base, d_fit, seed, thres);
            
            update_matrix_kernel<<< (HIDDEN_DIM*4*HIDDEN_DIM+511)/512, 512 >>>( (int8_t*)d_model->w_up[l], HIDDEN_DIM, 4*HIDDEN_DIM, SEED_OFF_MLP_UP_A, SEED_OFF_MLP_UP_B, s_base, d_fit, seed, thres);
            update_matrix_kernel<<< (4*HIDDEN_DIM*HIDDEN_DIM+511)/512, 512 >>>( (int8_t*)d_model->w_down[l], 4*HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_MLP_DOWN_A, SEED_OFF_MLP_DOWN_B, s_base, d_fit, seed, thres);
            
            update_vector_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>(d_model->ln_1[l], HIDDEN_DIM, SEED_OFF_LN_1, s_base, d_fit, seed, thres);
            update_vector_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>(d_model->ln_2[l], HIDDEN_DIM, SEED_OFF_LN_2, s_base, d_fit, seed, thres);
        }
        
        update_vector_kernel<<< (HIDDEN_DIM+255)/256, 256 >>>(d_model->ln_f, HIDDEN_DIM, SEED_OFF_LN_F, 0, d_fit, seed, thres);
        update_matrix_kernel<<< (HIDDEN_DIM*VOCAB_SIZE+511)/512, 512 >>>((int8_t*)d_model->head, HIDDEN_DIM, VOCAB_SIZE, SEED_OFF_HEAD, SEED_OFF_HEAD+VOCAB_SIZE, 0, d_fit, seed, thres);
        update_matrix_kernel<<< (VOCAB_SIZE*HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model->embedding, VOCAB_SIZE, HIDDEN_DIM, SEED_OFF_EMB, SEED_OFF_EMB+HIDDEN_DIM, 0, d_fit, seed, thres);
        
        CHECK_CUDA(cudaDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &t1);
        
        if(step % 10 == 0) printf("Step %ld | Loss: %.4f | Time: %.2f ms\n", step, avg_loss, get_time_diff_ms(t0, t1));
    }

    free(h_model); free(ds.data);
    cudaFree(d_model); cudaFree(d_dataset); cudaFree(d_loss); cudaFree(d_fit); cudaFree(d_kv_cache);
    return 0;
}
