#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* 
    dO.shape=(Br, d) 
    O.shape=(Br, d) 
    D.shape=(Br,) 
*/
__device__ void row_sum(const float* dO, const float* O, float* D,
                        int tx, int d, const int Tr, 
                        const int Br, int tile_size,
                        const int qkv_offset, const int lm_offset,
                        float* sram) {

    float* Oi =  sram;   
    float* dOi =  &sram[tile_size];

    for (int i = 0; i < Tr; i++)  {
        for (int x = 0; x < d; x++) {
            Oi[(tx * d) + x]  = O[qkv_offset + (tile_size * i) + (tx * d) + x];
            dOi[(tx * d) + x] = dO[qkv_offset + (tile_size * i) + (tx * d) + x];
        }
        __syncthreads(); 

        // row_sum(dO*O), shape=(Br, d)
        float row_m = 0; 
        for(int x=0; x<d; ++x) {
            row_m += Oi[tx*d+x]*dOi[tx*d+x];
        }

        D[lm_offset + (Br * i) + tx] = row_m;
    }
}

__global__
void backward_kernel(const float* Q, const float* K, const float* V, 
                     float* dQ, float* dK, float* dV, 
                     const int N, const int d,
                     const int Tc, const int Tr, const int Bc, const int Br, 
                     const float softmax_scale,
                     const float* l, const float* m, 
                     const float* O, const float* dO, float* row_D) {

    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    /* 
        SRAM: Kj, Vj, dKj, dVj ==> shape=(Bc, d)
              Oi, dOi, Qi, dQi ==> shape=(Br, d)
              Sij.shape = (Br, Bc)
              dSij.shape = (Br, Bc)
              dPij.shape = (Br, Bc)
     */
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    row_sum(dO, O, row_D, tx, d, Tr, Br, tile_size,qkv_offset, lm_offset, sram);

    float* Qi =  sram; 
    float* Kj =  &sram[tile_size*1];
    float* Vj =  &sram[tile_size*2];
    float* Oi =  &sram[tile_size*3];

    float* dQi = &sram[tile_size*4];
    float* dKj = &sram[tile_size*5];
    float* dVj = &sram[tile_size*6];
    float* dOi = &sram[tile_size*7];

    float* Sij  = &sram[tile_size*8];
    float* dSij = Sij  + Br*Bc; 
    float* dPij = dSij + Br*Bc;

    __syncthreads();

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];

            dKj[(tx * d) + x] = dK[qkv_offset + (tile_size * j) + (tx * d) + x];
            dVj[(tx * d) + x] = dV[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads(); 

        for (int i = 0; i < Tr; i++)  {

            // Load Qi,Oi,dOi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x]  = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
                Oi[(tx * d) + x]  = O[qkv_offset + (tile_size * i) + (tx * d) + x];

                dQi[(tx * d) + x] = dQ[qkv_offset + (tile_size * i) + (tx * d) + x];
                dOi[(tx * d) + x] = dO[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float Di = row_D[lm_offset + (Br * i) + tx];
            float row_l = l[lm_offset + (Br * i) + tx];
            __syncthreads();

            // 1. Sij = softmax_scale*Qi@Kj^T
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                Sij[(Bc * tx) + y] = sum;
            }

            // 2. Pij = exp(Sij - Li)
            for (int y = 0; y < Bc; y++)
                Sij[(Bc * tx) + y] = __expf(Sij[(Bc * tx) + y] - row_l);

            // 3. dVj = dVj + Pij^T@dOi
            for (int x = 0; x < d; x++) {
                for (int y = 0; y < Br; y++) {
                    dVj[tx*d+x] += Sij[(Bc * y) + tx] * dOi[(y * d) + x];
                }
            }
            __syncthreads();

            // 4. dPij = dOi@Vj^T 
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    //dPij[(Bc * tx) + y] = dOi[(tx * d) + x] * Vj[(y * d) + x]; // P(tx, y)
                    sum += dOi[(tx * d) + x] * Vj[(y * d) + x];
                }
                dPij[(Bc*tx)+y] = sum;
            }

            // 5. dSij = Pij*(dPij-Di) 
            for(int x=0; x<Bc; ++x) {
                dSij[tx*Bc+x] = Sij[tx*Bc+x]*(dPij[tx*Bc+x]-Di);
            }

            // 6. dQi = dQi + softmax_scale*Sij@Kj ==> (Br, Bc)@(Bc, d) = (Br, d)
            for(int x=0; x<d; ++x) {
                float sum = 0;
                for(int y=0; y<Bc; ++y) {
                    sum += dSij[tx*Bc+y]*Kj[y*d+x];
                }

                sum *= softmax_scale;
                dQi[tx*d+x] += sum; 
            }

            /* 
                Note: tricky here, only works when Bc=Br, 
                      actually, bad design!
            */
            // 7. dKj = dKj + softmax_scale*dSij^T@Qi ==> (Bc, Br)@(Br, d) = (Bc, d)
            for (int x = 0; x < d; x++) {
                float sum = 0;
                for (int y = 0; y < Br; y++) {
                    sum += dSij[y*Bc+tx]*Qi[y*d+x];
                }
                sum *= softmax_scale;
                dKj[tx*d+x] += sum; 

            }
            __syncthreads(); 

            for (int x = 0; x < d; x++) {
                dQ[qkv_offset + (tile_size * i) + (tx * d) + x] = dQi[(tx * d) + x];
            }
        }
        __syncthreads(); 

        for (int x = 0; x < d; x++) {
            dK[qkv_offset + (tile_size * j) + (tx * d) + x] = dKj[(tx * d) + x];
            dV[qkv_offset + (tile_size * j) + (tx * d) + x] = dVj[(tx * d) + x];
        }
        __syncthreads(); 
    }
}

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int i = 0; i < Tr; i++) {
        // Load Qi to SRAM, l and m to registers
        for (int x = 0; x < d; x++) 
            Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];

        float row_m_new;
        float row_l_new;
        float row_m_prev = m[lm_offset + (Br * i) + tx];
        float row_l_prev = l[lm_offset + (Br * i) + tx];
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int j = 0; j < Tc; j++)  {

            // Load Kj, Vj to SRAM
            for (int x = 0; x < d; x++) {
                Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
                Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
            }

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            row_m_new = max(row_m_prev, row_m);
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m_new);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + row_l; 

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }

                O[qkv_offset + (tile_size * i) + (tx * d) + x] = 
                    __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x] + pv;
            }

            row_m_prev = row_m_new;
            row_l_prev = row_l_new;

        } // for Tc, j  
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop

        for (int x = 0; x < d; x++)
            O[qkv_offset + (tile_size * i) + (tx * d) + x] *= 1/row_l_new;

        l[lm_offset + (Br * i) + tx] = row_m_new + __logf(row_l_new);
    } // for Tr, i
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor Q, 
                                                                torch::Tensor K, 
                                                                torch::Tensor V, 
                                                                torch::Tensor l, 
                                                                torch::Tensor m) {
    // TODO: determine Bc, Br dynamically
    const int Bc = 32; const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    torch::Device device(torch::kCUDA);

    // Calculate SRAM size needed per block
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );

    //return O;
    return std::make_tuple(l, m, O);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward(torch::Tensor Q, 
                                                                 torch::Tensor K, 
                                                                 torch::Tensor V, 
                                                                 torch::Tensor O, 
                                                                 torch::Tensor dO, 
                                                                 torch::Tensor l, 
                                                                 torch::Tensor m) {
    // TODO: determine Bc, Br dynamically
    const int Bc = 32; const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Calculate SRAM size needed per block
    const int sram_size = (8 * Bc * d * sizeof(float)) + (3*Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    /* init dO,dK,dV to 0 */
    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);
    auto D = torch::zeros_like(l);
    torch::Device device(torch::kCUDA);

    dQ = dQ.to(device); dK = dK.to(device); dV = dV.to(device);
    D = D.to(device);

    backward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), 
        O.data_ptr<float>(), dO.data_ptr<float>(), 
        D.data_ptr<float>()
    );

    return std::make_tuple(dQ, dK, dV);
}

