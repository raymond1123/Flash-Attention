#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

__global__
void backward_kernel(const __half* Q, const __half* K, const __half* V, 
                     __half* dQ, __half* dK, __half* dV, 
                     const int N, const int d,
                     const int Tc, const int Tr, const int Bc, const int Br, 
                     const __half* l, const __half* m, 
                     const __half* O, const __half* dO) {

    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    const __half softmax_scale = __hdiv(__float2half(1.0), hsqrt(__float2half(static_cast<float>(d))));

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
    extern __shared__ __half sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    __half* Qi =  sram; 
    __half* Kj =  &sram[tile_size*1];
    __half* Vj =  &sram[tile_size*2];
    __half* Oi =  &sram[tile_size*3];

    __half* dQi = &sram[tile_size*4];
    __half* dKj = &sram[tile_size*5];
    __half* dVj = &sram[tile_size*6];
    __half* dOi = &sram[tile_size*7];

    __half* Sij  = &sram[tile_size*8];
    __half* dSij = Sij  + Br*Bc; 
    __half* dPij = dSij + Br*Bc;

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
            __half row_m = m[lm_offset + (Br * i) + tx];
            __half row_l = l[lm_offset + (Br * i) + tx];
            __syncthreads();

            // 1. Sij = softmax_scale*Qi@Kj^T
            for (int y = 0; y < Bc; y++) {
                __half sum = __float2half(0.0f);
                for (int x = 0; x < d; x++) {
                    sum = __hadd(sum, __hmul(Qi[(tx * d) + x], Kj[(y * d) + x]));
                }
                sum = __hmul(sum, softmax_scale);
                Sij[(Bc * tx) + y] = sum;
            }

            // 2. Pij = exp(Sij - row_m)
            for (int y = 0; y < Bc; y++) {
                __half tmp = hexp(__hsub(Sij[(Bc * tx) + y], row_m));
                Sij[(Bc * tx) + y] = __hmul(__hdiv(__float2half(1.0), row_l), tmp);
            }

            // 3. dVj = dVj + Pij^T@dOi
            for (int x = 0; x < d; x++) {
                for (int y = 0; y < Br; y++) {
                    dVj[tx*d+x] = __hadd(dVj[tx*d+x], __hmul(Sij[(Bc * y) + tx], dOi[(y * d) + x]));
                }
            }
            __syncthreads();

            // 4. dPij = dOi@Vj^T 
            for (int y = 0; y < Bc; y++) {
                __half sum = __float2half(0.0f);
                for (int x = 0; x < d; x++) {
                    sum = __hadd(sum, __hmul(dOi[(tx * d) + x], Vj[(y * d) + x]));
                }
                dPij[(Bc*tx)+y] = sum;
            }

            // 5. rowsum(dOi*Oi)
            __half Di = __float2half(0.0f);
            for(int x=0; x<d; ++x) {
                Di = __hadd(Di, __hmul(dOi[tx*d+x], Oi[tx*d+x]));
            }

            // 6. dSij = Pij*(dPij-Di) 
            for(int x=0; x<Bc; ++x) {
                dSij[tx*Bc+x] = __hmul(Sij[tx*Bc+x], __hsub(dPij[tx*Bc+x], Di));
            }

            // 7. dQi = dQi + softmax_scale*Sij@Kj ==> (Br, Bc)@(Bc, d) = (Br, d)
            for(int x=0; x<d; ++x) {
                __half sum = __float2half(0.0f);
                for(int y=0; y<Bc; ++y) {
                    sum = __hadd(sum, __hmul(dSij[tx*Bc+y], Kj[y*d+x]));
                }

                sum = __hmul(sum, softmax_scale);
                dQi[tx*d+x] = __hadd(dQi[tx*d+x], sum); 
            }

            /* 
                Note: tricky here, only works when Bc=Br, 
                      actually, bad design!
            */
            // 8. dKj = dKj + softmax_scale*dSij^T@Qi ==> (Bc, Br)@(Br, d) = (Bc, d)
            for (int x = 0; x < d; x++) {
                __half sum = __float2half(0.0f);
                for (int y = 0; y < Br; y++) {
                    sum = __hadd(sum, __hmul(dSij[y*Bc+tx], Qi[y*d+x]));
                }
                sum = __hmul(sum, softmax_scale);
                dKj[tx*d+x] = __hadd(dKj[tx*d+x], sum); 

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
void forward_kernel(const __half* Q, const __half* K, const __half* V, 
                    const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, 
                    __half* l, __half* m, __half* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    const __half softmax_scale = __hdiv(__float2half(1.0), hsqrt(__float2half(static_cast<float>(d))));

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ __half sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    __half* Qi = sram;
    __half* Kj = &sram[tile_size];
    __half* Vj = &sram[tile_size * 2];
    __half* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++)
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];

            __half row_m_prev = m[lm_offset + (Br * i) + tx];
            __half row_l_prev = l[lm_offset + (Br * i) + tx];

            // 1. S = QK^T, row_m = rowmax(S) ==> (Br,d)@(d,Bc) = (Br,Bc)
            __half row_m = __float2half(-INFINITY);
            for (int y = 0; y < Bc; y++) {
                __half sum = __float2half(0.0f);
                for (int x = 0; x < d; x++) {
                    sum = __hadd(sum, __hmul(Qi[(tx * d) + x], Kj[(y * d) + x]));
                }
                sum = __hmul(sum, softmax_scale);
                S[(Bc * tx) + y] = sum;

                if (__hgt(sum, row_m))
                    row_m = sum;
            }

            // 2. P = exp(S - row_m), row_l = rowsum(P)
            __half row_l = __float2half(0.0f);
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = hexp(__hsub(S[(Bc * tx) + y], row_m));
                row_l = __hadd(row_l, S[(Bc * tx) + y]);
            }

            // Compute new m and l
            __half row_m_new = __hgt(row_m_prev, row_m)?row_m_prev:row_m;
            __half tmp = __hmul(hexp(__hsub(row_m_prev, row_m_new)), row_l_prev);
            __half row_l_new =  __hadd(tmp, __hmul(hexp(__hsub(row_m, row_m_new)), row_l));

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                __half pv = __float2half(0.0f);  // Pij * Vj ==> (Br, Bc)@(Bc, d) = (Br, d)
                for (int y = 0; y < Bc; y++) {
                    pv = __hadd(pv, __hmul(S[(Bc * tx) + y], Vj[(y * d) + x]));
                }
                __half tmp = __hmul(__hmul(row_l_prev, hexp(__hsub(row_m_prev, row_m_new))), O[qkv_offset+tile_size*i+(tx * d) + x]);
                tmp = __hadd(tmp, __hmul(hexp(__hsub(row_m, row_m_new)), pv));
                O[qkv_offset+tile_size*i+(tx * d) + x] = __hmul(__hdiv(__float2half(1.0f), row_l_new), tmp); 
            }

            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor Q, 
                                                                torch::Tensor K, 
                                                                torch::Tensor V, 
                                                                torch::Tensor l, 
                                                                torch::Tensor m) {
    // TODO: determine Bc, Br dynamically
    const int M = 64*1024; // shared memory size
    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Bc = M/(4*d*sizeof(__half))>d?d:(M/(4*d*sizeof(__half))); 
    const int Br = M/(4*d*sizeof(__half))>d?d:(M/(4*d*sizeof(__half))); 
    //const int Bc = 128, Br=128; 
    printf("Bc=%d, Br=%d\n", Bc, Br);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q, torch::dtype(torch::kHalf));
    torch::Device device(torch::kCUDA);
    O = O.to(device);

    // Calculate SRAM size needed per block
    const int sram_size = 4 * Bc * d * sizeof(__half);
    printf("Max shared memory: %d, requested shared memory: %d \n", M, sram_size);
    cudaFuncSetAttribute(forward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, M);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    forward_kernel<<<grid_dim, block_dim, M>>>(
        reinterpret_cast<__half*>(Q.data_ptr<c10::Half>()), 
        reinterpret_cast<__half*>(K.data_ptr<c10::Half>()), 
        reinterpret_cast<__half*>(V.data_ptr<c10::Half>()),
        N, d, Tc, Tr, Bc, Br, 
        reinterpret_cast<__half*>(l.data_ptr<c10::Half>()), 
        reinterpret_cast<__half*>(m.data_ptr<c10::Half>()), 
        reinterpret_cast<__half*>(O.data_ptr<c10::Half>())
    );

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
    const int M = 64*1024; // shared memory size
    //const int Bc = 32; const int Br = 32;
    const int Bc = 16; const int Br = 16;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);

    // Calculate SRAM size needed per block
    const int sram_size = (8 * Bc * d * sizeof(__half)) + (3*Bc * Br * sizeof(__half));
    //int max_sram_size;
    //cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    //printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);
    cudaFuncSetAttribute(backward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, M);
    printf("Max shared memory: %d, requested shared memory: %d \n", M, sram_size);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    /* init dO,dK,dV to 0 */
    auto dQ = torch::zeros_like(Q, torch::dtype(torch::kHalf));
    auto dK = torch::zeros_like(K, torch::dtype(torch::kHalf));
    auto dV = torch::zeros_like(V, torch::dtype(torch::kHalf));
    torch::Device device(torch::kCUDA);
    dQ = dQ.to(device); dK = dK.to(device); dV = dV.to(device);

    //backward_kernel<<<grid_dim, block_dim, sram_size>>>(
    backward_kernel<<<grid_dim, block_dim, M>>>(
        reinterpret_cast<__half*>(Q.data_ptr<c10::Half>()), 
        reinterpret_cast<__half*>(K.data_ptr<c10::Half>()), 
        reinterpret_cast<__half*>(V.data_ptr<c10::Half>()),
        reinterpret_cast<__half*>(dQ.data_ptr<c10::Half>()), 
        reinterpret_cast<__half*>(dK.data_ptr<c10::Half>()), 
        reinterpret_cast<__half*>(dV.data_ptr<c10::Half>()),
        N, d, Tc, Tr, Bc, Br,
        reinterpret_cast<__half*>(l.data_ptr<c10::Half>()), 
        reinterpret_cast<__half*>(m.data_ptr<c10::Half>()), 
        reinterpret_cast<__half*>(O.data_ptr<c10::Half>()), 
        reinterpret_cast<__half*>(dO.data_ptr<c10::Half>())
    );

    return std::make_tuple(dQ, dK, dV);
}

