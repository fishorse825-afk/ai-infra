#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matmul_shared(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * 16 + ty;
    int col = bx * 16 + tx;

    float sum = 0.0f;
    for (int k = 0; k < K; k += 16) {
        if (row < M && (k + tx) < K) {
            As[ty][tx] = A[row * K + k + tx];
        }else{
            As[ty][tx] = 0.0f;
        }
        if (col < N && (k + ty) < K) {
            Bs[ty][tx] = B[k + ty * N + col];
        }else{
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < 16; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    int M = 512, N = 512, K = 512;
    size_t A_size = M * K * sizeof(float);
    size_t B_size = K * N * sizeof(float);
    size_t C_size = M * N * sizeof(float);

    cudaEvent_t start, stop;
    float gpu_time = 0.0f;

    float *h_A = (float *)malloc(A_size);
    float *h_B = (float *)malloc(B_size);
    float *h_C = (float *)malloc(C_size);
    
    for (int i=0; i<M*K; i++) h_A[i] = 1.0f;
    for (int i=0; i<K*N; i++) h_B[i] = 2.0f;
    for (int i=0; i<M*N; i++) h_C[i] = 0.0f;
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A_size);
    cudaMalloc(&d_B, B_size);
    cudaMalloc(&d_C, C_size);

    cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    matmul_shared<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_C, d_C, C_size, cudaMemcpyDeviceToHost);
    printf("C[0] = %f (expected %f)\n", h_C[0], 2.0f * K);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
