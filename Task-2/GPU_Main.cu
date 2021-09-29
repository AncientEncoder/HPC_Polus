#include "cuda_runtime.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include<sys/time.h>
using namespace std;
__global__ void transposeKernel(const double* A, double* AT, int N) {
  int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    AT[yIndex+xIndex*N]=A[xIndex+yIndex*N];
}
int main(void) {
    int rank=100;//for 50*50 mart
    struct timeval start, end;
    int N =rank;

    dim3 threadPerBlock(N, N);
    dim3 blockNumber((N+threadPerBlock.x-1)/ threadPerBlock.x, (N+threadPerBlock.y-1)/ threadPerBlock.y );

    size_t size = N * N * sizeof(double);

    double* h_A = (double*)malloc(size);

    double* h_AT = (double*)malloc(size);

    for (int i = 0; i < N * N; i++) {
      h_A[i] = i +1;
    }

    int i = 0, k = 0;
    gettimeofday(&start,NULL);
    
    while (i < N * N) {
      for (int j = k; j < N * N; j += N) {
        h_AT[i++] = h_A[j];
      }
      k++;
    }

    gettimeofday(&end,NULL);
    int timeuseCPU = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    cout << "total time for cpu is " << timeuseCPU<< "us" <<endl;

    double* d_A = NULL;
    double* d_AT = NULL;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_AT, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    gettimeofday(&start,NULL);
    transposeKernel<<<blockNumber, threadPerBlock>>>(d_A, d_AT, N);
    cudaDeviceSynchronize();
    gettimeofday(&end,NULL);
    int timeuseGPU = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    cout << "total time use in GPU is " << timeuseGPU<< "us" <<endl;
    cudaMemcpy(h_AT, d_AT, size, cudaMemcpyDeviceToHost);
    if(timeuseGPU<timeuseCPU){
        cout<<"GPU is faster than CPU for "<<timeuseCPU-timeuseGPU<<" us"<<endl;
    }else{
        cout<<"CPU is faster than GPU for "<<timeuseGPU-timeuseCPU<<" us"<<endl;
    }
    free(h_A);
    free(h_AT);
    cudaFree(d_A);
    cudaFree(d_AT);
  return 0;
}