#include <assert.h>
#include <iostream>
#include <cstdlib>
#include<sys/time.h>
#include <cmath>
#include "cuda_runtime.h"
const int  LANGE = 16;

__global__ void vecAdd(double *d_a, double *d_b, double *d_c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (N / LANGE)) {
        int large = N / LANGE;
        for (int i = 0;  i < LANGE; i++)
            if (idx % 2 == 0){
                d_c[idx + i * large] = d_a[idx + i * large] + d_b[idx + i *large];
            }else{
                d_c[idx + i * large] = d_a[idx + i * large] - d_b[idx + i * large];
            }
    }
}

int main (void) {
    int n;
    std::cin>>n;
    assert(n % LANGE == 0);
    double *H_a, *H_b, *H_c;
    size_t bytes = n * sizeof(double);
    H_a = (double*)malloc(bytes);
    H_b = (double*)malloc(bytes);
    H_c = (double*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        H_a[i] = sin(i) * sin(i);
        H_b[i] = cos(i) * cos(i);
    }
    
    //GPU parper
    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, H_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, H_b, bytes, cudaMemcpyHostToDevice);
    int blockSize = 1024;
    int gridSize = ((n-1)/LANGE)/blockSize + 1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //gpu running
    cudaEventRecord(start);
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaMemcpy(H_c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float timeGPU = 0;
    cudaEventElapsedTime(&timeGPU, start, stop);
    std::cout << "Runtime for GPU is: " << timeGPU<<" ms "<< std::endl;

    //CPU running.....
    struct timeval startCPU,endCPU;
    gettimeofday(&startCPU,NULL);
    for (int i = 0; i < n; i++) {
        H_c[i] = H_a[i] + H_b[i];
    }
    gettimeofday(&endCPU,NULL);
    double timeCPU =  endCPU.tv_sec - startCPU.tv_sec  + (double)(endCPU.tv_usec - startCPU.tv_usec)/1000000;
    std::cout << "Runtime for CPU is: " << timeCPU <<" ms "<< std::endl;

    //summary
    if(timeGPU<timeCPU){
        std::cout<<"GPU is faster than CPU for "<<timeCPU-timeGPU<<" ms "<<std::endl;
    }else if(timeGPU>timeCPU){
        std::cout<<"CPU is faster than GPU for "<<timeGPU-timeCPU<<" ms "<<std::endl;
    }else{
        std::cout<<"same time for GPU and CPU"<<std::endl;
    }

    free(H_a);
    free(H_b);
    free(H_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
