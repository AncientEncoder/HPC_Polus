#include <iostream>
#include <assert.h>
#include <cstdlib>
#include "cuda_runtime.h"

const int SIZE = 4096;

__global__ void dymTrans(int *V, int N) {
    extern __shared__ int array[];

    int refIndex = threadIdx.x;
    array[refIndex] = V[refIndex];

    __syncthreads();

    V[refIndex] = array[N-refIndex-1];
}

__global__ void stdTrans(int *V, int N) {
    __shared__ int array[SIZE];

    int refIndex = threadIdx.x; 
    array[refIndex] = V[refIndex];

    __syncthreads();

    V[refIndex] = array[N-refIndex-1];
}

__global__ void normalTrans(int *V, int N) {
    int refIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (refIndex <=  N/2) {
        int reff = V[refIndex];
        V[refIndex] = V[N-refIndex-1];
        V[N-refIndex-1] = reff;
    }
}

int main(void) {
    
    int n =0;
    std::cin>>n;

    size_t size = n * sizeof(int);
    int *V = (int*)malloc(size);
    int *V_t = (int*)malloc(size);

    for (int i = 0; i < n; i++) {
        V[i] = i+1;
    }

    int block = 1024;
    int grid = 1;

    int *v_D;
    cudaMalloc(&v_D, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //nTrans
    cudaEventRecord(start);
    cudaMemcpy(v_D, V, size, cudaMemcpyHostToDevice);
    normalTrans<<<grid, block>>>(v_D, n);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaMemcpy(V_t, v_D, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    float nTrans = 0;
    cudaEventElapsedTime(&nTrans, start, stop);
    std::cout << "Normal Trans used time: " << nTrans << std::endl;
    //sTrans
    cudaEventRecord(start);
    cudaMemcpy(v_D, V, size, cudaMemcpyHostToDevice);
    stdTrans<<<grid, block, n>>>(v_D, n);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaMemcpy(V_t, v_D, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    float sTrans=0.0;
    cudaEventElapsedTime(&sTrans, start, stop);
    std::cout << "Static Trans used time: " << sTrans << std::endl;
        cudaMemcpy(V_t, v_D, size, cudaMemcpyDeviceToHost);
    //dTrans
    cudaEventRecord(start);
    cudaMemcpy(v_D, V, size, cudaMemcpyHostToDevice);
    dymTrans<<<grid, block, n>>>(v_D, n);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaMemcpy(V_t, v_D, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float dTrans=0.0;
    cudaEventElapsedTime(&dTrans, start, stop);
    std::cout << "Dynamic trans used: " << dTrans << std::endl;
    if(dTrans<=sTrans&&dTrans<=nTrans){
            std::cout<<"Dymic trans is btetter than Static trans "<<sTrans-dTrans<<" ms "<<"and btetter than normal trans"<<nTrans-dTrans<<" ms "<<std::endl;
        }else if(sTrans<=dTrans&&sTrans<=nTrans){
            std::cout<<"Static trans is btetter than Dymic trans "<<dTrans-sTrans<<" ms "<<"and btetter than normal trans"<<nTrans-sTrans<<" ms "<<std::endl;
        }else{
            std::cout<<"Normal trans is btetter than Dymic trans "<<dTrans-nTrans<<" ms "<<"and btetter than Static trans"<<sTrans-nTrans<<" ms "<<std::endl;
        }
    free(V);
    cudaFree(V_t);
    return 0;
}