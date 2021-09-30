#include "cuda_runtime.h"
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

__global__ void Plus(float A[], float B[], float C[], int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(void){
    int n = 1024*1024;
    struct timeval start, end;
    gettimeofday( &start, NULL );
    float*A, *Ad, *B, *Bd, *C, *Cd;
    int size = n * sizeof(float);

    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    for(int i=0;i<n;i++){
        A[i] = 20.0;
        B[i] = 10.0;
    }
    //GPU calc
    cudaMalloc((void**)&Ad, size);
    cudaMalloc((void**)&Bd, size);
    cudaMalloc((void**)&Cd, size);

    // copy to device 
    cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
    int Block=1024;
    int Grid=(n-1)/Block+1;

    // exec
    gettimeofday(&start,NULL);
    Plus<<<Grid, Block>>>(Ad, Bd, Cd, n);
    gettimeofday(&end,NULL);
    // return result to host 
    cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);

    // free memory for gpu und host
    free(A);
    free(B);
    free(C);
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    int timeuseGPU = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    cout << "total time use in GPU-Normal is " << timeuseGPU<< " us " <<endl;
    return 0;
}