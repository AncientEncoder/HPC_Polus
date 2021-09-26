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
    struct timeval start, end;
    gettimeofday( &start, NULL );
    float*A, *Ad, *B, *Bd, *C, *Cd;
    int n = 1024 * 1024;
    int size = n * sizeof(float);

    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    for(int i=0;i<n;i++){
        A[i] = 20.0;
        B[i] = 10.0;
    }
//CPU calc
    gettimeofday(&start,NULL);
     for(int i=0;i<n;i++){
        C[i] = A[i] + B[i];
    }
    gettimeofday( &end, NULL );
    float max_error = 0.0;
    for(int i=0;i<n;i++){
        max_error += fabs(30.0-C[i]);
    }
    cout << "max_error of CPU is " << max_error << endl;
    int timeuseCPU = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    cout << "total time for cpu is " << timeuseCPU/1000 << "ms" <<endl;
    //delete results
    for(int i=0;i<n;i++){
        C[i]=0;
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

    // check errors
    for(int i=0;i<n;i++)
    {
        max_error += fabs(30.0 - C[i]);
    }

    cout << "max error of GPU is " << max_error << endl;

    // free memory for gpu und host
    free(A);
    free(B);
    free(C);
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    int timeuseGPU = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    cout << "total time use in GPU is " << timeuseGPU/1000 << "ms" <<endl;
    if(timeuseGPU<timeuseCPU){
        cout<<"GPU is faster than CPU for "<<timeuseCPU-timeuseGPU<<" ms"<<endl;
    }
    return 0;
}