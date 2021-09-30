#include <iostream>
#include <cuda_profiler_api.h>
#include <sys/time.h>
#define STREAMS_NUM 8
__global__ void plus(float *a, float *b, float *c, int n, int offset) {

    int i = blockIdx.x*blockDim.x + threadIdx.x + offset;
        c[i] = a[i] + b[i];
}
int main(void){
    int n = 1024*1024;
    int size = n*sizeof(float);
    struct timeval start, end;
    float *a, *b;  
    float *c;

    cudaHostAlloc( (void**) &a, size ,cudaHostAllocDefault );
    cudaHostAlloc( (void**) &b, size ,cudaHostAllocDefault );
    cudaHostAlloc( (void**) &c, size ,cudaHostAllocDefault );

    float *a_d,*b_d,*c_d;

    for(int i=0; i < n; i++) {
        a[i] = 20.0;
        b[i] = 10.0;
    }
    cudaMalloc((void **)&a_d,size);
    cudaMalloc((void **)&b_d,size);
    cudaMalloc((void **)&c_d,size);
    const int StreamSize = n / STREAMS_NUM;
    cudaStream_t Stream[STREAMS_NUM];

    for (int i = 0; i < STREAMS_NUM; i++)
        cudaStreamCreate(&Stream[i]);

    dim3 block(1024);
    dim3 grid((n- 1)/1024 + 1);
    gettimeofday( &start, NULL );
    for ( int i = 0; i < STREAMS_NUM; i++) {

        int Offset = i * StreamSize;

        cudaMemcpyAsync(&a_d[Offset], &a[Offset], StreamSize * sizeof(float), cudaMemcpyHostToDevice, Stream[i]);
        cudaMemcpyAsync(&b_d[Offset], &b[Offset], StreamSize * sizeof(float), cudaMemcpyHostToDevice, Stream[i]);
        cudaMemcpyAsync(&c_d[Offset], &c[Offset], StreamSize * sizeof(float), cudaMemcpyHostToDevice, Stream[i]);

        plus<<<grid, block>>>(a_d, b_d, c_d, StreamSize, Offset);

        cudaMemcpyAsync(&a[Offset], &a_d[Offset], StreamSize * sizeof(float), cudaMemcpyDeviceToHost, Stream[i]);
        cudaMemcpyAsync(&b[Offset], &b_d[Offset], StreamSize * sizeof(float), cudaMemcpyDeviceToHost, Stream[i]);
        cudaMemcpyAsync(&c[Offset], &c_d[Offset], StreamSize * sizeof(float), cudaMemcpyDeviceToHost, Stream[i]);
    }
    gettimeofday(&end,NULL);
    int timeuseGPU = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    std::cout<<"total time use in GPU-Stream is "<<timeuseGPU<<" us "<<std::endl;
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
}