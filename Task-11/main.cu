#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <iostream>

__global__ void Plus(float* a, float* b, float* c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

void twoGPU(int size) {
  int n = size;
  int work_per_gpu = (n - 1) / 2 + 1;
  int nBytes = n * sizeof(float);
  int nBytes_per_gpu = work_per_gpu * sizeof(float);
  float *h_a, *h_b, *h_c;
  h_a = (float*)malloc(nBytes);
  h_b = (float*)malloc(nBytes);
  h_c = (float*)malloc(nBytes);
  cudaHostRegister(h_a, nBytes, 0);
  cudaHostRegister(h_b, nBytes, 0);
  cudaHostRegister(h_c, nBytes, 0);

  for (int i = 0; i < n; i++) {
    h_a[i] = i;
    h_b[i] = i + 1;
  }

  float *d_a0, *d_b0, *d_c0;
  float *d_a1, *d_b1, *d_c1;

  cudaSetDevice(0);
  cudaMalloc(&d_a0, nBytes_per_gpu);
  cudaMalloc(&d_b0, nBytes_per_gpu);
  cudaMalloc(&d_c0, nBytes_per_gpu);
  cudaSetDevice(1);
  cudaMalloc(&d_a1, nBytes_per_gpu);
  cudaMalloc(&d_b1, nBytes_per_gpu);
  cudaMalloc(&d_c1, nBytes_per_gpu);
  cudaSetDevice(0);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int BLOCK_SIZE = 1024;
  const int GRID_SIZE = (work_per_gpu - 1) / BLOCK_SIZE + 1;

  cudaEventRecord(start);

  cudaSetDevice(0);
  cudaMemcpyAsync(d_a0, &h_a[0], nBytes_per_gpu, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_b0, &h_b[0], nBytes_per_gpu, cudaMemcpyHostToDevice);

  Plus<<<GRID_SIZE, BLOCK_SIZE>>>(d_a0, d_b0, d_c0, n);

  cudaMemcpyAsync(&h_c[0], d_c0, nBytes_per_gpu, cudaMemcpyDeviceToHost);

  cudaSetDevice(1);
  cudaMemcpyAsync(d_a1, &h_a[work_per_gpu], nBytes_per_gpu,
                  cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_b1, &h_b[work_per_gpu], nBytes_per_gpu,
                  cudaMemcpyHostToDevice);
  Plus<<<GRID_SIZE, BLOCK_SIZE>>>(d_a1, d_b1, d_c1, n);

  cudaMemcpyAsync(&h_c[work_per_gpu], d_c1, nBytes_per_gpu,
                  cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  cudaSetDevice(0);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float twoGPU = 0;
  cudaEventElapsedTime(&twoGPU, start, stop);
  std::cout<<"Two GPUs run for :"<<twoGPU<<" ms "<<std::endl;

  cudaFree(d_a0);
  cudaFree(d_b0);
  cudaFree(d_c0);
  cudaSetDevice(1);
  cudaFree(d_a1);
  cudaFree(d_b1);
  cudaFree(d_c1);
  cudaSetDevice(0);
  cudaHostUnregister(h_a);
  cudaHostUnregister(h_b);
  cudaHostUnregister(h_c);
  free(h_a);
  free(h_b);
  free(h_c);
}

void oneGPU(int size) {
  int n = size;
  int nBytes = n * sizeof(float);

  float *h_a, *h_b, *h_c;

  h_a = (float*)malloc(nBytes);
  h_b = (float*)malloc(nBytes);
  h_c = (float*)malloc(nBytes);

  float *d_a, *d_b, *d_c;

  dim3 block(256);
  dim3 grid((unsigned int)ceil(n / (float)block.x));

  for (int i = 0; i < n; i++) {
    h_a[i] =20.0;
    h_b[i] = 10.0;
  }

  cudaMalloc((void**)&d_a, n * sizeof(float));
  cudaMalloc((void**)&d_b, n * sizeof(float));
  cudaMalloc((void**)&d_c, n * sizeof(float));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
  Plus<<<grid, block>>>(d_a, d_b, d_c, n);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float sigTime = 0;
  cudaEventElapsedTime(&sigTime, start, stop);
  std::cout<<"One GPU runs for :"<<sigTime<<" ms "<<std::endl;

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);
}

int main(int argc, char* argv[]) {
  assert(argc==2);
  oneGPU(atoi(argv[1]));
  twoGPU(atoi(argv[1]));
  return 0;
}
