/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
    

__global__ void VecAdd(int n, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A + B
     *   where A is a (1 * n) vector
     *   where B is a (1 * n) vector
     *   where C is a (1 * n) vector
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    long long start = clock64();
    long long cycles_elapsed;
    do { cycles_elapsed = clock64() - start; }
    while (cycles_elapsed < 20000);

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n){
        C[i] = A[i] + B[i];
    }

}

void streamVecAdd(float *A,  float *B, float *C, int n){
    const unsigned int BLOCK_SIZE = 512;
    cudaStream_t stream[3];
    float *d_A[3];
    float *d_B[3];
    float *d_C[3];
    unsigned long segSize = n/3;

    for (int i = 0; i < 2; ++i)
    {
        cudaMalloc((void **) &d_A[i], (segSize)*sizeof(float));
        cudaMalloc((void **) &d_B[i], (segSize)*sizeof(float));
        cudaMalloc((void **) &d_C[i], (segSize)*sizeof(float));
        //cudaStreamCreate(&stream[i]);
	cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
    }
    cudaMalloc((void **) &d_A[2], (segSize+n%3)*sizeof(float));
    cudaMalloc((void **) &d_B[2], (segSize+n%3)*sizeof(float));
    cudaMalloc((void **) &d_C[2], (segSize+n%3)*sizeof(float));
    //cudaStreamCreate(&stream[2]);
    cudaStreamCreateWithFlags(&stream[2], cudaStreamNonBlocking);

    cudaMemcpyAsync(d_A[0], A, segSize*sizeof(float),cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(d_B[0], B, segSize*sizeof(float),cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(d_A[1], A+segSize, segSize*sizeof(float),cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync(d_B[1], B+segSize, segSize*sizeof(float),cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync(d_A[2], A+2*segSize, (segSize+n%3)*sizeof(float),cudaMemcpyHostToDevice, stream[2]);
    cudaMemcpyAsync(d_B[2], B+2*segSize, (segSize+n%3)*sizeof(float),cudaMemcpyHostToDevice, stream[2]);

    VecAdd<<<(segSize-1)/BLOCK_SIZE+1, BLOCK_SIZE,0,stream[0]>>>(segSize,d_A[0],d_B[0],d_C[0]);
    VecAdd<<<(segSize-1)/BLOCK_SIZE+1, BLOCK_SIZE,0,stream[1]>>>(segSize,d_A[1],d_B[1],d_C[1]);
    VecAdd<<<(segSize+n%3-1)/BLOCK_SIZE+1, BLOCK_SIZE,0,stream[2]>>>(segSize+n%3,d_A[2],d_B[2],d_C[2]);

    cudaMemcpyAsync(C, d_C[0], segSize*sizeof(float),cudaMemcpyDeviceToHost, stream[0]);
    cudaMemcpyAsync(C+segSize, d_C[1], segSize*sizeof(float),cudaMemcpyDeviceToHost, stream[1]);
    cudaMemcpyAsync(C+2*segSize, d_C[2], (segSize+n%3)*sizeof(float),cudaMemcpyDeviceToHost, stream[2]);

    for (int i = 0; i < 3; ++i)
    {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
}


/*void basicVecAdd( float *A,  float *B, float *C, int n)
{

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = 512; 

    //INSERT CODE HERE
    VecAdd<<<(n-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(n,A,B,C);

}

*/
