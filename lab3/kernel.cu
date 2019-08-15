/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE
__global__ void histo_kernel(unsigned int *buffer, long size, unsigned int *histo, unsigned int num_bins)
{
	extern __shared__ unsigned int histo_private[];
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	// stride is total number of threads
	unsigned int stride = blockDim.x * gridDim.x;
	// All threads handle blockDim.x * gridDim.x
	// consecutive elements

	//inititialize private histogram
	for (int j = 0; j < (num_bins-1)/blockDim.x+1; ++j)
		if (blockDim.x*j+threadIdx.x<num_bins)
			histo_private[blockDim.x*j+threadIdx.x]=0;
	__syncthreads();

	//populate private histogram
	while (i < size) {
		atomicAdd(&(histo_private[buffer[i]]), 1);
		i += stride;
	}
	__syncthreads();

	//Transfer data from shared memories to global memory
	for (int k = 0; k < (num_bins-1)/blockDim.x+1; ++k)
		if (blockDim.x*k+threadIdx.x<num_bins)
			atomicAdd(&(histo[blockDim.x*k+threadIdx.x]),
				histo_private[blockDim.x*k+threadIdx.x]);
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements,
        unsigned int num_bins) {

    // INSERT CODE HERE
    const int BLOCK_SIZE = 512;
    histo_kernel<<<(num_elements-1)/BLOCK_SIZE+1,BLOCK_SIZE,num_bins*sizeof(unsigned int)>>>(input,num_elements,bins,num_bins);
}


