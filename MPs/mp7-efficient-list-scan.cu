#include    <wb.h>

#define BLOCK_SIZE 1024

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
    
__global__ void scan(float * input, float * output, int len, int offset, float acc) {
    __shared__ float storage[2 * BLOCK_SIZE];
	
	int block_idx = threadIdx.x;
	int global_idx = blockIdx.x * blockDim.x + block_idx;
	
	// 1. Load data into shared memory
	if (global_idx < (len - offset))
	{
		storage[block_idx] = input[offset + global_idx];
		if ((BLOCK_SIZE + global_idx) <= (len - offset))
			storage[BLOCK_SIZE + block_idx] = input[offset + BLOCK_SIZE + global_idx];
	}
	else
		storage[block_idx] = 0.0;
	
	// 2. Reduction Phase
	for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2)
	{
		__syncthreads();
		
		int index = (block_idx + 1) * stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE)
			storage[index] += storage[index - stride]; 
	}	
	
	// 3. Post-Reduction (Reverse Reduction) Phase
	for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2)
	{
		__syncthreads();
		
		int index = (block_idx + 1) * stride * 2 - 1;
		if ((stride + index) < 2 * BLOCK_SIZE)
			storage[index + stride] += storage[index];
	}
	
	__syncthreads();
	
	// 4. Write final results
	if (global_idx < (len - offset))
	{
		output[offset + global_idx] = storage[block_idx] + acc;
		if ((BLOCK_SIZE + global_idx) <= (len - offset))
			output[offset + BLOCK_SIZE + global_idx] = storage[BLOCK_SIZE + block_idx] + acc;
	}	
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
	
	int offset, last_item_pos, runs;
	float acc = 0.0;
	
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //dim3 GridDim((numElements - 1) / 2 * BLOCK_SIZE + 1, 1, 1);
	dim3 GridDim(1, 1, 1);
	dim3 BlockDim(BLOCK_SIZE, 1);

    wbTime_start(Compute, "Performing CUDA computation");
	
	runs = (numElements - 1) / (2 * BLOCK_SIZE) + 1;
	
	for (int i = 1; i <= runs; i++)
	{
		offset = (i - 1) * 2 * BLOCK_SIZE;
		
		scan<<<GridDim, BlockDim>>>(deviceInput, deviceOutput, numElements, offset, acc);
		
		cudaDeviceSynchronize();
		
		wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
		
		// last run?
		if (i != runs)
		{
			last_item_pos = numElements < i * 2 * BLOCK_SIZE ? numElements - 1 : i * 2 * BLOCK_SIZE - 1;
			acc = hostOutput[last_item_pos];
		}
	}
	
	wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

