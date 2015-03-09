#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main()
{
	cudaError_t err;

	// Device number
	int deviceCount = 0;
	err = cudaGetDeviceCount(&deviceCount);

	if (err != cudaSuccess)
		return 1;

	/*
		CUDA 3.0
		totalGlobalMem = 2GB
		sharedMemPerBlock = 49152 bytes
		regsPerBlock = 65536
		warpSize = 32
		maxThreadsPerBlock = 1024	
		maxThreadsDim = {1024, 1024, 64}
		maxGridSize	= {2147483647, 65535, 65535}
		totalConstMem = 65536
	*/

	cudaDeviceProp props;
	for (int i = 0; i < deviceCount; i++)
	{
		err = cudaGetDeviceProperties(&props, i);
		if (err != cudaSuccess)
			return 1;
	}
}
