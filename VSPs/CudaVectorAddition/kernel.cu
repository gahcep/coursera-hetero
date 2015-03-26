// CUDA API
#include "cuda_runtime.h"
// Built-in variables (gridDim, blockDim, blockIdx, threadIdx)
#include "device_launch_parameters.h"

// Load data from file
#include "../DataLoader.hpp"

#include <iostream>
#include <vector>
#include <sstream>
#include <string>

using namespace std;
using namespace DataLoader;

/*
* Kernel: Vector Addition
*/

__global__ void KernelAddVec(float * vecA, float * vecB, float * vecRes, size_t length) {
	// Get index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Check boundaries
	if (idx < length) {
		vecRes[idx] = vecA[idx] + vecB[idx];
	}
}

int main(int argc, char** argv)
{
	// Host arrays
	float * hostVecA;
	float * hostVecB;
	float * hostVecRes;

	// Device arrays
	float * devVecA;
	float * devVecB;
	float * devVecRes;

	size_t VECTOR_LENGTH = 100;
	int VECTOR_SIZE = VECTOR_LENGTH * sizeof(float);

	// Prepare input data
	Loader<float> loaderSet1("./InputData/input0.raw");
	Loader<float> loaderSet2("./InputData/input1.raw");

	loaderSet1.delimeter(';');
	loaderSet2.delimeter(';');
	loaderSet1.read_vector(Flags::NullOpts, DataLoader::Flags_Vector_Parse::Size | DataLoader::Flags_Vector_Parse::ItemsLineByLine);
	loaderSet2.read_vector(Flags::NullOpts, DataLoader::Flags_Vector_Parse::Size | DataLoader::Flags_Vector_Parse::ItemsLineByLine);

	if (!loaderSet1.is_ok())
	{
		loaderSet1.refresh();
		exit(1);
	}

	if (!loaderSet2.is_ok())
	{
		loaderSet2.refresh();
		exit(1);
	}

	// Allocate host memory
	hostVecA = (float *)malloc(VECTOR_SIZE);
	hostVecB = (float *)malloc(VECTOR_SIZE);
	hostVecRes = (float *)malloc(VECTOR_SIZE);

	// Allocate device memory
	cudaMalloc((void **)&devVecA, VECTOR_SIZE);
	cudaMalloc((void **)&devVecB, VECTOR_SIZE);
	cudaMalloc((void **)&devVecRes, VECTOR_SIZE);

	// Load data
	loaderSet1.arg_vector(0, hostVecA);
	loaderSet2.arg_vector(0, hostVecB);

	// Copy data from Host to Device
	cudaMemcpy(devVecA, hostVecA, VECTOR_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(devVecB, hostVecB, VECTOR_SIZE, cudaMemcpyHostToDevice);

	// Set Grid and Block dimensions
	dim3 GridDim((VECTOR_LENGTH - 1) / 256 + 1, 1, 1);
	dim3 BlockDim(256, 1, 1);

	// Run the kernel
	KernelAddVec<<<GridDim, BlockDim>>>(devVecA, devVecB, devVecRes, VECTOR_LENGTH);

	// Wait for kernel to finish
	cudaDeviceSynchronize();

	// Copy result data from Device to Host
	cudaMemcpy(hostVecRes, devVecRes, VECTOR_SIZE, cudaMemcpyDeviceToHost);

	// Clean resources
	free(hostVecA);
	free(hostVecB);
	free(hostVecRes);

	cudaFree(devVecA);
	cudaFree(devVecB);
	cudaFree(devVecRes);

	exit(0);
}
