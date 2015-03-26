// CUDA API
#include <cuda_runtime.h>
// Built-in variables (gridDim, blockDim, blockIdx, threadIdx)
#include <device_launch_parameters.h>

// Load data from file
#include "../DataLoader.hpp"

#include <iostream>
#include <vector>
#include <sstream>
#include <string>

using namespace std;
using namespace DataLoader;

/*
* Kernel: Matrix Multiplication
*/

__global__ void KernelMultiplyMatrix(float * matrixA, float * matrixB, float * matrixRes,
	int numMatrixARows, int numMatrixAColumns, int numMatrixBRows, int numMatrixBColumns,
	int numMatrixResRows, int numMatrixResColumns)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < numMatrixResRows && col < numMatrixResColumns)
	{
		float valRes = 0.0;

		for (int i = 0; i < numMatrixARows; i++)
		{
			valRes += matrixA[row * numMatrixAColumns + i] * matrixB[numMatrixBColumns * i + col];
		}

		matrixRes[row * numMatrixBColumns + col] = valRes;
	}
}

int main(int argc, char** argv)
{
	// Host matrixes
	float * hostMatrixA;
	float * hostMatrixB;
	float * hostMatrixRes;

	// Device matrixes
	float * devMatrixA;
	float * devMatrixB;
	float * devMatrixRes;

	// Dimentions
	int numMatrixARows;
	int numMatrixAColumns;
	int numMatrixBRows;
	int numMatrixBColumns;
	int numMatrixResRows;
	int numMatrixResColumns;

	// Sizes
	size_t MATRIX_A_LENGTH;
	size_t MATRIX_B_LENGTH;
	size_t MATRIX_RES_LENGTH;

	// Prepare input data
	Loader<float> loaderSet1("./InputData/input0.raw");
	Loader<float> loaderSet2("./InputData/input1.raw");

	loaderSet1.delimeter(' ');
	loaderSet2.delimeter(' ');

	loaderSet1.read_matrix(Flags::NullOpts, Flags_Matrix_Parse::DimentionsAtOnce);
	loaderSet2.read_matrix(Flags::NullOpts, Flags_Matrix_Parse::DimentionsAtOnce);

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

	// Set dimentions
	loaderSet1.arg_matrix_dims(numMatrixARows, numMatrixAColumns);
	loaderSet2.arg_matrix_dims(numMatrixBRows, numMatrixBColumns);
	numMatrixResRows = numMatrixARows;
	numMatrixResColumns = numMatrixBColumns;

	// Set sizes
	MATRIX_A_LENGTH = numMatrixARows * numMatrixAColumns * sizeof(float);
	MATRIX_B_LENGTH = numMatrixBRows * numMatrixBColumns * sizeof(float);
	MATRIX_RES_LENGTH = numMatrixResRows * numMatrixResColumns * sizeof(float);

	// Allocate host memory
	hostMatrixA = (float *)malloc(MATRIX_A_LENGTH);
	hostMatrixB = (float *)malloc(MATRIX_B_LENGTH);
	hostMatrixRes = (float *)malloc(MATRIX_RES_LENGTH);

	// Allocate device memory
	cudaMalloc((float **)&devMatrixA, MATRIX_A_LENGTH);
	cudaMalloc((float **)&devMatrixB, MATRIX_B_LENGTH);
	cudaMalloc((float **)&devMatrixRes, MATRIX_RES_LENGTH);

	// Load data
	loaderSet1.arg_matrix(hostMatrixA);
	loaderSet2.arg_matrix(hostMatrixB);

	// Copy data from Host to Device
	cudaMemcpy(devMatrixA, hostMatrixA, MATRIX_A_LENGTH, cudaMemcpyHostToDevice);
	cudaMemcpy(devMatrixB, hostMatrixB, MATRIX_B_LENGTH, cudaMemcpyHostToDevice);

	// Set Grid and Block dimensions
	int tile = 16;
	dim3 GridDim((numMatrixResColumns - 1) / tile + 1, (numMatrixResRows - 1) / tile + 1, 1);
	dim3 BlockDim(tile, tile, 1);

	// Run the kernel
	KernelMultiplyMatrix<<<GridDim, BlockDim>>>(devMatrixA, devMatrixB, devMatrixRes,
		numMatrixARows, numMatrixAColumns, numMatrixBRows, numMatrixBColumns, numMatrixResRows, numMatrixResColumns);

	// Wait for kernel to finish
	cudaDeviceSynchronize();

	// Copy result data from Device to Host
	cudaMemcpy(hostMatrixRes, devMatrixRes, MATRIX_RES_LENGTH, cudaMemcpyDeviceToHost);

	// Clean resources
	free(hostMatrixA);
	free(hostMatrixB);
	free(hostMatrixRes);

	cudaFree(devMatrixA);
	cudaFree(devMatrixB);
	cudaFree(devMatrixRes);

	exit(0);
}