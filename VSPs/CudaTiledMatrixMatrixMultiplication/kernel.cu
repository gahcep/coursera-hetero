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

#define TILE_WIDTH 16

__global__ void KernelMultiplyMatrix(float * matrixA, float * matrixB, float * matrixRes,
	int numMatrixARows, int numMatrixAColumns, int numMatrixBRows, int numMatrixBColumns,
	int numMatrixResRows, int numMatrixResColumns)
{
	// Allocate shared memory for the block
	__shared__ float store_a[TILE_WIDTH][TILE_WIDTH];
	__shared__ float store_b[TILE_WIDTH][TILE_WIDTH];

	// Aliases
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Block Size equals one Tile Width
	int Row = by * TILE_WIDTH + ty;	// or blockIdx.y * blockDim.y + threadIdx.y
	int Col = bx * TILE_WIDTH + tx; // or blockIdx.x * blockDim.x + threadIdx.x

	float resVal = 0.0;

	// Loop over TILEs count
	for (int t = 0; t < (numMatrixAColumns - 1) / TILE_WIDTH + 1; t++)
	{
		// Let all blocks copy data from global memory to shared for next tile
		// [row][col]
		if (Row < numMatrixARows && (t*TILE_WIDTH + tx) < numMatrixAColumns)
			// Can't use Col as it contains absolute 'path' as we need relative (within a block)
			store_a[ty][tx] = matrixA[Row*numMatrixAColumns + t*TILE_WIDTH + tx];
		else
			store_a[ty][tx] = 0.0;

		if (Col < numMatrixBColumns && (t*TILE_WIDTH + ty) < numMatrixBRows)
			// Can't use Row as it contains absolute 'path' as we need relative (within a block)
			store_b[ty][tx] = matrixB[(t*TILE_WIDTH + ty)*numMatrixBColumns + Col];
		else
			store_b[ty][tx] = 0.0;

		__syncthreads();

		for (int i = 0; i < TILE_WIDTH; i++)
			resVal += store_a[ty][i] * store_b[i][tx];

		__syncthreads();
	}

	if (Row < numMatrixResRows && Col < numMatrixResColumns)
		matrixRes[Row*numMatrixResColumns + Col] = resVal;
}

int main()
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
	dim3 GridDim((numMatrixResColumns - 1) / TILE_WIDTH + 1, (numMatrixResRows - 1) / TILE_WIDTH + 1, 1);
	dim3 BlockDim(TILE_WIDTH, TILE_WIDTH, 1);

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
