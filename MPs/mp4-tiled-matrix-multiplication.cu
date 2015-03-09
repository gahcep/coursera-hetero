#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILE_WIDTH 16

__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) 
{
	__device__ __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__device__ __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	// Row and Col of a cell in the resulting matrix
	int Row = by * blockDim.y + ty;
	int Col = bx * blockDim.x + tx;
	
	float cVal = 0.0;
	
	// Loop over all tiles
	for (int t = 0; t < (numAColumns - 1) / TILE_WIDTH + 1; t++)
	{
		if ((Row < numARows) && ((t * TILE_WIDTH + tx) < numAColumns))
			ds_A[ty][tx] = A[Row * numAColumns + t * TILE_WIDTH + tx];
		else
			ds_A[ty][tx] = 0.0;
		
		if ((Col < numBColumns) && ((t * TILE_WIDTH + ty) < numBRows))
			ds_B[ty][tx] = B[(t * TILE_WIDTH + ty) * numBColumns + Col];
		else
			ds_B[ty][tx] = 0.0;
		
		__syncthreads();
		
		for (int i = 0; i < TILE_WIDTH; i++)
			cVal += ds_A[ty][i] * ds_B[i][tx];
		__syncthreads();
	}
	
	if (Row < numARows && Col < numBColumns)
		C[Row * numBColumns + Col] = cVal;
}

int main(int argc, char **argv) {
	wbArg_t args;
	float *hostA; // The A matrix
	float *hostB; // The B matrix
	float *hostC; // The output C matrix
	float *deviceA;
	float *deviceB;
	float *deviceC;
	int numARows;    // number of rows in the matrix A
	int numAColumns; // number of columns in the matrix A
	int numBRows;    // number of rows in the matrix B
	int numBColumns; // number of columns in the matrix B
	int numCRows;    // number of rows in the matrix C (you have to set this)
	int numCColumns; // number of columns in the matrix C (you have to set this)
	
	int matrixAsize;
	int matrixBsize;
	int matrixCsize;
	
	args = wbArg_read(argc, argv);
	
	wbTime_start(Generic, "Importing data and creating memory on host");
	hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
	hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
	
	// Set numCRows and numCColumns
	numCRows = numARows;
	numCColumns = numBColumns;
	
	// Set sizes
	matrixAsize = numARows * numAColumns * sizeof(float);
	matrixBsize = numBRows * numBColumns * sizeof(float);
	matrixCsize = numCRows * numCColumns * sizeof(float);
	
	// Allocate the hostC matrix
	hostC = (float *) malloc(matrixCsize);
	wbTime_stop(Generic, "Importing data and creating memory on host");
	
	wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
	wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
	wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
	
	wbTime_start(GPU, "Allocating GPU memory.");
	cudaMalloc((void **) &deviceA, matrixAsize);
	cudaMalloc((void **) &deviceB, matrixBsize);
	cudaMalloc((void **) &deviceC, matrixCsize);
	
	// Zero fill the resulting array
	cudaMemset(deviceC, 0.0f, matrixCsize);
	
	wbTime_stop(GPU, "Allocating GPU memory.");
	
	wbTime_start(GPU, "Copying input memory to the GPU.");
	cudaMemcpy(deviceA, hostA, matrixAsize, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, matrixBsize, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceC, hostC, matrixCsize, cudaMemcpyHostToDevice);
	wbTime_stop(GPU, "Copying input memory to the GPU.");
	
	dim3 dimGrid((numCColumns-1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	
	wbTime_start(Compute, "Performing CUDA computation");
	matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC,
		numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
	
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");
	
	wbTime_start(Copy, "Copying output memory to the CPU");
	cudaMemcpy(hostC, deviceC, matrixCsize, cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying output memory to the CPU");
	
	wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
	wbTime_stop(GPU, "Freeing GPU Memory");
	
	wbSolution(args, hostC, numCRows, numCColumns);
	
	free(hostA);
	free(hostB);
	free(hostC);
	
	return 0;
}
