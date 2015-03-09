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

__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {
  
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
		
	if (row < numCRows && col < numCColumns)
	{
		float valC = 0.0;
		
		for(int i = 0; i < numAColumns; i++)
			valC += A[row * numAColumns + i] * B[numBColumns * i + col];
				
		C[row * numBColumns + col] = valC;
	}	
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
	
	int SIZE_MATRIX_A;
	int SIZE_MATRIX_B;
	int SIZE_MATRIX_C;
	
	args = wbArg_read(argc, argv);
	
	wbTime_start(Generic, "Importing data and creating memory on host");
	hostA =
		( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
	hostB =
		( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
	
	// Set up result matrix dimentions
	numCRows = numARows;
	numCColumns = numBColumns;
	
	// Calc sizes
	SIZE_MATRIX_A = numARows * numAColumns * sizeof(float);
	SIZE_MATRIX_B = numBRows * numBColumns * sizeof(float);
	SIZE_MATRIX_C = numCRows * numCColumns * sizeof(float);
	
	// Allocate the hostC matrix
	hostC = (float *) malloc(SIZE_MATRIX_C);
	
	wbTime_stop(Generic, "Importing data and creating memory on host");
	
	wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
	wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
	wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
	
	wbTime_start(GPU, "Allocating GPU memory.");
	cudaMalloc((void **) &deviceA, SIZE_MATRIX_A);
	cudaMalloc((void **) &deviceB, SIZE_MATRIX_B);
	cudaMalloc((void **) &deviceC, SIZE_MATRIX_C);
	wbTime_stop(GPU, "Allocating GPU memory.");
	
	wbTime_start(GPU, "Copying input memory to the GPU.");
	cudaMemcpy(deviceA, hostA, SIZE_MATRIX_A, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, SIZE_MATRIX_B, cudaMemcpyHostToDevice);
	wbTime_stop(GPU, "Copying input memory to the GPU.");
	
	int tile = 16;
	dim3 DimGrid((numCColumns - 1) / tile + 1, (numCRows - 1) / tile + 1, 1);
	dim3 DimBlock(tile, tile, 1);
	
	wbTime_start(Compute, "Performing CUDA computation");
	
	matrixMultiply<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, 
		numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
		
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");
	
	wbTime_start(Copy, "Copying output memory to the CPU");
	cudaMemcpy(hostC, deviceC, SIZE_MATRIX_C, cudaMemcpyDeviceToHost);
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
