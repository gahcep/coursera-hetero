#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

#define IMAGE_CHANNELS 3

#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH + (Mask_width - 1))

__global__ void Kernel2DConvolution(const float * const inImage, float * const outImage,
										const float * __restrict__ mask, int imageWidth, int imageHeight)
{
	__device__ __shared__ float tileArr[BLOCK_WIDTH][BLOCK_WIDTH][IMAGE_CHANNELS];
		
	// scope: block thread
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	// Output indices
	int row_o = blockIdx.y*O_TILE_WIDTH + ty;
	int col_o = blockIdx.x*O_TILE_WIDTH + tx;
	
	// Input 
	int row_i = row_o - Mask_radius;
	int col_i = col_o - Mask_radius;
		
	// Phase 1: data gathering
	
	if ((row_i >= 0) && (row_i < imageHeight) &&
	    (col_i >= 0) && (col_i < imageWidth))
	{
		for(int ch = 0; ch < IMAGE_CHANNELS; ch++)
			tileArr[ty][tx][ch] = inImage[(row_i * imageWidth + col_i)*IMAGE_CHANNELS + ch];
	}
	else
	{
		for(int ch = 0; ch < IMAGE_CHANNELS; ch++)
			tileArr[ty][tx][ch] = 0.0;
	}
	
	__syncthreads();
	
	// Phase 2: calculation
	// Cond1: do not consider local thread indices greater than the amount of threads doing actual calculation (O_TILE_WIDTH)
	if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH)
	{
		float output[IMAGE_CHANNELS] = { 0.0, 0.0, 0.0};
				
		for (int i = 0; i < Mask_width; i++)
			for (int j = 0; j < Mask_width; j++)
				for (int ch = 0; ch < IMAGE_CHANNELS; ch++)
					output[ch] += mask[i * Mask_width + j] * tileArr[ty + i][tx + j][ch];
		
		// Cond2: output indices should lay within valid range
		if (row_o < imageHeight && col_o < imageWidth)
			for (int ch = 0; ch < IMAGE_CHANNELS; ch++)
				outImage[(row_o * imageWidth + col_o)*IMAGE_CHANNELS + ch] = output[ch];
	}
	
}
	

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    	
	dim3 GridDim((imageWidth - 1)/O_TILE_WIDTH + 1, (imageHeight - 1)/O_TILE_WIDTH + 1, 1);
	dim3 BlockDim(BLOCK_WIDTH, BLOCK_WIDTH);
	
	Kernel2DConvolution<<<GridDim, BlockDim>>>(deviceInputImageData, deviceOutputImageData, deviceMaskData, 
											  imageWidth, imageHeight);
	
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
