#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

static unsigned char *h_dInputImage;
static unsigned char *h_dOutputImage;
static int hWidth = 0;
static int hHeight = 0;

struct lastMedianStruct
{
    unsigned char value;
    int numberOfItemsOnLeft;
};

// Check for CUDA errors
inline static cudaError_t
checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(
            stderr,
            "Line:%d: CUDA Runtime Error: %s\n",
            __LINE__,
            cudaGetErrorString(result)
        );
        assert(result == cudaSuccess);
    }
    return result;
}

// Add a window to the histogram
__device__ static void
d_addToHistogram(const unsigned char *window,
                 int *histogram,
                 int filterSize,
                 int length,
                 struct lastMedianStruct *lastMedian)
{
    const unsigned int numberOfZeros = (filterSize - length) *
                                       (int)(threadIdx.x == 0);

    for (int i = threadIdx.x; i < length; i += blockDim.x)
    {
        atomicAdd(&histogram[window[i]], 1);
        if (window[i] < lastMedian->value)
        {
            atomicAdd(&(lastMedian->numberOfItemsOnLeft), 1);
        }
    }

    if (0 < lastMedian->value && threadIdx.x == 0)
    {
        atomicAdd(
            &(lastMedian->numberOfItemsOnLeft),
            numberOfZeros
        );
    }
    atomicAdd(&histogram[threadIdx.x], numberOfZeros);
}

// Remove a window from the histogram
__device__ static void
d_removeFromHistogram(const unsigned char *window,
                      int *histogram,
                      int filterSize,
                      int length,
                      struct lastMedianStruct *lastMedian)
{
    const unsigned int numberOfZeros = (filterSize - length) *
                                       (int)(threadIdx.x == 0);

    for (int i = threadIdx.x; i < length; i += blockDim.x)
    {
        atomicSub(&histogram[window[i]], 1);
        if (window[i] < lastMedian->value)
        {
            atomicSub(&(lastMedian->numberOfItemsOnLeft), 1);
        }
    }

    if (0 < lastMedian->value && threadIdx.x == 0)
    {
        atomicSub(&(lastMedian->numberOfItemsOnLeft), numberOfZeros);
    }
    atomicSub(&histogram[threadIdx.x], numberOfZeros);
}

// Get the median value from the histogram
__device__ static void
d_getMedian(const int *histogram,
            int filterSize,
            struct lastMedianStruct *lastMedian)
{
    // Calculate the median
    int sum = lastMedian->numberOfItemsOnLeft;
    const int medianLocation = (filterSize * filterSize + 1) / 2;
    const int direction = (int)(sum < medianLocation) - (int)(sum >= medianLocation);

    for (int i = lastMedian->value - (int)(direction == -1);; i += direction)
    {
        sum += histogram[i] * direction;
        if ((sum >= medianLocation) == (direction == 1))
        {
            lastMedian->value = i;
            lastMedian->numberOfItemsOnLeft = sum - histogram[i] *
                                              (int)(direction == 1);
            return;
        }
    }

    return;
}

// Kernel function for median filter
__global__ static void
medianFilter(const unsigned char *inputImage,
             unsigned char *outputImage,
             int width,
             int height,
             int filterSize)
{
    extern __shared__ char sharedMem[];
    int *histogram = (int *)&sharedMem[(threadIdx.y) * 256 * sizeof(int)];
    __shared__ struct lastMedianStruct lastMedianShared[16];
    struct lastMedianStruct *lastMedian = &lastMedianShared[threadIdx.y];
    const int index = blockIdx.x * blockDim.y + threadIdx.y;
    int leftBound;
    int length;
    const int wingLength = (filterSize - 1) / 2;

    // Check if the pixel is within the image boundaries
    if (index >= width)
    {
        return;
    }

    // Clean the histogram
    memset(
        histogram + threadIdx.x * (256 / blockDim.x),
        0,
        256 / blockDim.x * sizeof(int)
    );

    // Initialize the last median cache
    if (threadIdx.x == 0)
    {
      lastMedian->value = 0;
      lastMedian->numberOfItemsOnLeft = 0;
    }

    __syncthreads();

    // Calculate the left bound of the window
    leftBound = max(index - wingLength, 0);
    // Calculate the length of the window
    length = min(wingLength + 1 + index, width) - leftBound;

    // Preload the window and add to histogram
    const int preloadRows = min(height, wingLength);
    const unsigned char *const leftInputImage = inputImage + leftBound;
    for (int row = 0; row < preloadRows; row++)
    {
        d_addToHistogram(
            leftInputImage + row * width,
            histogram,
            filterSize,
            length,
            lastMedian
        );
    }

    // Add empty windows to histogram for out of bounds pixels
    // on the top and the bottom
    d_addToHistogram(
        NULL,
        histogram,
        filterSize * (filterSize - preloadRows),
        0,
        lastMedian
    );

    // Calculate median for the pixel
    for (int row = 0; row < height; row++)
    {
        // Remove the top row from the histogram
        d_removeFromHistogram(
            leftInputImage + (row - wingLength - 1) * width,
            histogram,
            filterSize,
            length * (int)(row > wingLength),
            lastMedian
        );

        // Add the bottom row to the histogram
        d_addToHistogram(
            leftInputImage + (row + wingLength) * width,
            histogram,
            filterSize,
            length * (int)(row + wingLength < height),
            lastMedian
        );

        __syncthreads();

        if (threadIdx.x == 0)
        {
            // Store the filtered pixel value in the output image
            d_getMedian(
                histogram,
                filterSize,
                lastMedian
            );
            outputImage[row * width + index] = lastMedian->value;
        }

        __syncthreads();
    }
    return;
}

// Allocate memory for GPU
void
cudaAllocateMemory(int pWidth, int pHeight)
{
    int imageSize = pWidth * pHeight * sizeof(unsigned char);

    // Set image width and height
    hWidth = pWidth;
    hHeight = pHeight;

    // Free device memory if it has already been allocated
    if (h_dInputImage != NULL)
    {
        checkCuda(cudaFree(h_dInputImage));
    }
    if (h_dOutputImage != NULL)
    {
        checkCuda(cudaFree(h_dOutputImage));
    }

    // Allocate device memory
    checkCuda(cudaMalloc((void **)&h_dInputImage, imageSize));
    checkCuda(cudaMalloc((void **)&h_dOutputImage, imageSize));

    return;
}

// Upload image to GPU memory
void
cudaUploadImage(const unsigned char *hInputImage)
{
    int imageSize = hWidth * hHeight * sizeof(unsigned char);

    // Check if memory has been allocated
    if (hWidth == 0 || hHeight == 0 ||
        h_dInputImage == NULL || h_dOutputImage == NULL)
    {
        fprintf(stderr, "Memory has not been allocated.\n");
        return;
    }

    // Copy input image to device memory
    checkCuda(cudaMemcpy(
        h_dInputImage,
        hInputImage,
        imageSize,
        cudaMemcpyHostToDevice
    ));

    return;
}

// Download output image from GPU memory
void
cudaDownloadImage(unsigned char *hOutputImage)
{
    // Check if the image has been loaded
    if (hWidth == 0 || hHeight == 0 ||
        h_dInputImage == NULL || h_dOutputImage == NULL)
    {
        fprintf(stderr, "Image has not been loaded to GPU memory.\n");
        return;
    }

    int imageSize = hWidth * hHeight * sizeof(unsigned char);

    // Copy output image from device memory
    checkCuda(cudaMemcpy(
        hOutputImage,
        h_dOutputImage,
        imageSize,
        cudaMemcpyDeviceToHost
    ));

    return;
}

// Median filter function
void
cudaMedianFilter(int filterSize)
{
    // Check if the image has been loaded
    if (hWidth == 0 || hHeight == 0 ||
        h_dInputImage == NULL || h_dOutputImage == NULL)
    {
        fprintf(stderr, "Image has not been loaded to GPU memory.\n");
        return;
    }

    dim3 threadsPerBlock(1,32);
    dim3 gridSize(1,1);

    if (filterSize == 3)
    {
        threadsPerBlock.x = 4; // Number of thread per column/row of the image
        threadsPerBlock.y = 8; // Number of columns processed in the block
    }
    else
    {
      threadsPerBlock.x = 8; // Number of thread per column/row of the image
      threadsPerBlock.y = 4; // Number of columns processed in the block
    }
    gridSize.x = (hWidth + threadsPerBlock.y - 1) / (threadsPerBlock.y);

    medianFilter<<<
        gridSize,
        threadsPerBlock,
        threadsPerBlock.y * 256 * sizeof(int)
    >>>(
        h_dInputImage,
        h_dOutputImage,
        hWidth,
        hHeight,
        filterSize
    );

    return;
}