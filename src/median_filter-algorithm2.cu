#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_THREAD_PER_COL 8
#define THREAD_PER_BLOCK 32
#define PARALLELISM 8

static unsigned char *h_dInputImage;
static unsigned char *h_dOutputImage;
static int hWidth = 0;
static int hHeight = 0;

struct compositeHistogramStruct
{
    int histogram[256];
    int numberOfItemsOnLeft;
    unsigned char lastMedianValue;
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
__device__ static int
d_addToHistogram(const unsigned char *window,
                 int *histogram,
                 int length,
                 unsigned char lastMedianValue)
{
    int numberOfItemsOnLeft = 0;

    for (int i = threadIdx.x; i < length; i += blockDim.x)
    {
        atomicAdd(&histogram[window[i]], 1);
        numberOfItemsOnLeft += (int)(window[i] < lastMedianValue);
    }

    return numberOfItemsOnLeft;
}

// Remove a window from the histogram
__device__ static int
d_removeFromHistogram(const unsigned char *window,
                      int *histogram,
                      int length,
                      unsigned char lastMedianValue)
{
    int numberOfItemsOnLeft = 0;

    for (int i = threadIdx.x; i < length; i += blockDim.x)
    {
        atomicSub(&histogram[window[i]], 1);
        numberOfItemsOnLeft -= (int)(window[i] < lastMedianValue);
    }

    return numberOfItemsOnLeft;
}

// Get the median value from the histogram
__device__ static unsigned char
d_getMedian(const int *histogram,
            int histogramSize,
            unsigned char lastMedianValue,
            int *numberOfItemsOnLeft)
{
    // Calculate the median
    int sum = *numberOfItemsOnLeft;
    const int medianLocation = (histogramSize + 1) / 2;
    const int direction = (int)(sum < medianLocation) - (int)(sum >= medianLocation);

    for (int i = lastMedianValue - (int)(direction == -1);; i += direction)
    {
        sum += histogram[i] * direction;
        if ((sum >= medianLocation) == (direction == 1))
        {
            *numberOfItemsOnLeft = sum - histogram[i] * (int)(direction == 1);
            return i;
        }
    }
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
    struct compositeHistogramStruct *compositeHistogram =
        (struct compositeHistogramStruct *)sharedMem;
    struct compositeHistogramStruct *localCompositeHistogram =
        &compositeHistogram[threadIdx.y];

    int *histogram = localCompositeHistogram->histogram;
    unsigned char *lastMedianValue = &localCompositeHistogram->lastMedianValue;
    int *numberOfItemsOnLeft = &localCompositeHistogram->numberOfItemsOnLeft;
    int histogramSize;
    unsigned char localLastMedianValue = 0;
    int localNumberOfItemsOnLeft = 0;

    const int index = blockIdx.x * blockDim.y + threadIdx.y;
    const int col_split_no = blockIdx.y;
    const int number_of_splits = gridDim.y;
    const int number_of_rows = (height + number_of_splits - 1) / number_of_splits;

    int leftBound;
    int length;
    int removedRowLength;
    int addedRowLength;
    const int wingLength = (filterSize - 1) / 2;

    // Check if the pixel is within the image boundaries
    if (index >= width)
    {
        return;
    }

    // Calculate the left bound of the window
    leftBound = max(index - wingLength, 0);
    // Calculate the length of the window
    length = min(wingLength + 1 + index, width) - leftBound;

    // Preload the window and add to histogram
    const int preloadRowsStart = max(
        0,
        (int)(number_of_rows * col_split_no) - wingLength - 1
    );
    const int preloadRowsEnd = min(
        height,
        (int)(number_of_rows * col_split_no) + wingLength
    );
    const unsigned char *const leftInputImage = inputImage + leftBound;

    // Initialize the last median cache
    if (threadIdx.x == 0)
    {
        // Clean the histogram
        memset(histogram, 0, 256 * sizeof(int));
        *lastMedianValue = 0;
        *numberOfItemsOnLeft = 0;
        histogramSize = length * (preloadRowsEnd - preloadRowsStart);
    }

    __syncthreads();

    for (int row = preloadRowsStart; row < preloadRowsEnd; row++)
    {
        localNumberOfItemsOnLeft += d_addToHistogram(
            leftInputImage + row * width,
            histogram,
            length,
            localLastMedianValue
        );
    }

    const int startingRow = number_of_rows * col_split_no;
    const int endingRow = min(height, number_of_rows * (col_split_no + 1));

    // Calculate median for the pixel
    for (int row = startingRow; row < endingRow; row++)
    {
        removedRowLength = length * (int)(row > wingLength);
        addedRowLength = length * (int)(row + wingLength < height);
        localLastMedianValue = *lastMedianValue;

        // Remove the top row from the histogram
        localNumberOfItemsOnLeft += d_removeFromHistogram(
            leftInputImage + (row - wingLength - 1) * width,
            histogram,
            removedRowLength,
            localLastMedianValue
        );

        // Add the bottom row to the histogram
        localNumberOfItemsOnLeft += d_addToHistogram(
            leftInputImage + (row + wingLength) * width,
            histogram,
            addedRowLength,
            localLastMedianValue
        );

        if (localNumberOfItemsOnLeft != 0)
        {
            atomicAdd(numberOfItemsOnLeft, localNumberOfItemsOnLeft);
            localNumberOfItemsOnLeft = 0;
        }

        __syncthreads();

        if (threadIdx.x == 0)
        {
            histogramSize += addedRowLength - removedRowLength;
            // Store the filtered pixel value in the output image
            *lastMedianValue = d_getMedian(
                histogram,
                histogramSize,
                localLastMedianValue,
                numberOfItemsOnLeft
            );
            outputImage[row * width + index] = *lastMedianValue;
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

    dim3 threadsPerBlock(1, 2);
    dim3 gridSize(1, PARALLELISM);

    threadsPerBlock.x = min(
        (int)pow(2, (int)(log(filterSize) / log(2)) + 1),
        MAX_THREAD_PER_COL
    ); // Number of thread per column/row of the image
    threadsPerBlock.y = max(1, THREAD_PER_BLOCK / threadsPerBlock.x);

    gridSize.x = (hWidth + threadsPerBlock.y - 1) / (threadsPerBlock.y);

    medianFilter<<<
        gridSize,
        threadsPerBlock,
        threadsPerBlock.y * sizeof(struct compositeHistogramStruct)
    >>>(
        h_dInputImage,
        h_dOutputImage,
        hWidth,
        hHeight,
        filterSize
    );

    return;
}