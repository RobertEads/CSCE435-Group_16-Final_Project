#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include <cuda.h>

using namespace std;

int THREADS;
int BLOCKS;
int NUM_VALS;

__global__ void generateData(int *dataArray, int size, int inputType)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    switch (inputType)
    {
    case 0:
    { // Random
        if (idx < size)
        {
            unsigned int x = 12345687 + idx;
            x ^= (x << 16);
            x ^= (x << 25);
            x ^= (x << 4);
            dataArray[idx] = abs(static_cast<int>(x) % size);
        }
        break;
    }
    case 1:
    { // Sorted
        if (idx < size)
        {
            dataArray[idx] = idx;
        }
        break;
    }
    case 2:
    { // Reverse sorted
        if (idx < size)
        {
            dataArray[idx] = size - 1 - idx;
        }
        break;
    }
    }
}

__device__ void merge(int *input, int *output, int left, int mid, int right)
{
    int i = left + threadIdx.x;
    int j = mid + threadIdx.x;
    int k = left + threadIdx.x;

    while (i < mid && j < right)
    {
        if (input[i] < input[j])
        {
            output[k++] = input[i++];
        }
        else
        {
            output[k++] = input[j++];
        }
    }

    while (i < mid)
    {
        output[k++] = input[i++];
    }

    while (j < right)
    {
        output[k++] = input[j++];
    }

    for (int idx = left + threadIdx.x; idx < right; idx += blockDim.x)
    {
        input[idx] = output[idx];
    }
}

__global__ void mergeSort(int *input, int *output, int left, int right)
{
    if (right - left <= 1)
    {
        return;
    }

    int mid = left + (right - left) / 2;

    mergeSort(input, output, left, mid);
    mergeSort(input, output, mid, right);

    merge(input, output, left, mid, right);
}

void launchMergeSort(int *d_unsortedArray, int *d_tempArray, int numVals)
{
    mergeSort<<<1, numVals>>>(d_unsortedArray, d_tempArray, 0, numVals);
}

__global__ void checkArraySorted(int *array, bool *isSorted, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1)
    {
        isSorted[idx] = (array[idx] <= array[idx + 1]);
    }
}

int main(int argc, char *argv[])
{
    int sortingType;

    sortingType = atoi(argv[1]);
    THREADS = atoi(argv[2]);
    NUM_VALS = atoi(argv[3]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Input sorting type: %d\n", sortingType);
    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    int *d_unsortedArray;
    int *d_tempArray;

    cudaMalloc((void **)&d_unsortedArray, NUM_VALS * sizeof(int));
    cudaMalloc((void **)&d_tempArray, NUM_VALS * sizeof(int));

    generateData<<<BLOCKS, THREADS>>>(d_unsortedArray, NUM_VALS, sortingType);
    cudaDeviceSynchronize();

    launchMergeSort(d_unsortedArray, d_tempArray, NUM_VALS);

    int sortedArray[NUM_VALS];
    cudaMemcpy(sortedArray, d_unsortedArray, NUM_VALS * sizeof(int), cudaMemcpyDeviceToHost);

    bool isSorted[NUM_VALS - 1];
    bool *d_isSorted;
    cudaMalloc((void **)&d_isSorted, (NUM_VALS - 1) * sizeof(bool));
    checkArraySorted<<<BLOCKS, THREADS>>>(d_unsortedArray, d_isSorted, NUM_VALS);
    cudaDeviceSynchronize();
    cudaMemcpy(isSorted, d_isSorted, (NUM_VALS - 1) * sizeof(bool), cudaMemcpyDeviceToHost);

    bool sorted = true;
    for (int i = 0; i < NUM_VALS - 1; i++)
    {
        if (!isSorted[i])
        {
            sorted = false;
            break;
        }
    }

    cudaFree(d_unsortedArray);
    cudaFree(d_tempArray);
    cudaFree(d_isSorted);

    if (sorted)
    {
        printf("The array is sorted!\n");
    }
    else
    {
        printf("The array is not sorted!\n");
    }

    return 0;
}
