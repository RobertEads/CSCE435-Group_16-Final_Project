#include <stdio.h>
#include <stdlib.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

/* Define Caliper region names */
const char *mainFunction = "main";
const char *data_init = "data_init";
const char *correctness_check = "correctness_check ";
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comm_small = "comm_small";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comp_small = "comp_small";

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

__device__ void merge(int *arr, int *temp, int left, int mid, int right)
{
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right)
    {
        if (arr[i] <= arr[j])
        {
            temp[k++] = arr[i++];
        }
        else
        {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid)
    {
        temp[k++] = arr[i++];
    }

    while (j <= right)
    {
        temp[k++] = arr[j++];
    }

    for (int i = left; i <= right; i++)
    {
        arr[i] = temp[i];
    }
}

__global__ void mergeSort(int *arr, int *temp, int size)
{
    for (int currSize = 1; currSize <= size - 1; currSize = 2 * currSize)
    {
        for (int leftStart = 0; leftStart < size - 1; leftStart += 2 * currSize)
        {
            int mid = min(leftStart + currSize - 1, size - 1);
            int rightEnd = min(leftStart + 2 * currSize - 1, size - 1);
            merge(arr, temp, leftStart, mid, rightEnd);
        }
    }
}

void printArray(int *arr, int size)
{
    printf("Array: ");
    for (int i = 0; i < size; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

bool isSorted(int *arr, int size)
{
    for (int i = 0; i < size - 1; i++)
    {
        if (arr[i] > arr[i + 1])
        {
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv)
{

    if (argc != 4)
    {
        printf("Usage: %s <sorting_type> <num_processors> <num_elements>\n", argv[0]);
        return 1;
    }

    int sortingType = atoi(argv[1]);
    int numProcessors = atoi(argv[2]);
    int numElements = atoi(argv[3]);

    const char *sorting_type_name;
    switch (sortingType)
    {
    case 0:
        sorting_type_name = "Random";
        break;
    case 1:
        sorting_type_name = "Sorted";
        break;
    case 2:
        sorting_type_name = "ReverseSorted";
        break;
    default:
        sorting_type_name = "Unknown";
        break;
    }

    int *h_arr = new int[numElements];
    int *d_arr;
    int *temp;

    CALI_MARK_BEGIN(mainFunction);
    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN(data_init);

    // Call generateData kernel to initialize array based on sorting type
    int *d_generateResult;
    cudaMalloc((void **)&d_generateResult, sizeof(int) * numElements);
    generateData<<<(numElements + 255) / 256, 256>>>(d_generateResult, numElements, sortingType);
    CALI_MARK_END(data_init);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    CALI_MARK_BEGIN("cudaMemcpy");

    cudaMemcpy(h_arr, d_generateResult, sizeof(int) * numElements, cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy");

    cudaFree(d_generateResult);

    CALI_MARK_END(comm_small);

    CALI_MARK_BEGIN(comm_large);
    cudaMalloc((void **)&d_arr, sizeof(int) * numElements);
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(d_arr, h_arr, sizeof(int) * numElements, cudaMemcpyHostToDevice);
    CALI_MARK_END("cudaMemcpy");

    cudaMalloc((void **)&temp, sizeof(int) * numElements);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // Print initial array if size is less than or equal to 32
    // if (numElements <= 1024)
    // {
    //     printArray(h_arr, numElements);
    // }

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    // Call mergeSort kernel
    mergeSort<<<1, 1>>>(d_arr, temp, numElements);
    cudaDeviceSynchronize();

    CALI_MARK_END(comp_large);

    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    // Print sorted array if size is less than or equal to 32
    // if (numElements <= 1024)
    // {
    cudaMemcpy(h_arr, d_arr, sizeof(int) * numElements, cudaMemcpyDeviceToHost);
    //     printArray(h_arr, numElements);
    // }

    CALI_MARK_BEGIN(correctness_check);

    // Check if the array is sorted
    bool sorted = isSorted(h_arr, numElements);
    if (sorted)
    {
        printf("Array is sorted.\n");
    }
    else
    {
        printf("Array is not sorted.\n");
    }

    CALI_MARK_END(correctness_check);

    delete[] h_arr;
    cudaFree(d_arr);
    cudaFree(temp);

    // Flush Caliper output
    mgr.stop();
    mgr.flush();

    CALI_MARK_END(mainFunction);

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "Merge Sort");
    adiak::value("ProgrammingModel", "CUDA");
    adiak::value("Datatype", "int");
    adiak::value("SizeOfDatatype", sizeof(int));
    adiak::value("InputSize", numElements);
    adiak::value("InputType", sorting_type_name);
    adiak::value("num_processors", numProcessors);
    adiak::value("group_num", 16);
    adiak::value("implementation_source", "AI & Handwritten & Online");

    return 0;
}