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

__global__ void checkArraySorted(int *array, bool *isSorted, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1)
    {
        isSorted[idx] = (array[idx] <= array[idx + 1]);
    }
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
    cudaMemcpy(h_arr, d_generateResult, sizeof(int) * numElements, cudaMemcpyDeviceToHost);
    cudaFree(d_generateResult);
    CALI_MARK_END(data_init);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMalloc((void **)&d_arr, sizeof(int) * numElements);
    CALI_MARK_BEGIN(comm_small);
    cudaMemcpy(d_arr, h_arr, sizeof(int) * numElements, cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_small);
    cudaMalloc((void **)&temp, sizeof(int) * numElements);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    CALI_MARK_BEGIN(comp_small);
    // Call mergeSort kernel
    mergeSort<<<1, 1>>>(d_arr, temp, numElements);
    cudaDeviceSynchronize();
    CALI_MARK_END(comp_small);

    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(correctness_check);

    // Call checkArraySorted kernel
    bool *d_isSorted;
    bool *h_isSorted = new bool[numElements - 1];
    cudaMalloc((void **)&d_isSorted, sizeof(bool) * (numElements - 1));
    checkArraySorted<<<(numElements + 255) / 256, 256>>>(d_arr, d_isSorted, numElements);
    cudaMemcpy(h_isSorted, d_isSorted, sizeof(bool) * (numElements - 1), cudaMemcpyDeviceToHost);

    CALI_MARK_END(correctness_check);

    // // Print sorted array
    // printf("Sorted Array: ");
    // for (int i = 0; i < numElements; i++)
    //     printf("%d ", h_arr[i]);
    // printf("\n");

    // Check if the array is sorted
    bool sorted = true;
    for (int i = 0; i < numElements - 1; i++)
    {
        if (!h_isSorted[i])
        {
            sorted = false;
            break;
        }
    }

    if (sorted)
    {
        printf("Array is sorted.\n");
    }
    else
    {
        printf("Array is not sorted.\n");
    }

    delete[] h_arr;
    delete[] h_isSorted;
    cudaFree(d_arr);
    cudaFree(temp);
    cudaFree(d_isSorted);

    // Flush Caliper output
    mgr.stop();
    mgr.flush();

    CALI_MARK_END(mainFunction);

    adiak::init(NULL);
    adiak::launchdate();                         // launch date of the job
    adiak::libraries();                          // Libraries used
    adiak::cmdline();                            // Command line used to launch the job
    adiak::clustername();                        // Name of the cluster
    adiak::value("Algorithm", "Merge Sort");     // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA");    // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");             // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", numElements);      // The number of elements in input dataset (1000)
    adiak::value("InputType", sortingType);      // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%pert

    return 0;
}
