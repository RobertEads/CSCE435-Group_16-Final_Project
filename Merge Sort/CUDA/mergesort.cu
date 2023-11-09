#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

using namespace std;

int THREADS;
int BLOCKS;
int NUM_VALS;

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

__device__ void merge(int *input, int *output, int left, int mid, int right)
{
    CALI_MARK_BEGIN(comp);

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
    CALI_MARK_END(comp);
}

__global__ void mergeSort(int *input, int *output, int left, int right)
{
    CALI_MARK_BEGIN(comp_large);

    if (right - left <= 1)
    {
        return;
    }

    int mid = left + (right - left) / 2;

    mergeSort(input, output, left, mid);
    mergeSort(input, output, mid, right);

    merge(input, output, left, mid, right);
    CALI_MARK_END(comp_large);
}

void launchMergeSort(int *d_unsortedArray, int *d_tempArray, int numVals)
{
    CALI_MARK_BEGIN(comp_small);

    mergeSort<<<1, numVals>>>(d_unsortedArray, d_tempArray, 0, numVals);
    CALI_MARK_END(comp_small);
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

    CALI_MARK_BEGIN(mainFunction);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN(data_init);
    int *d_unsortedArray;
    int *d_tempArray;

    cudaMalloc((void **)&d_unsortedArray, NUM_VALS * sizeof(int));
    cudaMalloc((void **)&d_tempArray, NUM_VALS * sizeof(int));

    generateData<<<BLOCKS, THREADS>>>(d_unsortedArray, NUM_VALS, sortingType);
    cudaDeviceSynchronize();
    CALI_MARK_END(data_init);

    launchMergeSort(d_unsortedArray, d_tempArray, NUM_VALS);

    int sortedArray[NUM_VALS];
    CALI_MARK_BEGIN(comm_small);
    cudaMemcpy(sortedArray, d_unsortedArray, NUM_VALS * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_small);

    bool isSorted[NUM_VALS - 1];
    bool *d_isSorted;
    cudaMalloc((void **)&d_isSorted, (NUM_VALS - 1) * sizeof(bool));
    CALI_MARK_BEGIN(correctness_check);

    checkArraySorted<<<BLOCKS, THREADS>>>(d_unsortedArray, d_isSorted, NUM_VALS);
    cudaDeviceSynchronize();
    CALI_MARK_END(correctness_check);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(isSorted, d_isSorted, (NUM_VALS - 1) * sizeof(bool), cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
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

    CALI_MARK_END(mainFunction);

    if (sorted)
    {
        printf("The array is sorted!\n");
    }
    else
    {
        printf("The array is not sorted!\n");
    }

    string inputType;
    switch (sortingType)
    {
    case 0:
    {
        inputType = "Randomized";
        break;
    }
    case 1:
    {
        inputType = "Sorted";
        break;
    }
    case 2:
    {
        inputType = "Reverse Sorted";
        break;
    }
    }

    adiak::init(NULL);
    adiak::launchdate();                                       // launch date of the job
    adiak::libraries();                                        // Libraries used
    adiak::cmdline();                                          // Command line used to launch the job
    adiak::clustername();                                      // Name of the cluster
    adiak::value("Algorithm", "SampleSort");                   // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA");                  // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                           // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));               // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS);                       // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType);                      // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", THREADS);                      // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS);                        // The number of CUDA blocks
    adiak::value("group_num", 16);                             // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI & Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output
    mgr.stop();
    mgr.flush();
    return 0;
}