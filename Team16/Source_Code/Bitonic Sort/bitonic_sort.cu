/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include <time.h>

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
const char *correctness_check = "correctness_check";
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *cudamemcpy = "cudaMemcpy";

/* Data generation */
void generateData(int *arr, int length, int inputType)
{
    switch (inputType)
    {
    case 0:
    { // Random
        srand(static_cast<unsigned int>(time(NULL)));
        for (int i = 0; i < length; ++i)
        {
            arr[i] = rand() % length;
        }
        break;
    }
    case 1:
    { // Sorted
        for (int i = 0; i < length; ++i)
        {
            arr[i] = i;
        }
        break;
    }
    case 2:
    { // Reverse sorted
        for (int i = 0; i < length; ++i)
        {
            arr[i] = length - i - 1;
        }
        break;
    }
    case 3:
    { // 1% Perturbed Reverse sorted
        for (int i = 0; i < length; ++i)
        {
            arr[i] = length - i - 1;
        }

        // Perturb 1% of the elements
        for (int i = 0; i < length; i += 100)
        {
            if (i < length - 1)
            {
                // Swap adjacent elements to introduce perturbation
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
            }
        }
        break;
    }
    }
}

__global__ void bitonic_sort_step(int *dev_values, int j, int k)
{
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i ^ j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj) > i)
    {
        if ((i & k) == 0)
        {
            /* Sort ascending */
            if (dev_values[i] > dev_values[ixj])
            {
                /* exchange(i,ixj); */
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        if ((i & k) != 0)
        {
            /* Sort descending */
            if (dev_values[i] < dev_values[ixj])
            {
                /* exchange(i,ixj); */
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(int *values)
{
    int *dev_values;
    size_t size = NUM_VALS * sizeof(int);

    cudaMalloc((void **)&dev_values, size);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(cudamemcpy);
    // MEM COPY FROM HOST TO DEVICE
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
    CALI_MARK_END(cudamemcpy);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    dim3 blocks(BLOCKS, 1);   /* Number of blocks   */
    dim3 threads(THREADS, 1); /* Number of threads  */

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    int j, k;
    /* Major step */
    for (k = 2; k <= NUM_VALS; k <<= 1)
    {
        /* Minor step */
        for (j = k >> 1; j > 0; j = j >> 1)
        {
            bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
        }
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    cudaDeviceSynchronize();

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(cudamemcpy);
    // MEM COPY FROM DEVICE TO HOST
    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END(cudamemcpy);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    cudaFree(dev_values);
}

/* Verification */
bool isSorted(const int *arr, int length)
{
    for (int i = 1; i < length; ++i)
    {
        if (arr[i - 1] > arr[i])
        {
            std::cout << arr[i - 1] << ' ' << arr[i] << ' ';
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[])
{

    int sortingType;
    sortingType = atoi(argv[1]);
    THREADS = atoi(argv[2]);
    NUM_VALS = atoi(argv[3]);

    BLOCKS = NUM_VALS / THREADS;
    BLOCKS = (BLOCKS == 0) ? 1 : BLOCKS;

    printf("Input sorting type: %d\n", sortingType);
    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    CALI_MARK_BEGIN(mainFunction);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN(data_init);
    int *values = (int *)malloc(NUM_VALS * sizeof(int));
    generateData(values, NUM_VALS, sortingType);
    cudaDeviceSynchronize();
    CALI_MARK_END(data_init);

    bitonic_sort(values);
    cudaDeviceSynchronize();

    CALI_MARK_BEGIN(correctness_check);
    bool correct = isSorted(values, NUM_VALS);
    if (correct)
    {
        printf("Array is Sorted!\n");
    }
    else
    {
        printf("Array is NOT sorted\n");
    }
    CALI_MARK_END(correctness_check);

    // Free GPU memory
    cudaFree(values);

    CALI_MARK_END(mainFunction);

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
    case 3:
    {
        inputType = "1% Perturbed";
        break;
    }
    }

    adiak::init(NULL);
    adiak::launchdate();                                                // launch date of the job
    adiak::libraries();                                                 // Libraries used
    adiak::cmdline();                                                   // Command line used to launch the job
    adiak::clustername();                                               // Name of the cluster
    adiak::value("Algorithm", "Bitonic Sort");                          // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA");                           // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                                    // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));                        // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS);                                // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType);                               // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", THREADS);                               // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS);                                 // The number of CUDA blocks
    adiak::value("group_num", 16);                                      // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten, AI, & Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
}