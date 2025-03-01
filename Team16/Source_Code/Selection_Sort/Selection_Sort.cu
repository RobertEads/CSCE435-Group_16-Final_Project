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
const char *correctness_check = "correctness_check";
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comm_small = "comm_small";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comp_small = "comp_small";

/* Data generation */
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

// CUDA kernel for Selection Sort
__global__ void selectionSort(int *array, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        for (int i = 0; i < size - 1; i++)
        {
            int minIndex = i;

            // Find the index of the minimum element in the unsorted part
            for (int j = i + 1; j < size; j++)
            {
                if (array[j] < array[minIndex])
                {
                    minIndex = j;
                }
            }

            // Swap the found minimum element with the first element
            int temp = array[i];
            array[i] = array[minIndex];
            array[minIndex] = temp;
        }
    }
}


/* Verification */
// CUDA kernel to check if the array is sorted
__global__ void checkArraySorted(int *array, bool *isSorted, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1)
    {
        isSorted[idx] = (array[idx] <= array[idx + 1]);
    }
}

/* Program main */
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
    /* Data generation */
    int *d_unsortedArray;

    // Allocate memory on the GPU and fill
    cudaMalloc((void **)&d_unsortedArray, NUM_VALS * sizeof(int));
    generateData<<<BLOCKS, THREADS>>>(d_unsortedArray, NUM_VALS, sortingType);
    cudaDeviceSynchronize();
    CALI_MARK_END(data_init);

    /* Main Alg */
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    CALI_MARK_BEGIN(comp_small);
    // Launch the Selection Sort kernel
    selectionSort<<<BLOCKS, THREADS>>>(d_unsortedArray, NUM_VALS);
    cudaDeviceSynchronize();
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(comm_small);
    // Copy data back to the host
    int sortedArray[NUM_VALS];
    cudaMemcpy(sortedArray, d_unsortedArray, NUM_VALS * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(correctness_check);
    /* Verify Correctness */
    bool isSorted[NUM_VALS - 1];
    bool *d_isSorted;
    cudaMalloc((void **)&d_isSorted, (NUM_VALS - 1) * sizeof(bool));
    checkArraySorted<<<BLOCKS, THREADS>>>(d_unsortedArray, d_isSorted, NUM_VALS);
    cudaDeviceSynchronize();

    cudaMemcpy(isSorted, d_isSorted, (NUM_VALS - 1) * sizeof(bool), cudaMemcpyDeviceToHost);

    // Verify if the array is sorted
    bool sorted = true;
    for (int i = 0; i < NUM_VALS - 1; i++)
    {
        if (!isSorted[i])
        {
            sorted = false;
            break;
        }
    }
    CALI_MARK_END(correctness_check);

    // Free GPU memory
    cudaFree(d_unsortedArray);
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
    adiak::value("Algorithm", "Selection Sort");                           // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA");                           // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                                    // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));                        // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS);                                // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType);                               // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", THREADS);                               // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS);                                 // The number of CUDA blocks
    adiak::value("group_num", 16);                                      // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten, AI, & Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output
    mgr.stop();
    mgr.flush();
}