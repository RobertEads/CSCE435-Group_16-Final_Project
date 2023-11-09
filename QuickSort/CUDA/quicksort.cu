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
const char* mainFunction = "main";
const char* data_init = "data_init";
const char* correctness_check = "correctness_check ";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm _small";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comp_small = "comp_small";


__device__ void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

__device__ int partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

__global__ void quicksort(int* arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        if (pi - low < high - pi) {
            quicksort<<<1, 1>>>(arr, low, pi - 1);
            quicksort<<<1, 1>>>(arr, pi + 1, high);
        } else {
            quicksort<<<1, 1>>>(arr, pi + 1, high);
            quicksort<<<1, 1>>>(arr, low, pi - 1);
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
    // Launch the Quicksort kernel
    quicksortLauncher<<<BLOCKS, THREADS>>>(d_unsortedArray, 0, NUM_VALS - 1);
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
    }

    adiak::init(NULL);
    adiak::launchdate();                                                // launch date of the job
    adiak::libraries();                                                 // Libraries used
    adiak::cmdline();                                                   // Command line used to launch the job
    adiak::clustername();                                               // Name of the cluster
    adiak::value("Algorithm", "Quicksort");                             // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
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
