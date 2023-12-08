#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <string>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>

using namespace std;

int THREADS;
int BLOCKS;
int NUM_VALS;

/* Define Caliper region names */
const char* mainFunction = "main";
const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm_small";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comp_small = "comp_small";

/* Data generation */
__global__ void generateData(int* dataArray, int numValues, int inputType, int numThreads) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(idx < numThreads) {
        int valuesPerThread = numValues/numThreads;
        int start = idx*valuesPerThread;
        switch (inputType) {
            case 0: { //Random    
                curandState state;
                curand_init(44738791+idx, idx, 0, &state);

                for(int i = start; i < (start+valuesPerThread) ; i++) {dataArray[i] = curand(&state) % numValues;}
                break;
            }
            case 1: { //Sorted
                for(int i = start; i < (start+valuesPerThread) ; i++) {dataArray[i] = i;}
                break;
            }
            case 2: { //Reverse sorted
                for(int i = start; i < (start+valuesPerThread) ; i++) {dataArray[i] = numValues - 1 - i;}
                break;
            }
            case 3: { //1% - but just the sorted part
                for(int i = start; i < (start+valuesPerThread) ; i++) {dataArray[i] = i;}
                break;
            }
        }
    }   
}

//CUDA kernal function to make the data 1% perturbed
__global__ void perturbData(int* dataArray, int numValues) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    curandState state;
    curand_init(82408367+idx, idx, 0, &state);

    int num_elements_to_randomize = numValues / 100; // 1% of the array size
    for (int i = 0; i < num_elements_to_randomize; ++i)
    {
        int index = curand(&state) % numValues;  // Generate a random index
        dataArray[index] = curand(&state) % numValues; // Randomize the value at the index
    }
}


/* Main Alg Stuff */
// CUDA kernel to select and gather samples from the array
__global__ void selectSamples(int* dataArray, int numValues, int numberOfThreads, int* samples, int numSamplesPerThread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numberOfThreads) {
        curandState state;
        curand_init(18602669+idx, idx, 0, &state);
        
        for(int i = 0; i < numSamplesPerThread; i++) {
            int index = curand(&state) % numValues;
            samples[(idx*numSamplesPerThread)+i] = dataArray[index];
        }
    }
}

__device__ void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

__device__ int partition(int* array, int low, int high) {
    curandState state;
    curand_init(73071834+0, 0, 0, &state);

    int pivotIndex = low + curand(&state) % (high - low + 1);
    int pivot = array[pivotIndex];

    swap(array[pivotIndex], array[high]);

    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (array[j] < pivot) {
            i++;
            swap(array[i], array[j]);
        }
    }

    swap(array[i + 1], array[high]);
    return (i + 1);
}

__device__ void quicksort_recursive(int* array, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(array, low, high);

        quicksort_recursive(array, low, pivotIndex - 1);
        quicksort_recursive(array, pivotIndex + 1, high);
    }
}

// CUDA kernel to sort the samples
__global__ void quicksort(int* array, int size) {
    quicksort_recursive(array, 0, size - 1);
}

//CUDA kernel to select pivots/splitters - SKIP FOR NOW
__global__ void selectedSplitters(int numberOfThreads, int* samples, int* selectedSplitters, int numSamplesPerThreads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 < idx && idx < numberOfThreads) {
        selectedSplitters[idx-1] = samples[idx*numSamplesPerThreads];
        //selectedSplitters[idx] = samples[idx];
    }
}

// CUDA kernel to calculate the data offsets for grouping
__global__ void partitionDataCalculation(int* dataArray, int numValues, int numberOfThreads, int* splitters, int* bucketOffsets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (numberOfThreads-1)) {
        for(int i = 0; i < numValues; i++) {
            if(dataArray[i] < splitters[idx]) {
                bucketOffsets[idx] += 1;
            }
        }   
    }
}

__global__ void updateArrays(int numberOfThreads, int numValues, int* expandedPivots, int* expandedStarts, int* startPosition, int* pivots) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0) {
        for(int i = 0; i < numberOfThreads-1; i++) {expandedPivots[i] = pivots[i];}
        expandedPivots[numberOfThreads-1] = numValues;  //Try abstracting again I think zero is done before the last few are done, so it resets
    }
    if(idx == 1) {
        for(int i = 1; i < numberOfThreads; i++) {expandedStarts[i] = startPosition[i-1];}
        expandedStarts[0] = 0; //Try abstracting again I think zero is done before the last few are done, so it resets
    }
}

// CUDA kernel to partition the data into buckets based on the samples
__global__ void partitionData(int* unsortedData, int* groupedData, int numberOfThreads, int numValues, int* expandedPivots, int* expandedStarts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numberOfThreads) {    
        int previousCutoff = (idx == 0) ? 0 : expandedPivots[idx-1];

        for(int i = 0; i < numValues; i++) {
            if(previousCutoff <= unsortedData[i] && unsortedData[i] < expandedPivots[idx]) {
                groupedData[expandedStarts[idx]] = unsortedData[i];
                atomicAdd(&expandedStarts[idx], 1);
            }
        }
    }
}

// CUDA kernel to sort each bucket using insertion sort
__global__ void sortBuckets(int* groupedData, int numValues, int numberOfThreads, int* bucketOffsets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numberOfThreads) {
        int bucket = idx;
        int start = (bucket == 0) ? 0 : bucketOffsets[bucket - 1];
        int end = (bucket == (numberOfThreads-1)) ? (numValues) : bucketOffsets[bucket];
        for (int i = start + 1; i < end; i++) {
            int key = groupedData[i];
            int j = i - 1;
            while (j >= start && groupedData[j] > key) {
                groupedData[j + 1] = groupedData[j];
                j--;
            }
            groupedData[j + 1] = key;
        }
    }
}


/* Verification */
// CUDA kernel to check if the array is sorted
__global__ void checkArraySorted(int* dataArrays, int numValues, int* isSorted) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numValues - 1) {
        if(!(dataArrays[idx] <= dataArrays[idx + 1])) {
            atomicAdd(isSorted, 1);
        }
    }
}


/* Program main */
int main(int argc, char *argv[]) {
    int sortingType;

    sortingType = atoi(argv[1]);
    THREADS = atoi(argv[2]);
    NUM_VALS = atoi(argv[3]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Input sorting type: %d\n", sortingType);
    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Max Number of blocks: %d\n", BLOCKS);

    int numUsableBlocks = NUM_VALS / (64 * THREADS);
    int numBuckets = numUsableBlocks * THREADS;
    int numTotalThreads = numBuckets;
    printf("Usable Number of blocks: %d\n", numUsableBlocks);
    printf("Number of buckets: %d\n\n", numBuckets);
    fflush(stdout);
    

    CALI_MARK_BEGIN(mainFunction);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    dim3 usableBlocks(numUsableBlocks,1);    /* Usable Number of blocks   */
    dim3 maxBlocks(BLOCKS,1);    /* Max number of blocks   */
    dim3 threads(THREADS,1);  /* Number of threads  */

    CALI_MARK_BEGIN(data_init);
    /* Data generation */
    int* d_unsortedArray;

    // Allocate memory on the GPU and fill
    cudaMalloc((void**)&d_unsortedArray, NUM_VALS * sizeof(int));
    generateData<<<usableBlocks, threads>>>(d_unsortedArray, NUM_VALS, sortingType, numTotalThreads);
    cudaDeviceSynchronize();

    if(sortingType == 3) {perturbData<<<1,1>>>(d_unsortedArray, NUM_VALS);}
    cudaDeviceSynchronize();
    CALI_MARK_END(data_init);
    
    
    /* Main Alg */
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    int *d_samples, *d_selectedSplitters, *d_bucketOffsets, *d_groupedData, *d_expandedPivots, *d_expandedStarts;
   
    int numSamplesPerThreads = 2;
    int numSamples = numSamplesPerThreads * numBuckets;
    
    // Launch the kernel to select and gather samples
    cudaMalloc((void**)&d_samples, numSamples * sizeof(int));
    selectSamples<<<usableBlocks, threads>>>(d_unsortedArray, NUM_VALS, numTotalThreads, d_samples, numSamplesPerThreads);
    cudaDeviceSynchronize();

    // Launch the kernel to sort the samples
    quicksort<<<1, 1>>>(d_samples, numSamples);
    cudaDeviceSynchronize();

    //Launch the kernel to select that pivots/splitters
    cudaMalloc((void**)&d_selectedSplitters, (numBuckets-1) * sizeof(int));
    selectedSplitters<<<usableBlocks, threads>>>(numTotalThreads, d_samples, d_selectedSplitters, numSamplesPerThreads);
    cudaDeviceSynchronize();

    // Launch the kernel to count the data in each bucket
    cudaMalloc((void**)&d_bucketOffsets, numBuckets * sizeof(int));
    partitionDataCalculation<<<usableBlocks, threads>>>(d_unsortedArray, NUM_VALS, numTotalThreads, d_selectedSplitters, d_bucketOffsets);
    cudaDeviceSynchronize();

    //Update pivots as needed
    cudaMalloc((void**)&d_expandedPivots, numBuckets * sizeof(int));
    cudaMalloc((void**)&d_expandedStarts, numBuckets * sizeof(int));
    updateArrays<<<1,2>>>(numTotalThreads, NUM_VALS, d_expandedPivots, d_expandedStarts, d_bucketOffsets, d_selectedSplitters);
    cudaDeviceSynchronize();
    CALI_MARK_END(comp_small);
    
    CALI_MARK_BEGIN(comp_large);
    //Launch the kernel to partition the data into buckets
    cudaMalloc((void**)&d_groupedData, NUM_VALS * sizeof(int));
    partitionData<<<usableBlocks, threads>>>(d_unsortedArray, d_groupedData, numTotalThreads, NUM_VALS, d_expandedPivots, d_expandedStarts);
    cudaDeviceSynchronize();

    // Launch the kernel to sort each bucket using insertion sort
    sortBuckets<<<usableBlocks, threads>>>(d_groupedData, NUM_VALS, numTotalThreads, d_bucketOffsets);
    cudaDeviceSynchronize();
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);


    /* Verify Correctness */ 
    CALI_MARK_BEGIN(correctness_check);
    int isSorted = 0;
    int* d_isSorted;
    cudaMalloc((void**)&d_isSorted, sizeof(int));
    cudaMemcpy(d_isSorted, &isSorted, sizeof(int), cudaMemcpyHostToDevice);
    
    checkArraySorted<<<maxBlocks, threads>>>(d_groupedData, NUM_VALS, d_isSorted);
    cudaDeviceSynchronize();

    int isSortedCheck = -1;
    cudaMemcpy(&isSortedCheck, d_isSorted, sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END(correctness_check);


    /* Clean up */
    // Free GPU memory
    cudaFree(d_unsortedArray);
    cudaFree(d_samples);
    cudaFree(d_selectedSplitters);
    cudaFree(d_bucketOffsets);
    cudaFree(d_groupedData);
    cudaFree(d_expandedPivots);
    cudaFree(d_expandedStarts);
    cudaFree(d_isSorted);
    
    CALI_MARK_END(mainFunction);


    if(isSortedCheck == -1) {printf("copy error - unable to tell");}
    else if(isSortedCheck == 0) {printf("Sorted!!!");}
    else if(isSortedCheck > 0) {printf("Not sorted :(");}
    else {printf("This should never happen");}

    string inputType;
    switch (sortingType) {
        case 0: {
            inputType = "Random";
            break; }
        case 1: {
            inputType = "Sorted";
            break; }
        case 2: {
            inputType = "ReverseSorted";
            break; }
        case 3: {
            inputType = "1%perturbed";
            break; }
    }

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 16); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output
    mgr.stop();
    mgr.flush();
}

