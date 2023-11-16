#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <iostream>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;

const char *main_function = "main_function";
const char *data_init = "data_init";
const char *comp = "comp";
const char *comp_small = "comp_small";
const char *comp_large = "comp_large";
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comm_small = "comm_small";
const char *correctness_check = "correctness_check";

void create_sorted_arr(float *arr, int length)
{
    for (int i = 0; i < length; ++i)
    {
        arr[i] = i;
    }
}

void create_reverse_arr(float *arr, int length)
{
    for (int i = 0; i < length; ++i)
    {
        arr[i] = length - i - 1;
    }
}

void create_random_arr(float *arr, int length)
{
    srand(static_cast<unsigned int>(time(NULL)));
    for (int i = 0; i < length; ++i)
    {
        arr[i] = rand() % length;
    }
}

bool isSorted(const float *arr, int length)
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

void mix(float *arr, int length, float perturbation_factor)
{
    srand(static_cast<unsigned int>(time(NULL)));
    for (int i = 0; i < length; ++i)
    {
        arr[i] += (rand() % length * 2 - 1) * perturbation_factor;
    }
}

void print_array(const float *arr, int length)
{
    for (int i = 0; i < length; ++i)
    {
        std::cout << arr[i] << ' ';
    }
    std::cout << std::endl;
}

__global__ void merge_sort_step(float *dev_values, float *temp, int n, unsigned int width)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int start = 2 * width * idx;

    if (start < n)
    {
        unsigned int middle = min(start + width, n);
        unsigned int end = min(start + 2 * width, n);
        unsigned int i = start;
        unsigned int j = middle;
        unsigned int k = start;

        while (i < middle && j < end)
        {
            if (dev_values[i] < dev_values[j])
            {
                temp[k++] = dev_values[i++];
            }
            else
            {
                temp[k++] = dev_values[j++];
            }
        }
        while (i < middle)
            temp[k++] = dev_values[i++];
        while (j < end)
            temp[k++] = dev_values[j++];

        for (i = start; i < end; i++)
        {
            dev_values[i] = temp[i];
        }
    }
}

void merge_sort(float *initial_arr, int length)
{
    float *dev_values, *temp;
    size_t bytes = length * sizeof(float);
    cudaMalloc((void **)&dev_values, bytes);
    cudaMalloc((void **)&temp, bytes);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(dev_values, initial_arr, bytes, cudaMemcpyHostToDevice);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    dim3 threadsPerBlock(THREADS, 1);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    for (int width = 1; width < length; width *= 2)
    {

        long long totalThreads = (long long)length / (2 * width);
        int numBlocks = (totalThreads + threadsPerBlock.x - 1) / threadsPerBlock.x;

        merge_sort_step<<<numBlocks, threadsPerBlock>>>(dev_values, temp, length, width);
        cudaDeviceSynchronize();

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
            exit(1);
        }
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN("cudaMemcpy");
    cudaMemcpy(initial_arr, dev_values, bytes, cudaMemcpyDeviceToHost);
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    cudaFree(dev_values);
    cudaFree(temp);
}

int main(int argc, char *argv[])
{
    CALI_MARK_BEGIN(main_function);
    cali::ConfigManager mgr;
    mgr.start();

    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s <threads> <num_elements> <sorting_type (0-3)>\n", argv[0]);
        exit(1);
    }

    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    int sorting_type = atoi(argv[3]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Number of threads per block: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    float *initial_arr = (float *)malloc(NUM_VALS * sizeof(float));

    const char *input_type;
    // Initialize data
    CALI_MARK_BEGIN(data_init);

    switch (sorting_type)
    {
    case 0:
        create_random_arr(initial_arr, NUM_VALS);
        input_type = "random_array";
        break;
    case 1: // Sorted
        create_sorted_arr(initial_arr, NUM_VALS);
        input_type = "sorted_array";
        break;
    case 2: // Reverse Sorted
        create_reverse_arr(initial_arr, NUM_VALS);
        input_type = "reversed_array";
        break;
    case 3: // almost sorted  (perturbed)
        create_sorted_arr(initial_arr, NUM_VALS);
        mix(initial_arr, NUM_VALS, 0.01);
        input_type = "perturbed_array";
        break;
    }

    CALI_MARK_END(data_init);

    merge_sort(initial_arr, NUM_VALS);

    CALI_MARK_BEGIN(correctness_check);
    bool correct = isSorted(initial_arr, NUM_VALS);
    if (correct)
    {
        printf("Array is Sorted!\n");
    }
    else
    {
        printf("Array is NOT sorted\n");
    }

    CALI_MARK_END(correctness_check);

    CALI_MARK_END(main_function);

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "MergeSort");               // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA");             // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                      // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));          // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS);                  // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type);                // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", THREADS);                 // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS);                   // The number of CUDA blocks
    adiak::value("group_num", 16);                        // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online & AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten")

    mgr.stop();
    mgr.flush();

    free(initial_arr);

    return 0;
}
