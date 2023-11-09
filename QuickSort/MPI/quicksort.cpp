#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <vector>
#include <string>
#include <iostream>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

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


void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

int partition(int* arr, int low, int high) {
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

void quicksort(int* arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        if (pi - low < high - pi) {
            quicksort(arr, low, pi - 1);
            quicksort(arr, pi + 1, high);
        } else {
            quicksort(arr, pi + 1, high);
            quicksort(arr, low, pi - 1);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    CALI_MARK_BEGIN(mainFunction);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size < 2) {
        fprintf(stderr, "This program requires at least 2 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (world_rank == 0) {
        int n = 10;
        int h_arr[n] = {4, 1, 7, 3, 9, 8, 2, 5, 6, 0};

        int* sub_arr = (int*)malloc(n * sizeof(int));
        MPI_Scatter(h_arr, n / world_size, MPI_INT, sub_arr, n / world_size, MPI_INT, 0, MPI_COMM_WORLD);

        int local_low = 0;
        int local_high = n / world_size - 1;
        quicksort(sub_arr, local_low, local_high);

        MPI_Gather(sub_arr, n / world_size, MPI_INT, h_arr, n / world_size, MPI_INT, 0, MPI_COMM_WORLD);

        free(sub_arr);

        printf("Sorted array: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", h_arr[i]);
        }
        printf("\n");
    } else {
        int n;
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int* sub_arr = (int*)malloc(n / world_size * sizeof(int));
        MPI_Scatter(NULL, 0, MPI_INT, sub_arr, n / world_size, MPI_INT, 0, MPI_COMM_WORLD);

        int local_low = 0;
        int local_high = n / world_size - 1;
        quicksort(sub_arr, local_low, local_high);

        MPI_Gather(sub_arr, n / world_size, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);

        free(sub_arr);
    }

    CALI_MARK_END(mainFunction);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "quicksort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numProcesses); // The number of processors (MPI ranks)
    adiak::value("group_num", 16); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten & Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
   mgr.stop();
   mgr.flush();

    MPI_Finalize();
    return 0;
}