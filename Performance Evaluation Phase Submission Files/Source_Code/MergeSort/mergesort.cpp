#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char *main_function = "main_function";
const char *data_init = "data_init";
const char *comm = "comm";
const char *comm_small = "comm_small";
const char *comm_large = "comm_large";
const char *comp = "comp";
const char *comp_small = "comp_small";
const char *comp_large = "comp_large";
const char *correctness_check = "correctness_check";

int inputSize, numProcesses;

void generate_random_array(int *array, int size)
{
    for (int c = 0; c < size; c++)
    {
        array[c] = rand() % size;
    }
}

void generate_sorted_array(int *array, int size)
{
    for (int c = 0; c < size; c++)
    {
        array[c] = c;
    }
}

void generate_reverse_sorted_array(int *array, int size)
{
    for (int c = 0; c < size; c++)
    {
        array[c] = size - c - 1;
    }
}

void generate_randomized_sorted_array(int *array, int size)
{
    // Generate a sorted array
    for (int c = 0; c < size; c++)
    {
        array[c] = c;
    }

    // Randomize 1% of the data
    int random_percentage = size / 100; // Calculate 1% of the size
    for (int c = 0; c < random_percentage; c++)
    {
        int index = rand() % size;    // Generate a random index
        array[index] = rand() % size; // Randomize the value at the index
    }
}

/********** Merge Function **********/
void merge(int *a, int *b, int l, int m, int r)
{

    int h, i, j, k;
    h = l;
    i = l;
    j = m + 1;

    while ((h <= m) && (j <= r))
    {

        if (a[h] <= a[j])
        {

            b[i] = a[h];
            h++;
        }

        else
        {

            b[i] = a[j];
            j++;
        }

        i++;
    }

    if (m < h)
    {

        for (k = j; k <= r; k++)
        {

            b[i] = a[k];
            i++;
        }
    }

    else
    {

        for (k = h; k <= m; k++)
        {

            b[i] = a[k];
            i++;
        }
    }

    for (k = l; k <= r; k++)
    {

        a[k] = b[k];
    }
}

/********** Recursive Merge Function **********/
void mergeSort(int *a, int *b, int l, int r)
{

    int m;

    if (l < r)
    {

        m = (l + r) / 2;

        mergeSort(a, b, l, m);
        mergeSort(a, b, (m + 1), r);
        merge(a, b, l, m, r);
    }
}

bool isSorted(const int *array, int size)
{
    CALI_MARK_BEGIN(correctness_check);

    for (int i = 0; i < size - 1; i++)
    {
        if (array[i] > array[i + 1])
        {
            CALI_MARK_END(correctness_check);
            return false;
        }
    }

    CALI_MARK_END(correctness_check);
    return true;
}

int main(int argc, char **argv)
{
    int sortingType;
    if (argc == 4)
    {
        sortingType = atoi(argv[1]);
        numProcesses = atoi(argv[2]);
        inputSize = atoi(argv[3]);
    }
    else
    {
        printf("\n Please ensure input is as follows [input sorted status (0-2)] [# processes] [size of input]");
        return 0;
    }

    std::string inputType;
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
    CALI_MARK_BEGIN(main_function);
    cali::ConfigManager mgr;
    mgr.start();

    /* MPI Setup */
    int my_rank, num_ranks, rc;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks < 2)
    {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    if (num_ranks != numProcesses)
    {
        printf("Target number of processes and actual number of ranks do not match. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    if (my_rank == 0)
    {
        printf("Input type: %d\n", sortingType);
        printf("Number Processes: %d\n", numProcesses);
        printf("Input Size: %d\n", inputSize);
    }

    int *original_array = (int *)malloc(inputSize * sizeof(int));

    int c;
    srand(time(NULL));
    // printf("This is the ");

    std::string sorting_type_name;

    CALI_MARK_BEGIN(data_init);
    switch (sortingType)
    {
    case 0:
        generate_random_array(original_array, inputSize);
        sorting_type_name = "Random";
        break;
    case 1:
        // printf("sorted array: ");
        generate_sorted_array(original_array, inputSize);
        sorting_type_name = "Sorted";
        break;
    case 2:
        // printf("reverse sorted array: ");
        generate_reverse_sorted_array(original_array, inputSize);
        sorting_type_name = "ReverseSorted";
        break;
    case 3:
        // printf("partially randomized sorted array: ");
        generate_randomized_sorted_array(original_array, inputSize);
        sorting_type_name = "1%perturbed";
        break;
    default:
        printf("Invalid sorting type. Please use 0, 1, 2, or 3.");
        free(original_array);
        MPI_Finalize();
        return 1;
    }

    CALI_MARK_END(data_init);

    // printf("WORLD SIZE: %d\n", numProcesses);

    double start_time = MPI_Wtime();

    /********** Divide the array in equal-sized chunks **********/
    int size = inputSize / numProcesses;

    /********** Send each subarray to each process **********/
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    int *sub_array = (int *)malloc(size * sizeof(int));
    CALI_MARK_BEGIN("MPI_Scatter");
    MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Scatter");
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    /********** Perform the mergesort on each process **********/
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    int *tmp_array = (int *)malloc(size * sizeof(int));
    mergeSort(sub_array, tmp_array, 0, (size - 1));
    CALI_MARK_END(comp_large);

    CALI_MARK_END(comp);

    /********** Gather the sorted subarrays into one **********/
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    int *sorted = NULL;
    if (my_rank == 0)
    {

        sorted = (int *)malloc(inputSize * sizeof(int));
    }
    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gather");
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    /********** Make the final mergeSort call **********/
    if (my_rank == 0)
    {

        int *other_array = (int *)malloc(inputSize * sizeof(int));
        mergeSort(sorted, other_array, 0, (inputSize - 1));

        /********** Display the sorted array **********/
        // printf("This is the sorted array: ");
        // for (c = 0; c < n; c++)
        // {

        //     printf("%d ", sorted[c]);
        // }

        // printf("\n");
        // printf("\n");

        double end_time = MPI_Wtime();
        if (isSorted(sorted, inputSize))
        {
            printf("Array is sorted!\n");
        }
        else
        {

            printf("Array is not sorted!\n");
        }

        std::cout << "Time taken: " << end_time - start_time << " seconds." << std::endl;

        /********** Clean up root **********/
        free(sorted);
        free(other_array);
    }

    /********** Clean up rest **********/
    free(original_array);
    free(sub_array);
    free(tmp_array);

    CALI_MARK_END(main_function);

    adiak::init(NULL);
    adiak::launchdate();                                              // launch date of the job
    adiak::libraries();                                               // Libraries used
    adiak::cmdline();                                                 // Command line used to launch the job
    adiak::clustername();                                             // Name of the cluster
    adiak::value("Algorithm", "Merge Sort");                          // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");                          // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                                  // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));                      // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize);                             // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType);                             // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numProcesses);                          // The number of processors (MPI ranks)
    adiak::value("group_num", 16);                                    // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online, AI, Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    /********** Finalize MPI **********/
    MPI_Barrier(MPI_COMM_WORLD);

    mgr.stop();
    mgr.flush();
    MPI_Finalize();
}
