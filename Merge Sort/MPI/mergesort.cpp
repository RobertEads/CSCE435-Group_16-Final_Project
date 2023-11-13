#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <mpi.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char *mainFunction = "main";
const char *data_init = "data_init";
const char *correctness_check = "correctness_check";
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comm_small = "comm_small";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comp_small = "comp_small";

void merge(std::vector<int> &arr, int l, int m, int r)
{
    CALI_MARK_BEGIN(comp_small);
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temporary vectors
    std::vector<int> L(arr.begin() + l, arr.begin() + l + n1);
    std::vector<int> R(arr.begin() + m + 1, arr.begin() + m + 1 + n2);

    // Merge the temporary vectors back into arr[l..r]
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
    CALI_MARK_END(comp_small);
}

void mergeSort(std::vector<int> &arr, int l, int r)
{
    CALI_MARK_BEGIN(comp_large);
    if (l < r)
    {
        // Same as (l+r)/2, but avoids overflow for large l and r
        int m = l + (r - l) / 2;

        // Sort first and second halves
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        // Merge the sorted halves
        merge(arr, l, m, r);
    }
    CALI_MARK_END(comp_large);
}

void generateData(std::vector<int> &localData, int startingSortChoice, int amountToGenerate, int startingPosition, int my_rank, int inputSize)
{
    CALI_MARK_BEGIN(data_init);
    srand((my_rank + 5) * (my_rank + 12) * 1235);

    switch (startingSortChoice)
    {
    case 0:
    { // Random
        for (int i = 0; i < amountToGenerate; i++)
        {
            localData.push_back(rand() % inputSize);
        }
        break;
    }
    case 1:
    { // Sorted
        int endValue = startingPosition + amountToGenerate;
        for (int i = startingPosition; i < endValue; i++)
        {
            localData.push_back(i);
        }
        break;
    }
    case 2:
    { // Reverse sorted
        int startValue = inputSize - 1 - startingPosition;
        int endValue = inputSize - amountToGenerate - startingPosition;
        for (int i = startValue; i >= endValue; i--)
        {
            localData.push_back(i);
        }
        break;
    }
    }
    CALI_MARK_END(data_init);
}

bool verifyCorrect(std::vector<int> &sortedData, int my_rank)
{
    CALI_MARK_BEGIN(correctness_check);
    for (int i = 1; i < sortedData.size(); i++)
    {
        if (sortedData[i - 1] > sortedData[i])
        {
            printf("Sorting error on process with rank: %d\n", my_rank);
            CALI_MARK_END(correctness_check);
            return false;
        }
    }

    CALI_MARK_END(correctness_check);
    return true;
}

int main(int argc, char *argv[])
{
    CALI_MARK_BEGIN(mainFunction);
    cali::ConfigManager mgr;
    mgr.start();
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <sorting_type> <num_processors> <num_elements>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int sorting_type = std::atoi(argv[1]);
    int num_processors = std::atoi(argv[2]);
    int num_elements = std::atoi(argv[3]);

    // MPI Initialization
    MPI_Init(&argc, &argv);

    // Get the rank and size of the communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != num_processors)
    {
        if (rank == 0)
        {
            std::cerr << "Number of processors specified does not match the actual number of processors." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::vector<int> arr;

    // Generate data based on sorting type
    if (sorting_type == 0)
    {
        // Random data
        generateData(arr, 0, num_elements, 0, rank, num_elements);
    }
    else if (sorting_type == 1)
    {
        // Sorted data
        generateData(arr, 1, num_elements, rank * (num_elements / size), rank, num_elements);
    }
    else if (sorting_type == 2)
    {
        // Reverse sorted data
        generateData(arr, 2, num_elements, rank * (num_elements / size), rank, num_elements);
    }
    else
    {
        std::cerr << "Invalid sorting type. Please use 0 for random data, 1 for sorted data, or 2 for reverse sorted data." << std::endl;
        MPI_Finalize();
        return 1;
    }

    // Output initial array
    if (rank == 0)
    {
        printf("Process %d: Initial Array: ", rank);
        for (int i = 0; i < num_elements; ++i)
        {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }

    // Start time measurement
    double start_time = MPI_Wtime();

    // Perform local merge sort
    int local_size = num_elements / size;
    std::vector<int> local_arr(arr.begin() + rank * local_size, arr.begin() + (rank + 1) * local_size);

    CALI_MARK_BEGIN(comp);
    mergeSort(local_arr, 0, local_size - 1);
    CALI_MARK_END(comp);

    // Gather sorted local arrays to the root process);
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(comm_small);
    MPI_Gather(&local_arr[0], local_size, MPI_INT, &arr[0], local_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // Perform final merge on the root process
    if (rank == 0)
    {
        mergeSort(arr, 0, num_elements - 1);
    }

    // Stop time measurement
    double end_time = MPI_Wtime();

    // Output results or time taken
    if (rank == 0)
    {

        // Verify correctness
        if (verifyCorrect(arr, rank))
        {
            std::cout << "Sorting is correct." << std::endl;
        }
        else
        {
            std::cout << "Sorting is incorrect." << std::endl;
        }

        printf("Process %d: Final Array: ", rank);
        for (int i = 0; i < num_elements; ++i)
        {
            printf("%d ", arr[i]);
        }
        printf("\n");

        // Output time taken
        std::cout << "Time taken: " << end_time - start_time << " seconds." << std::endl;
    }

    CALI_MARK_END(mainFunction);

    adiak::init(NULL);
    adiak::launchdate();                                              // launch date of the job
    adiak::libraries();                                               // Libraries used
    adiak::cmdline();                                                 // Command line used to launch the job
    adiak::clustername();                                             // Name of the cluster
    adiak::value("Algorithm", "Merge Sort");                          // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");                          // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");                                  // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int));                      // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", num_elements);                          // The number of elements in input dataset (1000)
    adiak::value("InputType", sorting_type);                          // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_processors);                        // The number of processors (MPI ranks)
    adiak::value("group_num", 16);                                    // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten, Online, AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();

    return 0;
}