#include "mpi.h"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using std::string;
using std::swap;
using std::vector;

int inputSize, numProcesses;

/* Define Caliper region names */
const char *mainFunction = "main";
const char *data_init = "data_init";
const char *correctness_check = "correctness_check";
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *mpi_gather = "mpi_gather";
const char *mpi_scatter = "mpi_scatter";

/* Data generation */
void generateData(vector<int> &localData, int startingSortChoice, int amountToGenerate, int startingPosition, int my_rank)
{
    switch (startingSortChoice)
    {
    case 0:
    { // Random
        srand((my_rank + 5) * (my_rank + 12) * 1235);
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
    case 3:
    { // 1% Perturbed
        srand((my_rank + 5) * (my_rank + 12) * 1235);
        for (int i = 0; i < amountToGenerate; i++)
        {
            // Generate 1% perturbed data
            int perturb = static_cast<int>(0.01 * inputSize * (rand() % 2 ? 1 : -1));
            localData.push_back((i + perturb + inputSize) % inputSize);
        }
        break;
    }
    }
}

/* Bitonic Sort */
void bitonicMerge(vector<int> &arr, int low, int cnt, int dir)
{
    if (cnt > 1)
    {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++)
            if ((arr[i] > arr[i + k] && dir == 1) || (arr[i] < arr[i + k] && dir == 0))
                swap(arr[i], arr[i + k]);

        bitonicMerge(arr, low, k, dir);
        bitonicMerge(arr, low + k, k, dir);
    }
}

void bitonicSort(vector<int> &arr, int low, int cnt, int dir)
{
    if (cnt > 1)
    {
        int k = cnt / 2;
        bitonicSort(arr, low, k, 1);
        bitonicSort(arr, low + k, k, 0);
        bitonicMerge(arr, low, cnt, dir);
    }
}

/* Verification */
bool isSorted(const std::vector<int> &arr)
{
    for (size_t i = 1; i < arr.size(); ++i)
    {
        if (arr[i - 1] > arr[i])
        {
            std::cout << arr[i - 1] << ' ' << arr[i] << ' ';
            return false;
        }
    }
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

    CALI_MARK_BEGIN(mainFunction);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // Generate data
    CALI_MARK_BEGIN(data_init);
    vector<int> myLocalData;
    int amountToGenerateMyself = inputSize / numProcesses;
    int startingPos = my_rank * amountToGenerateMyself;
    generateData(myLocalData, sortingType, amountToGenerateMyself, startingPos, my_rank);
    CALI_MARK_END(data_init);

    // Gather all data at root process
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(mpi_gather);
    vector<int> globalData(inputSize);
    MPI_Gather(myLocalData.data(), amountToGenerateMyself, MPI_INT, globalData.data(), amountToGenerateMyself, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(mpi_gather);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    if (my_rank == 0)
    {
        // Perform bitonic sort on the gathered data
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        bitonicSort(globalData, 0, inputSize, 1);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);
    }

    // Scatter the sorted data back to all processes
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(mpi_scatter);
    MPI_Scatter(globalData.data(), amountToGenerateMyself, MPI_INT, myLocalData.data(), amountToGenerateMyself, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(mpi_scatter);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // Verification
    CALI_MARK_BEGIN(correctness_check);
    bool correct = isSorted(myLocalData);
    CALI_MARK_END(correctness_check);


    CALI_MARK_END(mainFunction);

    if (!correct)
    {
        printf("There is a problem with the sorting. Quitting...\n");
    }
    else
    {
        if (my_rank == 0)
        {
            printf("\nAll data sorted correctly!");
        }
    }

    adiak::init(NULL);
    adiak::launchdate();                         // launch date of the job
    adiak::libraries();                          // Libraries used
    adiak::cmdline();                            // Command line used to launch the job
    adiak::clustername();                        // Name of the cluster
    adiak::value("Algorithm", "Bitonic Sort");   // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");     // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");             // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize);        // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType);        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numProcesses);     // The number of processors (MPI ranks)
    adiak::value("group_num", "16");             // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten, AI, & Online");  // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    mgr.stop();
    mgr.flush();
    MPI_Finalize();

    return 0;
}
