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
const char *comm_small = "comm_small";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comp_small = "comp_small";

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

void selectionSort(int *data, int size)
{
    if (size > 0)
    {
        for (int i = 0; i < size - 1; i++)
        {
            // Assume the current index is the minimum
            int minIndex = i;

            // Find the minimum element in the unsorted part of the array
            for (int j = i + 1; j < size; j++)
            {
                if (data[j] < data[minIndex])
                {
                    minIndex = j;
                }
            }

            // Swap the found minimum element with the first element
            std::swap(data[i], data[minIndex]);
        }
    }
}

/* Parallel Selection Sort (with help from http://www.macs.hw.ac.uk/~hwloidl/Courses/F21DP/srcs/bsort.c)*/
void parallelSelectionSort(vector<int> &localData, vector<int> &sortedData, int my_rank, int numProcesses)
{
    int localN = localData.size();
    int *globalData = new int[localN * numProcesses];

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Gather(localData.data(), localN, MPI_INT, globalData, localN, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);

    std::vector<int> tempVector(globalData, globalData + localN * numProcesses);

    if (my_rank == 0)
    {
        selectionSort(tempVector.data(), localN * numProcesses);

        MPI_Scatter(tempVector.data(), localN, MPI_INT, localData.data(), localN, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Only insert data if my_rank == 0.
    if (my_rank == 0)
    {
        sortedData.insert(sortedData.end(), &tempVector[0], &tempVector[localN]);
        delete[] globalData;
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
}

/* Verify */
bool verifyCorrect(vector<int> &sortedData, int my_rank)
{
    if (sortedData.size() > 0)
    {
        // Verify local data is in order
        for (int i = 1; i < sortedData.size() - 1; i++)
        {
            if (sortedData.at(i - 1) > sortedData.at(i))
            {
                printf("Sorting error on process with rank: %d\n", my_rank);
                return false;
            }
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

    // Sort
    vector<int> sortedData;
    parallelSelectionSort(myLocalData, sortedData, my_rank, numProcesses);

    // Verification
    CALI_MARK_BEGIN(correctness_check);
    bool correct = verifyCorrect(sortedData, my_rank);
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
    adiak::value("Algorithm", "Selection Sort");    // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");     // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int");             // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize);        // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType);        // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numProcesses);     // The number of processors (MPI ranks)
    adiak::value("group_num", "16");             // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten, AI, & Online");
    mgr.stop();
    mgr.flush();
    MPI_Finalize();
}