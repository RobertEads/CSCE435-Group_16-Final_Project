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

const char *mainFunction = "main";
const char *data_init = "data_init";
const char *correctness_check = "correctness_check";
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comm_small = "comm_small";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comp_small = "comp_small";

void generateData(vector<int> &localData, int startingSortChoice, int amountToGenerate, int startingPosition, int my_rank)
{
    CALI_MARK_BEGIN(data_init);
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
    }
    CALI_MARK_END(data_init);
}

void merge(int arr[], int left, int mid, int right)
{
    CALI_MARK_BEGIN(comp);

    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *leftArray = new int[n1];
    int *rightArray = new int[n2];

    for (int i = 0; i < n1; i++)
        leftArray[i] = arr[left + i];
    for (int i = 0; i < n2; i++)
        rightArray[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
    {
        if (leftArray[i] <= rightArray[j])
        {
            arr[k] = leftArray[i];
            i++;
        }
        else
        {
            arr[k] = rightArray[j];
            j++;
        }
        k++;
    }

    while (i < n1)
    {
        arr[k] = leftArray[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        arr[k] = rightArray[j];
        j++;
        k++;
    }

    delete[] leftArray;
    delete[] rightArray;
    CALI_MARK_END(comp);
}

void mergeSort(int arr[], int left, int right)
{
    if (left < right)
    {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

void parallelMerge(vector<int> &localData, vector<int> &partnerData, vector<int> &mergedData, int my_rank, int partner_rank, int localSize, int partnerSize)
{
    int *localArray = localData.data();
    int *partnerArray = partnerData.data();

    int left = 0, mid = localSize - 1, right = localSize + partnerSize - 1;
    int i = 0, j = 0, k = 0;

    CALI_MARK_BEGIN(comp_small);

    while (i <= mid && j <= partnerSize)
    {
        if (localArray[i] <= partnerArray[j])
        {
            mergedData[k++] = localArray[i++];
        }
        else
        {
            mergedData[k++] = partnerArray[j++];
        }
    }

    while (i <= mid)
    {
        mergedData[k++] = localArray[i++];
    }

    while (j < partnerSize)
    {
        mergedData[k++] = partnerArray[j++];
    }

    swap(localData, mergedData);

    CALI_MARK_END(comp_small);
}

void parallelMergeSort(vector<int> &localData, int my_rank)
{
    CALI_MARK_BEGIN(comp_large);

    int *localArray = localData.data();
    int localSize = localData.size();

    mergeSort(localArray, 0, localSize - 1);

    for (int step = 1; step < numProcesses; step <<= 1)
    {
        int partner_rank = my_rank ^ step;

        if (partner_rank < numProcesses)
        {
            vector<int> partnerData(localArray, localArray + localSize);
            vector<int> mergedData(localSize + localSize);

            CALI_MARK_BEGIN(comm_small);

            MPI_Sendrecv(localArray, localSize, MPI_INT, partner_rank, 0, partnerData.data(), localSize, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            CALI_MARK_END(comm_small);

            parallelMerge(localData, partnerData, mergedData, my_rank, partner_rank, localSize, localSize);
        }
    }
    CALI_MARK_END(comp_large);
}

bool verifyCorrect(vector<int> &sortedData, int my_rank)
{
    CALI_MARK_BEGIN(correctness_check);

    for (int i = 1; i < sortedData.size(); i++)
    {
        if (sortedData[i - 1] > sortedData[i])
        {
            printf("Sorting error on process with rank: %d\n", my_rank);
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
    }

    int my_rank, num_ranks, rc;

    CALI_MARK_BEGIN(comm);

    /* MPI Setup */
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

    CALI_MARK_END(comm);

    if (my_rank == 0)
    {
        printf("Input type: %d\n", sortingType);
        printf("Number Processes: %d\n", numProcesses);
        printf("Input Size: %d\n", inputSize);
    }

    // Data generation
    vector<int> myLocalData;
    int amountToGenerateMyself = inputSize / numProcesses;
    int startingPos = my_rank * (amountToGenerateMyself);
    generateData(myLocalData, sortingType, amountToGenerateMyself, startingPos, my_rank);

    // Print original array
    printf("Original Array (Rank %d):\n", my_rank);
    for (int i = 0; i < myLocalData.size(); i++)
    {
        printf("%d ", myLocalData[i]);
    }
    printf("\n");

    // Main Alg
    CALI_MARK_BEGIN(comm_large);
    parallelMergeSort(myLocalData, my_rank);
    CALI_MARK_END(comm_large);

    // Print sorted array
    printf("Sorted Array (Rank %d):\n", my_rank);
    for (int i = 0; i < myLocalData.size(); i++)
    {
        printf("%d ", myLocalData[i]);
    }
    printf("\n");
    CALI_MARK_END(mainFunction);
    // Verification
    bool correct = verifyCorrect(myLocalData, my_rank);

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
    adiak::value("implementation_source", "Handwritten, Online, AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();

    return 0;
}
