#include "mpi.h"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using std::vector;
using std::string;
using std::swap;

int inputSize, numProcesses;

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
void generateData(vector<int> &localData, int startingSortChoice, int amountToGenerate, int startingPosition, int my_rank) {
    switch (startingSortChoice) {
        case 0: { //Random
            srand((my_rank+5)*(my_rank+12)*1235);
            for(int i = 0; i < amountToGenerate; i++) {
                localData.push_back(rand() % inputSize);
            }
            break;
        }

        case 1: {//Sorted
            int endValue = startingPosition + amountToGenerate;
            for(int i = startingPosition; i < endValue; i++) {
                localData.push_back(i);
            }
            break;
        }
        case 2: { //Reverse sorted
            int startValue = inputSize - 1 - startingPosition;
            int endValue = inputSize - amountToGenerate - startingPosition;
            for(int i = startValue; i >= endValue; i--) {
                localData.push_back(i);
            }
            break;
        }
    }
}

/* Sequential Quick Sort & Helpers 
*  quickSort and partition function from geeksforgeeks.org
*/
int partition(int arr[], int start, int end)
{
    int pivot = arr[start];
 
    int count = 0;
    for (int i = start + 1; i <= end; i++) {
        if (arr[i] <= pivot)
            count++;
    }
 
    // Giving pivot element its correct position
    int pivotIndex = start + count;
    swap(arr[pivotIndex], arr[start]);
 
    // Sorting left and right parts of the pivot element
    int i = start, j = end;
 
    while (i < pivotIndex && j > pivotIndex) {
 
        while (arr[i] <= pivot) {
            i++;
        }
 
        while (arr[j] > pivot) {
            j--;
        }
 
        if (i < pivotIndex && j > pivotIndex) {
            swap(arr[i++], arr[j--]);
        }
    }
 
    return pivotIndex;
}

void quickSort(int arr[], int start, int end)
{
    // base case
    if (start >= end)
        return;
 
    // partitioning the array
    int p = partition(arr, start, end);
 
    // Sorting the left part
    quickSort(arr, start, p - 1);
 
    // Sorting the right part
    quickSort(arr, p + 1, end);
}

/* Main Alg */
void sampleSort(vector<int> &localData, vector<int> &sortedData, int my_rank) {
    /* Sample splitters */
    int numSplitters = 4; //# Sampled per node
    vector<int> sampledSplitters;
    srand(84723840);
    
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    for(int i = 0; i < numSplitters; i++) {
        sampledSplitters.push_back(localData.at(rand() % localData.size()));
    }
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);


    /* Combine splitters */
    int totalSplitterArraySize = numSplitters * numProcesses;
    int allSplitters[totalSplitterArraySize];

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    CALI_MARK_BEGIN("MPI_Allgather");
    MPI_Allgather(&sampledSplitters[0], numSplitters, MPI_INT, &allSplitters[0], numSplitters, MPI_INT, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Allgather");
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);


    /* Sort splitters & Decide cuts */
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    quickSort(allSplitters, 0, totalSplitterArraySize-1); //In place sort

    vector<int> choosenSplitters;
    for(int i = 1; i < numProcesses; i++) {
        choosenSplitters.push_back(allSplitters[i*numSplitters]);
    }
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    /* Eval local elements and place into buffers */
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    vector<vector<int>> sendBuckets;
    for(int i = 0; i < numProcesses; i++){sendBuckets.push_back(vector<int>());}

    for(int i = 0; i < localData.size(); i++) {
        int notUsed = 1;
        for(int j = 0; j < choosenSplitters.size(); j++) {
            if(localData.at(i) < choosenSplitters.at(j)) {
                sendBuckets.at(j).push_back(localData.at(i));
                notUsed = 0;
                break;
            }
        }
        if(notUsed){sendBuckets.at(sendBuckets.size()-1).push_back(localData.at(i));}
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    /* Send/Receive Data */ 
    //Gather sizes
    int localBucketSizes[numProcesses];
    for(int i = 0; i < numProcesses; i++) {localBucketSizes[i] = sendBuckets.at(i).size();}

    //Communicate sizes
    int targetSizes[numProcesses];
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    CALI_MARK_BEGIN("MPI_Gather");
    for(int i = 0; i < numProcesses; i++) {
        MPI_Gather(&localBucketSizes[i], 1, MPI_INT, &targetSizes[0], 1, MPI_INT, i, MPI_COMM_WORLD);
    }
    CALI_MARK_END("MPI_Gather");
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    //Sum and calculate displacements
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    int myTotalSize = 0;
    for(int i = 0; i < numProcesses; i++) {myTotalSize += targetSizes[i];}

    int displacements[numProcesses];
    displacements[0] = 0;
    for(int i = 0; i < (numProcesses-1); i++) {displacements[i+1] = displacements[i] + targetSizes[i];}
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);
    
    //Allocate array
    int unsortedData[myTotalSize];

    //Gather data
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN("MPI_Gatherv");
    for(int i = 0; i < numProcesses; i++) {
        MPI_Gatherv(&sendBuckets[i][0], sendBuckets.at(i).size(), MPI_INT, &unsortedData, targetSizes, displacements, MPI_INT, i, MPI_COMM_WORLD);
    }
    CALI_MARK_END("MPI_Gatherv");
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    /* Sort */
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    quickSort(unsortedData, 0, myTotalSize-1);
    sortedData.insert(sortedData.end(), &unsortedData[0], &unsortedData[myTotalSize]);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    
}

/* Verify */
bool verifyCorrect(vector<int> &sortedData, int my_rank) {
    //Verify local data is in order
    for(int i = 1; i < sortedData.size()-1; i++) {
        if(sortedData.at(i-1) > sortedData.at(i)) {printf("Sorting error on process with rank: %d\n", my_rank); return false;}
    }

    //Verify my start and end line up
    int myDataBounds[] = {sortedData.at(0), sortedData.at(sortedData.size()-1)};
    int boundsArraySize = 2*numProcesses;
    int allDataBounds[boundsArraySize];
    MPI_Allgather(&myDataBounds, 2, MPI_INT, &allDataBounds, 2, MPI_INT, MPI_COMM_WORLD);

    for(int i = 1; i < boundsArraySize-1; i++) {
        if(allDataBounds[i-1] > allDataBounds[i]) {printf("Sorting error on bounds regions: %d\n", my_rank); return false;}
    }

    return true;
}


/* Program Main */
int main (int argc, char *argv[])
{
    int sortingType;
    if (argc == 4) {
        sortingType = atoi(argv[1]);
        numProcesses = atoi(argv[2]);
        inputSize = atoi(argv[3]);
    }
    else {
        printf("\n Please ensure input is as follows [input sorted status (0-2)] [# processes] [size of input]");
        return 0;
    }

    string inputType;
    switch (sortingType) {
    case 0: {
        inputType = "Randomized";
        break; }
    case 1: {
        inputType = "Sorted";
        break; }
    case 2: {
        inputType = "Reverse Sorted";
        break; }
    }

    int my_rank,        /* rank id of my process */
        num_ranks,      /* total number of ranks*/
        rc;             /* misc */

    /* MPI Setup */
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&num_ranks);
    /*if (num_ranks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }*/
    if(num_ranks != numProcesses) {
        printf("Target number of processes and actual number of ranks do not match. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    if(my_rank == 0) {
        printf("Input type: %d\n", sortingType);
        printf("Number Processes: %d\n", numProcesses); 
        printf("Input Size: %d\n", inputSize);  
    }

    CALI_MARK_BEGIN(mainFunction);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    //Data generation
    vector<int> myLocalData;
    int amountToGenerateMyself = inputSize/numProcesses; //Should aways be based around powers of 2
    int startingPos = my_rank * (amountToGenerateMyself);
    CALI_MARK_BEGIN(data_init);
    generateData(myLocalData, sortingType, amountToGenerateMyself, startingPos, my_rank);
    CALI_MARK_END(data_init);

    //Main Alg
    vector<int> sortedData;
    sampleSort(myLocalData, sortedData, my_rank);

    //Verification
    CALI_MARK_BEGIN(correctness_check);
    bool correct = verifyCorrect(sortedData, my_rank);
    CALI_MARK_END(correctness_check);

    CALI_MARK_END(mainFunction);
    if(!correct){printf("There is a problem with the sorting. Quitting...\n");}
    else {if(my_rank == 0){printf("\nAll data sorted correctly!");}}
  
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numProcesses); // The number of processors (MPI ranks)
    adiak::value("group_num", 16); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
   mgr.stop();
   mgr.flush();

   MPI_Finalize();
}
