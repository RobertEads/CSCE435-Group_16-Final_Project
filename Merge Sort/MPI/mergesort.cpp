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
}

void mergeSort(std::vector<int> &arr, int l, int r)
{

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
}

void generateData(std::vector<int> &localData, int startingSortChoice, int amountToGenerate, int startingPosition, int my_rank, int inputSize)
{
    CALI_MARK_BEGIN(data_init);
    srand((my_rank + 5) * (my_rank + 12) * 1235);

    switch (startingSortChoice)
    {
    case 0:
        // Random
        for (int i = 0; i < amountToGenerate; i++)
        {
            localData.push_back(rand() % inputSize);
        }
        break;

    case 1:
        // Sorted
        for (int i = startingPosition; i < startingPosition + amountToGenerate; i++)
        {
            localData.push_back(i);
        }
        break;

    case 2:
        // Reverse sorted
        for (int i = startingPosition + amountToGenerate - 1; i >= startingPosition; i--)
        {
            localData.push_back(i);
        }
        break;
        
    //  case 3:
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

int main(int argc, char *argv[]) {
    
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <sorting_type> <num_processors> <num_elements>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int sorting_type = std::atoi(argv[1]);
    int num_processors = std::atoi(argv[2]);
    int num_elements = std::atoi(argv[3]);
    
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> input_array;

    if (rank == 0) {
    // Manually create the input array
    generateData(input_array, sorting_type, num_elements / size, rank * (num_elements / size), rank, num_elements);

    // Print the initial array
    std::cout << "Initial Array: ";
    for (int i = 0; i < num_elements; ++i) {
        std::cout << input_array[i] << " ";
    }
    std::cout << std::endl;
}


    int local_size = num_elements / size;
    std::vector<int> local_arr(local_size);

    // Scatter the input array among processes
    MPI_Scatter(&input_array[0], local_size, MPI_INT, &local_arr[0], local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Display initial elements on each rank
    for (int i = 0; i < size; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == rank) {
            std::cout << "Rank " << rank << ": Initial Elements: ";
            for (int j = 0; j < local_size; ++j) {
                std::cout << local_arr[j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Perform local merge sort
    mergeSort(local_arr, 0, local_size - 1);

    // Gather the sorted local arrays back to the root process
    MPI_Gather(&local_arr[0], local_size, MPI_INT, &input_array[0], local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Merge the sorted local arrays on the root process
    if (rank == 0) {
    std::vector<int> sorted_result(num_elements);

    // Merge all sorted halves
    for (int i = 0; i < size; ++i) {
        merge(input_array, i * local_size, (i + 1) * local_size - 1, std::min((i + 2) * local_size - 1, num_elements - 1));
    }

    // Display the final sorted array on rank 0
    std::cout << "Rank 0: Final Sorted Array: ";
    for (int i = 0; i < num_elements; ++i) {
        std::cout << input_array[i] << " ";
    }
    std::cout << std::endl;
}

    MPI_Finalize();
    return 0;
}