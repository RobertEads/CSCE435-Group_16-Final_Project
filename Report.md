# CSCE 435 Group 16 project

## 0. Group number: 16

## 1. Group members:

1. Anjali Kumar
2. Shawn Mathen
3. Ashrita Vadlapatla
4. Robert Eads

## 2. Project topic (e.g., parallel sorting algorithms)

Parallel Sorting Algorithms

## 2a. Communication Method

Our team's primary method of communication will be GroupMe with Slack as a secondary method.

## 2b. Brief project description (what algorithms will you be comparing and on what architectures)

Each of the selected sort algorithms, Bubble, Merge, Selection, & Sample will be run in parallel using MPI and CUDA separately.

## 2c. Pseudocode for each parallel algorithm

### Algorithm 1: Bubble Sort

#### MPI

```
1. Distribute the data among processes using MPI_Scatter.
2. Each process performs a local sequential Bubble Sort on its portion of the data.
3. Exchange data with neighboring processes using MPI_Send and MPI_Recv for each iteration of the Bubble Sort.
4. Perform multiple iterations of the Bubble Sort while exchanging and comparing adjacent elements to sort the entire dataset.
5. Repeat the Bubble Sort and data exchange steps for the required number of iterations (in parallel) to ensure a fully sorted dataset.
6. Verify the correctness of the sorted data.
```

#### CUDA

```
1. Each CUDA thread loads a portion of the data into shared memory.
2. Perform a local sequential Bubble Sort within each thread's shared memory.
3. Synchronize threads to ensure all shared memory sorting is complete.
4. No need for pivot or reduction steps, as Bubble Sort doesn't require pivot-based partitioning.
5. Each thread can individually compare and swap adjacent elements in its partition as part of the Bubble Sort.
6. No offset calculations are needed in Bubble Sort.
7. There is no need for a scatter operation, as Bubble Sort is an in-place sorting algorithm.
8. Continue the Bubble Sort for the required number of iterations, ensuring all elements are correctly sorted.
9. There's no merging step in Bubble Sort, as it's an exchange-based sorting algorithm that works directly on the data in place.
```

#### General Pseudocode

```
begin BubbleSort(list)

   for all elements of list
      if list[i] > list[i+1]
         swap(list[i], list[i+1])
      end if
   end for

   return list

end BubbleSort
```

#### MPI Pseudocode

```
void parallelBubbleSort(vector<int> &localData) {
    int myRank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    int localN = localData.size();

    for (int i = 0; i < localN - 1; i++) {
        for (int j = 0; j < localN - i - 1; j++) {
            if (localData[j] > localData[j + 1]) {
                swap(localData[j], localData[j + 1]);
            }
        }
    }
}
```

#### CUDA Pseudocode

```
__global__ void bubbleSort(int *array, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        for (int i = 0; i < size - 1; i++) {
            for (int j = 0; j < size - i - 1; j++) {
                if (array[j] > array[j + 1]) {
                    int temp = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = temp;
                }
            }
        }
    }
}
```

### Algorithm 2: Merge Sort

#### MPI

```
1. Distribute data among processes
2. Each process performs a local sequential merge sort
3. Pairwise merging of sorted sublists using MPI_Send and MPI_Recv
4. Recursively perform merging until the data is fully sorted
5. Verify correctness
```

#### CUDA

```
1. Each CUDA thread loads a portion of the data into shared memory
2. Perform a local sequential merge sort within each thread's shared memory
3. Use CUDA parallel reduction to find the pivot elements
4. Broadcast to all threads
5. Each thread partitions its data around pivot
6. Calculate the offsets for each partition
7. CUDA scatter to move elements to their respective partitions
8. Recursively sort each partition using CUDA parallel sort
9. Perform parallel merge or merge in shared memory, depending on the size of the partitions
```

#### Pseudocode

Parallel:
The following parallel Merge Sort algorithms will be implemented using MPI.

```python
function parallel_merge_sort(arr, num_threads)
    if length(arr) <= 1
        return arr

    mid = length(arr) // 2
    left = arr[0...mid-1]
    right = arr[mid...]

    if num_threads > 1
        left_thread = start_thread(parallel_merge_sort(left, num_threads/2))
        right_thread = start_thread(parallel_merge_sort(right, num_threads/2))
        join_thread(left_thread)
        join_thread(right_thread)

    else
        left = merge_sort(left)
        right = merge_sort(right)

    return parallel_merge(left, right)
```

The following is pseudocode for a CUDA implementation

```c++
__global__ void mergeCUDA(int* arr, int left, int mid, int right) {
    // Sequential merge
}

__global__ void mergeSortParallelCUDA(int* arr, int size) {
    if (size <= 1) return;

    int mid = size / 2;

    int* left = arr;
    int* right = arr + mid;

    mergeSortParallelCUDA<<<1, 1>>>(left, mid);
    mergeSortParallelCUDA<<<1, 1>>>(right, size - mid);

    mergeCUDA(arr, 0, mid - 1, size - 1);
}

```

### Algorithm 3: Selection Sort

#### MPI

```
1. Distribute the data among processes using MPI_Scatter.
2. Each process applies the selection sort algorithm to its portion of the data.
3. The sort selects the minimum element and swaps it with the first unsorted element so that there is a local sorted segment.
3. Communicate with neighboring processes using MPI_Send and MPI_Recv to exchange boundary data so that elements are placed correctly.
4. There is no need for a scatter operation, as Selection Sort is an in-place sorting algorithm.
5. Repeat the Selection Sort for the required number of iterations, ensuring all elements are correctly sorted.
6. Verify the correctness of the sorted data.
```

#### CUDA

```
1. Each CUDA thread loads a portion of the data into shared memory
2. Implement selection sort within the shared memory to locally sort the data.
3. Utilize CUDA parallel reduction techniques to find the minimum element within the shared memory.
4. Share the minimum element among all threads.
5. Partition the data based on the broadcasted minimum element.
6. Determine offsets for each partition to facilitate data movement.
7. Utilize CUDA scatter operations to move elements to their respective partitions.
8. Continue the selection sort for a certain number of iterations.
9. As selection sort is an exchange-based sorting algorithm that works directly on the data in place, there is no merging.
```

```

void selectionSort(int arr[], int n)
{
    int i, j, min_idx;

    // One by one move boundary of
    // unsorted subarray
    for (i = 0; i < n - 1; i++) {

        // Find the minimum element in
        // unsorted array
        min_idx = i;
        for (j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx])
                min_idx = j;
        }

        // Swap the found minimum element
        // with the first element
        if (min_idx != i)
            swap(arr[min_idx], arr[i]);
    }
}

```

### Algorithm 4: Sample Sort

#### MPI

```
FOR number of splitters selected by each process
      Push back randomly selected value from local data to sample vector
Allocate memory for all sampled splitters
MPI_Allgather sampled splitters to each process
Sort sampled splitters and choose bucket bounds from thoughout sampled splitters
Evaluate local elements and place into send buffers
Calculate required buffer size from each process
FOR number of processes
      MPI_Gather the buffer sizes from each process on each process
Allocate receive buffer for incoming data
FOR number of processes
      MPI_Gatherv to send data to the correct process
Sort with sequential quicksort
```

#### CUDA

```
*All steps are performed with CUDA kernals with the expection of data transfer*
Sample and select splitters to use
Sort samples
Gather required info about each bucket
Group elements into buckets within an array
Sort the data within each bucket
Transfer data from GPU to CPU
```

## 2d. Citations

- https://www.tutorialspoint.com/data_structures_algorithms/bubble_sort_algorithm.htm
- https://compucademy.net/algorithmic-thinking-with-python-part-3-divide-and-conquer-strategy/#:~:text=There%20is%20a%20really%20clever%20trick%20that,the%20same%20type%20as%20the%20original%20problem.
- https://teivah.medium.com/parallel-merge-sort-in-java-e3213ae9fa2c
- https://www.geeksforgeeks.org/quick-sort/
- https://cse.buffalo.edu/faculty/miller/Courses/CSE702/Nicolas-Barrios-Fall-2021.pdf
- https://en.wikipedia.org/wiki/Samplesort
- https://www.geeksforgeeks.org/selection-sort/
- https://pushkar2196.wordpress.com/2017/04/19/mergesort-cuda-implementation/

## 2e. Evaluation plan - what and how will you measure and compare

Each algorithm with be run with the input types of sorted, reverse sorted, and randomized. The input sizes are planned to be 2^16, 2^20, and 2^24.
Both weak and strong scaling will be analyzed and compared against the other algorithms being tested.
The number of threads in a block on the GPU will be [64, 128, 512, 1024].

## 3. Project Implementation

The listed algorithms were fully implemented using both MPI and CUDA.

## 4. Performance Evaluation

### Sample Sort

#### MPI

##### Randomized

<!-- Strong -->

For randomized inputs, this sample sort implementation clearly displays a lack of strong scaling on all three input sizes tested, 2^16, 2^18, & 2^20. While there is a minor decrease in runtime for 64 processes, the rest of the graph does not share this trend. The pattern presented below can most likely be attributed to the computation portion of its runtime due to the increase in overhead and therefore required time when adding more processes.

![Total time-main-Randomized-Strong_Scaling](./Report_Images/SampleSort/SS-MPI_0.png)
![Total time-comm-Randomized-Strong_Scaling](./Report_Images/SampleSort/SS-MPI_1.png)
![Total time-comp-Randomized-Strong_Scaling](./Report_Images/SampleSort/SS-MPI_2.png)

<!-- Weak -->

Weak scaling for this sample sort implementation presents an interesting graph shape for randomized input data. While the main and comm sections have decent weak scaling performace between 16 and 64 processes and then have poor weak scaling after that, the computation portion of the algorithm has no indication of anything but poor weak scaling performance. This is likely to hamper the algorithms performance when working with larger inputs.

![main-Randomized-Weak_Scaling](./Report_Images/SampleSort/SS-MPI_3.png)
![comm-Randomized-Weak_Scaling](./Report_Images/SampleSort/SS-MPI_4.png)
![comp-Randomized-Weak_Scaling](./Report_Images/SampleSort/SS-MPI_5.png)

##### Sorted

<!-- Strong -->

For sorted inputs, while not quite true strong scaling, the sample sort implementation does a much better job with lower numbers of processes. The computation time stays roughly same from 2 processes up to 32 processes before fluxuating and eventually rising. The communication time shows a similar pattern to that of randomized inputs, and the graph for overall time moves closer to the computation graph for the beginning portion. This flat lining of computation time near the beginning could be due to the few number of buckets and underlying quick sort algorithm enabling it to only need a small number of runs to verifiy the data is sorted.

![Total time-main-Sorted-Strong_Scaling](./Report_Images/SampleSort/SS-MPI_6.png)
![Total time-comm-Sorted-Strong_Scaling](./Report_Images/SampleSort/SS-MPI_7.png)
![Total time-comp-Sorted-Strong_Scaling](./Report_Images/SampleSort/SS-MPI_8.png)

<!-- Weak -->

Weak scaling performance for a sorted input is very similar to that of the randomized input. Interesting to note however, the larger impact of the comp section can be seen as the main graph has less of a flatline between 16 and 64 processes.

![main-Sorted-Weak_Scaling](./Report_Images/SampleSort/SS-MPI_9.png)
![comm-Sorted-Weak_Scaling](./Report_Images/SampleSort/SS-MPI_10.png)
![comp-Sorted-Weak_Scaling](./Report_Images/SampleSort/SS-MPI_11.png)

##### Reverse Sorted

<!-- Strong -->

A very similar story is told by the reverse sorted graphs as by the sorted graph. Computation flatlines until about 32 processes and then increases wildly, communication displays poor strong scaling performance, and the overall graphs is good mix of the two. As this is the third comm graph to display a sharp decrease in runtime at the 64 process mark, it is likely this benifit is due to the CPUs on Grace being 24 cores each, 48 hardware threads, which means at 64 processes the second CPU is being used and the additional cache memory is now introduced into the equation during communication.

![Total time-main-Reverse-Strong_Scaling](./Report_Images/SampleSort/SS-MPI_12.png)
![Total time-comm-Reverse-Strong_Scaling](./Report_Images/SampleSort/SS-MPI_13.png)
![Total time-comp-Reverse-Strong_Scaling](./Report_Images/SampleSort/SS-MPI_14.png)

<!-- Weak -->

The weak scaling for reverse sorted inputs looks very similar to the other two input types. Comm has moderate weak scaling in the beginning, but comp's poor weak scaling performance over shadows any positives when it comes to overall runtime.

![main-Sorted-Weak_Scaling](./Report_Images/SampleSort/SS-MPI_15.png)
![comm-Sorted-Weak_Scaling](./Report_Images/SampleSort/SS-MPI_16.png)
![comp-Sorted-Weak_Scaling](./Report_Images/SampleSort/SS-MPI_17.png)

#### CUDA

##### Randomized

<!-- Strong -->

The CUDA implementation of sample sort does seem to display decent strong scaling performance for the larger input sizes. This is mainly due to the computation time as the comm graph is mostly flat and seems to have little overall impact if any. The sharp drop seen at 1024 threads for 2^16 input size seems to be an outlier as that data point does not make sense given the rest of the graph.

![Total time-main-Randomized-Strong_Scaling](./Report_Images/SampleSort/SS-CUDA_0.png)
![Total time-comm-Randomized-Strong_Scaling](./Report_Images/SampleSort/SS-CUDA_1.png)
![Total time-comp-Randomized-Strong_Scaling](./Report_Images/SampleSort/SS-CUDA_2.png)

<!-- Weak -->

There is not much to be said about the weak scaling performance of this sample sort implementation. It has very poor performance overall due to the similarly poor graphs in both the comm and comp regions.

![main-Randomized-Weak_Scaling](./Report_Images/SampleSort/SS-CUDA_3.png)
![comm-Randomized-Weak_Scaling](./Report_Images/SampleSort/SS-CUDA_4.png)
![comp-Randomized-Weak_Scaling](./Report_Images/SampleSort/SS-CUDA_5.png)

##### Sorted

<!-- Strong -->

When looking at strong scaling performance with sorted input, no clear picture emerges one way or the other. When compared to randomized input, sorted input seems to have worse overall strong scaling performance, but it's can be hard to tell due to the inconsistent shape of the graph. The comm regions are roughly the same between the two inputs, but computation for sorted is much less strongly scaled than randomized, leading to the main graph looking a bit wild.

![Total time-main-Sorted-Strong_Scaling](./Report_Images/SampleSort/SS-CUDA_6.png)
![Total time-comm-Sorted-Strong_Scaling](./Report_Images/SampleSort/SS-CUDA_7.png)
![Total time-comp-Sorted-Strong_Scaling](./Report_Images/SampleSort/SS-CUDA_8.png)

<!-- Weak -->

While the strong scaling got worse with sorted input, the weak scaling seems to be a little bit better. While comm did not change much, the comp regions' scaling improved and so the main graphs looks a little bit better between 64 an 256 threads.

![main-Sorted-Weak_Scaling](./Report_Images/SampleSort/SS-CUDA_9.png)
![comm-Sorted-Weak_Scaling](./Report_Images/SampleSort/SS-CUDA_10.png)
![comp-Sorted-Weak_Scaling](./Report_Images/SampleSort/SS-CUDA_11.png)

##### Reverse Sorted

<!-- Strong -->

Reverse sorted inputs for this sample sort implementation seems have decent strong scaling performance and get better as the size of the input increases. While the communication portion is still relatively flat, computation has fairly good strong scaling performance and that translates to the main graph as well. The trend being seen with this input type is most likely due to the underlying sorting algorithm being insertion sort. Because reverse sorted input is typically the worst case for insertion sort, being able to break it into smaller chunks likely helps to combact that usual disadvtanges that come with the algorithm.

![Total time-main-Reverse-Strong_Scaling](./Report_Images/SampleSort/SS-CUDA_12.png)
![Total time-comm-Reverse-Strong_Scaling](./Report_Images/SampleSort/SS-CUDA_13.png)
![Total time-comp-Reverse-Strong_Scaling](./Report_Images/SampleSort/SS-CUDA_14.png)

<!-- Weak -->

Weaking scaling for reverse sorted inputs appears to fall somewhere in between randomized and sorted. Comm is pretty much the same as before and comp is only slightly better than with randomized input, so the main graphs has fairly poor weak scaling performance overall.

![main-Reverse-Weak_Scaling](./Report_Images/SampleSort/SS-CUDA_15.png)
![comm-Reverse-Weak_Scaling](./Report_Images/SampleSort/SS-CUDA_16.png)
![comp-Reverse-Weak_Scaling](./Report_Images/SampleSort/SS-CUDA_17.png)

# Merge Sort

## Weak Scaling

### MPI
One thing to note when I did my weak scaling is that I chose to measure the average time per rank over number of processors. The reason I chose average time over total time is that disussing with the TA about how these sorts work, total time is always going to grow as you increase processors because its an aggregate of all times over all processors. Additionally, I wanted to begin by focusing on the main function times for everything because I thought a good introduction to the analysis is how the program as a whole ran on average.

#### RANDOM INPUT ARRAY
The analysis begins with weak scaling, specifically with MPI. For weak scaling, I began with the input type of a random array, where each element is a randomly generated number from 0 to n, where n is the size of the array. The most obvious trend between all of the graphs in this section is that the as you increase the number of elements, the algorithm performs more slowly. This is shown best with the largest input size of 2^28 having the largest graph, as signified by the pink line. It is interesting to note that the computation time for all the times looks relatively the same as we parallelize, and that most of the variation in times comes from the communication. I meausred both the MPI_Scatter and MPI_Gather functions, and the MPI_Gather function takes much longer than the scatter. In the sorted array input type, we see very similar behavior, where larger input sizes seemed to have a much longer time to complete. The same behavior is present in the reverse sorted array as well. Finally, in the data initialized with 1% of the data perturbed, we see the similar data. We can assume this dude to merge sort typically always breaking the arrays down into a single element subarray and merging them back together, so we should expect relatively consistent behavior.

![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak1.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak2.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak3.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak4.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak5.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak6.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak7.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak8.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak9.png)

#### SORTED INPUT ARRAY
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak10.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak11.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak12.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak13.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak14.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak15.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak16.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak17.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak18.png)

#### REVERSE SORTED INPUT ARRAY
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak19.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak20.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak21.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak22.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak23.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak24.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak25.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak26.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak27.png)

#### 1% PERTURBED INPUT ARRAY
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak28.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak29.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak30.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak31.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak32.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak33.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak34.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak35.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/mpi-weak36.png)

### CUDA
For the CUDA weak scaling of the different input types, we had a lot of the same activity as the MPI implementation. As we increased the number of elements in the input array, there was more time needed for the merge sort algorithm to complete. It also seems thats the CUDA implementation takes a bit longer to complete than the MPI implementation. We can see this most notably with the 2^28th line (the pink), where it takes over 120 seconds to complete the main_function, whereas the MPI implementation had a maximum of 40 seconds to complete. We also sea a lot steadier increase in time for the CUDA implementation. It is interesting to see that all of the different performance regions increase quite steadily. Another big difference with CUDA is how it handles communication. Rather than speaking to multiple processors like in MPI, CUDA handles only cudamemcpy, so we see rather constant communciation times between all of the different input sizes.

#### Random Array
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak1.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak2.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak3.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak4.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak5.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak6.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak7.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak8.png)

#### Sorted Array
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak9.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak10.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak11.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak12.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak13.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak14.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak15.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak16.png)

#### Reverse Sorted Array
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak17.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak18.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak19.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak20.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak21.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak22.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak23.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak24.png)

### 1% Perturbed Array
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak25.png) 
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak26.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak27.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak28.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak29.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak30.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak31.png)
![Average-Time-main-Weak Scaling](./Report_Images/MergeSort/cuda-weak32.png)

## Strong Scaling
### MPI
We see some similar behavior as the weak scaling here in terms of shape of the graph. The most notable one here is the comp and comp_large regions, which have the same exponentially decreasing graph, where each input type has relatively the same behavior. This is more or less expected given the nature of merge sort, where each input is eventually recursively broken down into single input subarrays. Especially considering that each graph has the same number of elements, we can understand why the computation is so much closer between the lines for this strong scaling. It is interesting because this strong scaling shows that MPI_Scatter is much slower than MPI_Gather, which is the opposite than from the weak scaling. We also see that as we increase the number of elements, our behavior becomes more erratic, which is most easily seen in the main_function graph of each input size. We're also seeing that the time peaks around 128 processors before decreasing and then further increasing again. For communication, we are seeing some steady increase in time as well, but overall pretty fast across the ranks. 
### 2^16 Elements
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong1.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong2.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong3.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong4.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong5.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong6.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong7.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong8.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong9.png)

### 2 ^18 Elements
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong10.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong11.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong12.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong13.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong14.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong15.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong16.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong17.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong18.png)

### 2^20 Elements
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong19.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong20.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong21.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong22.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong23.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong24.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong25.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong26.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong27.png)

### 2^22 Elements
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong28.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong29.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong30.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong31.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong32.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong33.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong34.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong35.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong36.png)

### 2^24 Elements
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong37.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong38.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong39.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong40.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong41.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong42.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong43.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong44.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong45.png)

### 2^26 Elements
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong46.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong47.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong48.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong49.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong50.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong51.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong52.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong53.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong54.png)

### 2^28 Elements
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong55.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong56.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong57.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong58.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong59.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong60.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong61.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong62.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/mpi-strong63.png)

### CUDA
Looking at the graphs for CUDA strong scaling before, the data seems to be much more erratic. Rather than steady increases, we have many fluctuations as we increase the number of threads at a certain size, regardless of the input type. There doesn't seem to be an input type that outperforms the others. This is rather expected because merge sort will continue to break down the elements, regardless of the input type. It does look like the random array input does take a little more time than the other inputs, as it generally sits above the rest of the input types. This is followed by the 1% perturbed array. However, it is important to note that the scale of these graphs tend to be very close together, which can signify that there is no real tradeoff here based on input type. We definietly see that the comm region of the code, which again is around the CUDAMemcpys, fluctuates quite a bit, regardless of input type. There is no definite winner here. We get the main differences in computation, but again due to scale, isn't very significant. It is interesting to see though what could potentially cause this. Random taking the longest makes sense because there are a lot of comparisons, but it may be concerning that the reverse sorted array is one of the quicker ones, since this should theoretically have the maximum number of comparisons to make. One thing to note is that these two slowest, the random and perturbed array, have the slowest data initialization timees as well, so this could be a cause. We also see that the merge step was done on the CPU, which introduces a bit of a bottleneck as we increase the problem size and number of threads.
### 2^16 Elements
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong1.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong2.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong3.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong4.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong5.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong6.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong7.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong8.png)

### 2 ^18 Elements
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong9.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong10.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong11.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong12.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong13.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong14.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong15.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong16.png)

### 2^20 Elements
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong17.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong18.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong19.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong20.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong21.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong22.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong23.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong24.png)

### 2^22 Elements
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong25.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong26.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong27.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong28.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong29.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong30.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong31.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong32.png)

### 2^24 Elements

![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong33.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong34.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong35.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong36.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong37.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong38.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong39.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong40.png)

### 2^26 Elements
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong41.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong42.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong43.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong44.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong45.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong46.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong47.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong48.png)

### 2^28 Elements
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong49.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong50.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong51.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong52.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong53.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong54.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong55.png)
![Average-Time-main-Strong Scaling](./Report_Images/MergeSort/cuda-strong56.png)

## Speedup
### MPI
When looking at the computation speedups, we see a linear trend between the input sizes. This shows the importance of parallelism. For all sizes, as we increase the number of processors, the pure computation time gets better and better, this showing the higher speedup. We do see that speedup for the 2^16 elements begins to taper off for the comp regions, indiating that the parallelism is good for very large sizes, but smaller sizes aren't as impacts by increasing processors. This makes sense because you would spend more percentage of your time increasing your processors and have more communication and face more overhead for the smaller values. If we look at the entire main region as a whole, we can see that the larger the input the size, the more speedup there is. This further reinforces that there is a need for parallelism, and a definite benefit. For all of the input sizes, there seems to be a peak speedup at 256 processors before decreasing again. After 256 processors, there may be so much overhead that we lose overall performance. It is a little difficult to look at it easily for the smaller sizes, but the speedup graphs tend to all show that a bigger input size benefits from this parallelism. We also see that for communication, speedup decreases for more processsors, again reiterating that we are likely facing some signifcant overhead.
### Random Array
![speedup-mpi](./Report_Images/MergeSort/scale-mpi1.png)
![speedup-mpi](./Report_Images/MergeSort/scale-mpi2.png)
![speedup-mpi](./Report_Images/MergeSort/scale-mpi3.png)

### Sorted Array
![speedup-mpi](./Report_Images/MergeSort/scale-mpi4.png)
![speedup-mpi](./Report_Images/MergeSort/scale-mpi5.png)
![speedup-mpi](./Report_Images/MergeSort/scale-mpi6.png)

### Reverse Sorted Array
![speedup-mpi](./Report_Images/MergeSort/scale-mpi7.png)
![speedup-mpi](./Report_Images/MergeSort/scale-mpi8.png)
![speedup-mpi](./Report_Images/MergeSort/scale-mpi9.png)

# 1% Perturbed Array
![speedup-mpi](./Report_Images/MergeSort/scale-mpi10.png)
![speedup-mpi](./Report_Images/MergeSort/scale-mpi11.png)
![speedup-mpi](./Report_Images/MergeSort/scale-mpi12.png)

### CUDA
In the CUDA speedup graphs, there doesn't seem to be any inherent benefit to paralleism. When we look at the main_function as a whole, we see that the speedup values tend to hover around 1.0. In fact, most of the lines tend to be below a value of 1, indicating that it is performing worse by being parallel. There also seems to be no pattern with the varying input size. We see that at the highest number of processors, the worst speedups are 2^16 elements and 2^28 elements, which are both the smallest and largest values respectively. It seems like as we increase processors, speedup for the communication is worst for the 2^16 element, but the computation is worst for 2^28 elements. Communication is hard to guage completely becuase there is quite a bit of fluctuation here, where computation speedup hovers around 1.0 and steadily decreases for most of the input sizes. This could also be because theres no real inter-process communication for CUDA like there is for MPI.
### Random Array
![speedup-cuda](./Report_Images/MergeSort/scale-cuda1.png)
![speedup-cuda](./Report_Images/MergeSort/scale-cuda2.png)
![speedup-cuda](./Report_Images/MergeSort/scale-cuda3.png)

### Sorted Array
![speedup-cuda](./Report_Images/MergeSort/scale-cuda4.png)
![speedup-cuda](./Report_Images/MergeSort/scale-cuda5.png)
![speedup-cuda](./Report_Images/MergeSort/scale-cuda6.png)

### Reverse Sorted Array
![speedup-cuda](./Report_Images/MergeSort/scale-cuda7.png)
![speedup-cuda](./Report_Images/MergeSort/scale-cuda8.png)
![speedup-cuda](./Report_Images/MergeSort/scale-cuda9.png)

### 1% Perturbed Array
![speedup-cuda](./Report_Images/MergeSort/scale-cuda10.png)
![speedup-cuda](./Report_Images/MergeSort/scale-cuda11.png)
![speedup-cuda](./Report_Images/MergeSort/scale-cuda12.png)


### Bubble Sort

#### Weak Scaling

##### MPI

Like Shawn for merge sort, I also opted to assess the average time per rank in relation to the number of processors instead of total time since total time invariably increases with the addition of processors, as it represents an aggregate of all times across all processors.

For this analysis, I examined weak scaling using 2^16 elements across all four input types (sorted, reverse sorted, randomized, and 1% perturbed). It is evident that as the number of processors increases, the sorted array consistently exhibits the fastest runtime. This scenario represents the optimal condition for bubble sort, as it involves zero swaps for elements. The efficiency of this input type remains largely unaffected by the increase in processors, given its inherently efficient nature. The next fastest is the 1% perturbed input, which demonstrates a noticeable increase in runtime as the number of processors grows. The subsequent rankings alternate between reverse sorted and randomized inputs, as expected, considering they require a greater number of swaps. Reverse sorted input is relatively insensitive to the number of processors, as it consistently involves the maximum number of swaps.
![Bubble-main-Weak-MPI](./Report_Images/BubbleSort/BubbleSort-MPI-Weak-Main-65536.jpg)

##### CUDA

For this analysis, I examined weak scaling using 2^16 elements across three input types (sorted, reverse sorted, and randomized). The results are unexpected, as all average times increase with the number of threads used. One would anticipate a decrease for some time, given that parallelization should enhance efficiency. Furthermore, similar to MPI, we observe that "Sorted" consistently ranks as the fastest, as expected, since no swaps are required. "Reverse sorted" and "randomized" exhibit variations in their next fastest rankings, which aligns with expectations as they entail more swaps. The graph suggests that for this input size (65536), parallelization is not beneficial. With additional graphs featuring different input sizes, we can discern the general trend.
![Bubble-main-Weak-CUDA](./Report_Images/BubbleSort/BubbleSort-CUDA-Weak-Main-65536.jpg)


### Bubble Sort

#### Weak Scaling

##### MPI
For the analysis, weak scaling was examined across the sorted input type. 2, 4, and 8 processors were run and it showed that there was an increase in average time. However, this was interesting as usually as the number of processors increases, the time should decrease due to parallelization and efficiency. However, since smaller numbers were run it might not be as easy to tell. Therefore, as more trials with more data is run, then the graph will show parallelization. The rest of the input types with the number of processors and threads will be run as the next time in order to gather more accurate data creating the various graphs. 
![selection-main-Weak-MPI](./Report_Images/SelectionSort/weakscale.png)

##### CUDA
For the analysis, for CUDA only two cali files with 16 were run so only two points are showing. It does seem that one of the points has a higher average time. Due to the small number of trials, not much can be observed from the graph. The jobs were queued on grace portal so the data was not fully able to be collected for this implementation. However, as the number of processors increases, the average time should decrease as the the paralleization of the tasks has been implemented. Therefore, the rest of the input types will be run 

![selection-main-Weak-CUDA](./Report_Images/SelectionSort/weakscaleCUDA.jpeg)