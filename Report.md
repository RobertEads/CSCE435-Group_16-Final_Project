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

Each of the selected sort algorithms, Bitonic, Merge, Selection, & Sample, will be run using MPI and CUDA separately.

## 2c. Pseudocode for each parallel algorithm

### Algorithm 1: Bitonic Sort

#### MPI

```
1. Distribute Data: Use MPI_Scatter to distribute the data among processes. Each process will receive a portion of the dataset.
2. Local Bitonic Sort: Each process performs a local Bitonic Sort on its portion of the data. This involves performing Bitonic Sort on the local dataset using parallel threads or any parallel programming model within the process.
3. Bitonic Merge: Implement the Bitonic Merge operation for pairs of processes. This involves exchanging data between neighboring processes to create a bitonic sequence.
4. Global Bitonic Sort: Perform multiple iterations of the Bitonic Sort while exchanging and comparing adjacent elements between neighboring processes. In each iteration, processes exchange data and perform Bitonic Merge.
5. Repeat for Required Iterations: Repeat the Bitonic Sort and data exchange steps for the required number of iterations to ensure a fully sorted dataset. The number of iterations depends on the size of the dataset and the number of processes.
6. Verify Correctness: After the sorting is complete, use a verification step to ensure the correctness of the sorted data. You can implement a global verification step where each process checks the correctness of its portion, or you can gather data to the root process for a centralized verification.
```

#### CUDA

```
1. Allocate Device Memory: Use cudaMalloc to allocate memory on the GPU for the dataset. Allocate memory for temporary buffers if needed.
2. Copy Data to GPU: Use cudaMemcpy to copy the data from the host to the GPU. Launch Kernel for Local Bitonic Sort:
3. Write a CUDA kernel to perform local Bitonic Sort on the GPU. Each thread will work on a portion of the dataset. Threads in the same block can cooperate to perform the sort using shared memory.
4. Synchronize Threads: Use __syncthreads() to synchronize threads within a block after the local sort.
5. Launch Bitonic Merge Kernel: Write a CUDA kernel to perform Bitonic Merge operation for pairs of blocks. This may involve exchanging data between neighboring blocks and creating bitonic sequences.
6. Synchronize Threads Again: Use __syncthreads() to synchronize threads after the merge operation.
7. Repeat for Multiple Iterations: Repeat steps 3-6 for the required number of iterations. The number of iterations depends on the size of the dataset and the structure of the bitonic sequence.
8. Copy Data Back to CPU: Use cudaMemcpy to copy the sorted data from the GPU back to the host.
9. Free Device Memory: Use cudaFree to release the allocated memory on the GPU.
10. Verify Correctness: After the sorting is complete, use a verification step to ensure the correctness of the sorted data. You can use a similar verification function as before.
```

#### General Pseudocode

```
void bitonicMerge(int a[], int low, int cnt, int dir)
{
    if (cnt>1)
    {
        int k = cnt/2;
        for (int i=low; i<low+k; i++)
            compAndSwap(a, i, i+k, dir);
        bitonicMerge(a, low, k, dir);
        bitonicMerge(a, low+k, k, dir);
    }
}

void bitonicSort(int a[],int low, int cnt, int dir)
{
    if (cnt>1)
    {
        int k = cnt/2;

        // sort in ascending order since dir here is 1
        bitonicSort(a, low, k, 1);

        // sort in descending order since dir here is 0
        bitonicSort(a, low+k, k, 0);

        // Will merge whole sequence in ascending order
        // since dir=1.
        bitonicMerge(a,low, cnt, dir);
    }
}
```

#### MPI Pseudocode

```
int main(int argc, char **argv) {
    // MPI Initialization

    // Scatter data among processes
    MPI_Scatter(/* ... */);

    // Local Bitonic Sort on each process
    localBitonicSort(/* ... */);

    // Global Bitonic Sort
    for (int k = 2; k <= numProcesses; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            // Bitonic Merge for pairs of processes
            bitonicMerge(/* ... */);
        }
    }

    // Gather sorted data to root process
    MPI_Gather(/* ... */);

    // MPI Finalization

    return 0;
}
```

#### CUDA Pseudocode

```
__global__ void bitonicSortKernel(int *data, int dataSize) {
    // Calculate thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Bitonic Sort
    for (int k = 2; k <= dataSize; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;

            if ((ixj) > tid) {
                if ((tid & k) == 0) {
                    // Sort ascending
                    if (data[tid] > data[ixj]) {
                        // Swap elements
                        int temp = data[tid];
                        data[tid] = data[ixj];
                        data[ixj] = temp;
                    }
                }
                if ((tid & k) != 0) {
                    // Sort descending
                    if (data[tid] < data[ixj]) {
                        // Swap elements
                        int temp = data[tid];
                        data[tid] = data[ixj];
                        data[ixj] = temp;
                    }
                }
            }

            __syncthreads(); // Synchronize threads after each iteration
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
*All steps are performed with CUDA kernals*
Sample elements
Sort samples
Select pivots from sampled elements
Gather required size and start position for each bucket
Group elements into buckets within an array
Have each threads sort the data within the bounds of its respective bucket
```

## 2d. Citations

### Bitonic Sort

- https://www.geeksforgeeks.org/bitonic-sort/
- https://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm

### Merge Sort

- https://compucademy.net/algorithmic-thinking-with-python-part-3-divide-and-conquer-strategy/#:~:text=There%20is%20a%20really%20clever%20trick%20that,the%20same%20type%20as%20the%20original%20problem.
- https://teivah.medium.com/parallel-merge-sort-in-java-e3213ae9fa2c
- https://pushkar2196.wordpress.com/2017/04/19/mergesort-cuda-implementation/

### Selection Sort

- https://www.geeksforgeeks.org/selection-sort/

### Sample Sort

- https://en.wikipedia.org/wiki/Samplesort
- https://cse.buffalo.edu/faculty/miller/Courses/CSE702/Nicolas-Barrios-Fall-2021.pdf
- https://www.geeksforgeeks.org/quicksort-using-random-pivoting/

## 2e. Evaluation plan - what and how will you measure and compare

Each algorithm with be run with the input types of randomized, sorted, reverse sorted, and 1% perturbed. The input sizes are planned to be 2^16, 2^18, 2^20, 2^22, 2^24, 2^26 and 2^28.
Graphs for strong and weak scaling, as well as strong speedup, will be created and used in the analysis and drawing of conclusions for each algorithm across the different input types and input sizes. A strong scaling comparison will also be run for the algorithms against each other to see if and where one will stand out from the rest.
The number of threads in a block on the GPU to be tested will be [64, 128, 256, 512, 1024].

## 3. Project Implementation

The listed algorithms were fully implemented using both MPI and CUDA separately.

## 4. Performance Evaluation

# Sample Sort

**Note:** The CUDA implementation of sample sort did not require any comm regions (cudaMemcpy), so the following section will have no comm graphs for the CUDA subsections.

## Strong Scaling

### MPI

#### main

The strong scaling graphs for main tell conflicting stories. Looking at the lower input sizes, up to 2^22, the algorithm looks to have poor strong scaling performance as it is relatively flat for the beginning before spiking up as it gets closer to the end. However, looking at the larger input sizes, from 2^24 and up, the graphs start to display good strong scaling performance as they slope downwards and flatten off once a point of diminishing returns is reached, around the 2^6 or 2^7 number of processes mark. This behavior is caused by the algorithm shifting from being communication-bound in the smaller input sizes to being computation-bound for the larger ones.

<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^16_main_Strong.png" alt="MPI_Strong_Scaling_main_2^16" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^18_main_Strong.png" alt="MPI_Strong_Scaling_main_2^18" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^20_main_Strong.png" alt="MPI_Strong_Scaling_main_2^20" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^22_main_Strong.png" alt="MPI_Strong_Scaling_main_2^22" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^24_main_Strong.png" alt="MPI_Strong_Scaling_main_2^24" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^26_main_Strong.png" alt="MPI_Strong_Scaling_main_2^26" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^28_main_Strong.png" alt="MPI_Strong_Scaling_main_2^28" width="325"/>

#### comm

The communication part of this algorithm has poor strong scaling but does not make huge jumps in time as the computation does. Comm starts at less than 0.1 seconds for the smaller input sizes and takes up to a little over 5 seconds for the largest input size. This means comm will dominate the overall runtime in the beginning, but then become less important as input sizes get larger. This matches what is seen in the graphs for main.

<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^16_comm_Strong.png" alt="MPI_Strong_Scaling_comm_2^16" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^18_comm_Strong.png" alt="MPI_Strong_Scaling_comm_2^18" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^20_comm_Strong.png" alt="MPI_Strong_Scaling_comm_2^20" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^22_comm_Strong.png" alt="MPI_Strong_Scaling_comm_2^22" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^24_comm_Strong.png" alt="MPI_Strong_Scaling_comm_2^24" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^26_comm_Strong.png" alt="MPI_Strong_Scaling_comm_2^26" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^28_comm_Strong.png" alt="MPI_Strong_Scaling_comm_2^28" width="325"/>

#### comp_large

Comp_large has good strong scaling for all input sizes and hits a point of diminishing returns somewhere in the 2^6 to 2^7 processes range. As expected, the runtime grows from less than a second to around 50 seconds as larger input sizes are tested. This further explains why computation dominates the graphs of main only for the larger input sizes, as 2^24 is where computation begins to consistently take more time than communication.

<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^16_comp_large_Strong.png" alt="MPI_Strong_Scaling_comp_large_2^16" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^18_comp_large_Strong.png" alt="MPI_Strong_Scaling_comp_large_2^18" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^20_comp_large_Strong.png" alt="MPI_Strong_Scaling_comp_large_2^20" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^22_comp_large_Strong.png" alt="MPI_Strong_Scaling_comp_large_2^22" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^24_comp_large_Strong.png" alt="MPI_Strong_Scaling_comp_large_2^24" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^26_comp_large_Strong.png" alt="MPI_Strong_Scaling_comp_large_2^26" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Strong/SS_MPI_2^28_comp_large_Strong.png" alt="MPI_Strong_Scaling_comp_large_2^28" width="325"/>

### CUDA

#### main

The strong scaling graphs for CUDA do not paint a very clear picture in either direction. The graphs are all over the place, but most tend to generally trend upwards overall, leading to a conclusion of poor strong scaling. This behavior is most likely due to the implementation not taking advantage of the GPU as much as it could. An interesting point to note when looking at these graphs, while the lines are very up and down, relative to the overall time taken, the range on the Y-axis is not that big.

<img src="./Report_Images/SampleSort/CUDA_Strong/SS_CUDA_Strong_2^16_main.png" alt="CUDA_Strong_Scaling_main_2^16" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Strong/SS_CUDA_Strong_2^18_main.png" alt="CUDA_Strong_Scaling_main_2^18" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Strong/SS_CUDA_Strong_2^20_main.png" alt="CUDA_Strong_Scaling_main_2^20" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Strong/SS_CUDA_Strong_2^22_main.png" alt="CUDA_Strong_Scaling_main_2^22" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Strong/SS_CUDA_Strong_2^24_main.png" alt="CUDA_Strong_Scaling_main_2^24" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Strong/SS_CUDA_Strong_2^26_main.png" alt="CUDA_Strong_Scaling_main_2^26" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Strong/SS_CUDA_Strong_2^28_main.png" alt="CUDA_Strong_Scaling_main_2^28" width="325"/>

#### comp_large

As CUDA did not have communication for the algorithm, comp_large looks fairly similar to main. While the 2^28 graph shape seems to be an outlier with the lines trending downwards, the rest generally trend upwards implying poor strong scaling performance. The up-and-down nature of these graphs is again most likely due to the algorithm not taking proper advantage of the GPU.

<img src="./Report_Images/SampleSort/CUDA_Strong/SS_CUDA_Strong_2^16_comp_large.png" alt="CUDA_Strong_Scaling_comp_large_2^16" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Strong/SS_CUDA_Strong_2^18_comp_large.png" alt="CUDA_Strong_Scaling_comp_large_2^18" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Strong/SS_CUDA_Strong_2^20_comp_large.png" alt="CUDA_Strong_Scaling_comp_large_2^20" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Strong/SS_CUDA_Strong_2^22_comp_large.png" alt="CUDA_Strong_Scaling_comp_large_2^22" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Strong/SS_CUDA_Strong_2^24_comp_large.png" alt="CUDA_Strong_Scaling_comp_large_2^24" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Strong/SS_CUDA_Strong_2^26_comp_large.png" alt="CUDA_Strong_Scaling_comp_large_2^26" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Strong/SS_CUDA_Strong_2^28_comp_large.png" alt="CUDA_Strong_Scaling_comp_large_2^28" width="325"/>

## Speed up

### MPI

#### main

The MPI Speedup graphs for main have a similar pattern of switching what they are bounded by as seen in the strong scaling graphs, but reversed. Each one starts with a fairly good speedup shape as it is dominated by the speedup of computation in the early process counts, then starts to fall off as the communication overhead becomes greater and greater. As expected, the larger the input size, generally the better the speedup the algorithm sees.

<img src="./Report_Images/SampleSort/MPI_Speedup/SS_MPI_Random_Speedup_main.png" alt="MPI_Random_Speedup_main" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Speedup/SS_MPI_Sorted_Speedup_main.png" alt="MPI_Sorted_Speedup_main" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Speedup/SS_MPI_Reverse_Speedup_main.png" alt="MPI_Reverse_Speedup_main" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Speedup/SS_MPI_Perturbed_Speedup_main.png" alt="MPI_Perturbed_Speedup_main" width="325"/>

#### comm

Communication speedup graphs tell an expected story when considering the region has poor strong scaling performance. It starts off around 1.0 for the smaller input sizes but always ends up nearing 0 by the time the largest input size is reached.

<img src="./Report_Images/SampleSort/MPI_Speedup/SS_MPI_Random_Speedup_comm.png" alt="MPI_Random_Speedup_comm" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Speedup/SS_MPI_Sorted_Speedup_comm.png" alt="MPI_Sorted_Speedup_comm" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Speedup/SS_MPI_Reverse_Speedup_comm.png" alt="MPI_Reverse_Speedup_comm" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Speedup/SS_MPI_Perturbed_Speedup_comm.png" alt="MPI_Perturbed_Speedup_comm" width="325"/>

#### comp_large

The comp_large speedup graphs are almost picture-perfect when talking about speedups found in the real world. All input types see large speedups in the beginning as more processes are added for small numbers, but then plateau to various degrees once it reaches the point of diminishing returns at around 2^7 processes. As expected, the larger input sizes tend to achieve better speedups as they allow the algorithm to better utilize the maximum allowed resources.

<img src="./Report_Images/SampleSort/MPI_Speedup/SS_MPI_Random_Speedup_comp_large.png" alt="MPI_Random_Speedup_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Speedup/SS_MPI_Sorted_Speedup_comp_large.png" alt="MPI_Sorted_Speedup_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Speedup/SS_MPI_Reverse_Speedup_comp_large.png" alt="MPI_Reverse_Speedup_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Speedup/SS_MPI_Perturbed_Speedup_comp_large.png" alt="MPI_Perturbed_Speedup_comp_large" width="325"/>

### CUDA

#### main

The CUDA speedup graphs for main show an interesting pattern of the larger input sizes hugging the 1.0 line and the smaller inputs bouncing around to a greater degree. This most likely means there are some small variations in the runtime no matter the input size and it is just more noticeable on the smaller sizes because their overall shorter times make them more susceptible to change.

<img src="./Report_Images/SampleSort/CUDA_Speedup/SS_CUDA_Random_Speedup_main.png" alt="CUDA_Random_Speedup_main" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Speedup/SS_CUDA_Sorted_Speedup_main.png" alt="CUDA_Sorted_Speedup_main" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Speedup/SS_CUDA_Reverse_Speedup_main.png" alt="CUDA_Reverse_Speedup_main" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Speedup/SS_CUDA_Perturbed_Speedup_main.png" alt="CUDA_Perturbed_Speedup_main" width="325"/>

#### comp_large

The comp_large graphs are similar to the main graphs, but there seems to be a little bit more randomness. This most likely means the data initialization and correctness check sections of the program also have unclear runtime trends, therefore stabilizing the overall times a little bit, leading to the calmer graphs for main shown above.

<img src="./Report_Images/SampleSort/CUDA_Speedup/SS_CUDA_Random_Speedup_comp_large.png" alt="CUDA_Random_Speedup_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Speedup/SS_CUDA_Sorted_Speedup_comp_large.png" alt="CUDA_Sorted_Speedup_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Speedup/SS_CUDA_Reverse_Speedup_comp_large.png" alt="CUDA_Reverse_Speedup_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Speedup/SS_CUDA_Perturbed_Speedup_comp_large.png" alt="CUDA_Perturbed_Speedup_comp_large" width="325"/>

## Weak Scaling

**Note:** Due to the longer runtimes of larger input sizes, the graph shape for the smaller inputs gets lost. To combat this, the left column of graphs has all input sizes and the right column has the same graph, but with the larger input sizes removed to enable analysis of smaller input sizes.

### MPI

#### main

Weak scaling in main for the MPI implementation seems pretty good when looking at all input sizes. It is not until just the smaller input sizes are graphed that the upward spike near the largest number of processes can be seen. This is due to the communication overhead for larger number of processes that can be seen the comm graphs below. Prior to that, the runtime is dominated by the good weak scaling of the comp_large region.

<img src="./Report_Images/SampleSort/MPI_Weak/SS_MPI_Random_Weak_main.png" alt="MPI_Random_Weak_main" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Small_Weak/SS_MPI_Random_small_Weak_main.png" alt="MPI_Random_small_Weak_main" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Weak/SS_MPI_Sorted_Weak_main.png" alt="MPI_Sorted_Weak_main" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Small_Weak/SS_MPI_Sorted_small_Weak_main.png" alt="MPI_Sorted_small_Weak_main" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Weak/SS_MPI_Reverse_Weak_main.png" alt="MPI_Reverse_Weak_main" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Small_Weak/SS_MPI_Reverse_small_Weak_main.png" alt="MPI_Reverse_small_Weak_main" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Weak/SS_MPI_Perturbed_Weak_main.png" alt="MPI_Perturbed_Weak_main" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Small_Weak/SS_MPI_Perturbed_small_Weak_main.png" alt="MPI_Perturbed_small_Weak_main" width="325"/>

#### comm

Graphs for the comm region, with the exception of the largest input size, seem to show decent weak scaling for lower numbers of processes. It is not until 2^6 or 2^7, when more than one node and therefore network communication is added, do the comm graphs start to show poor weak scaling and bleed into the main graphs.

<img src="./Report_Images/SampleSort/MPI_Weak/SS_MPI_Random_Weak_comm.png" alt="MPI_Random_Weak_comm" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Small_Weak/SS_MPI_Random_small_Weak_comm.png" alt="MPI_Random_small_Weak_comm" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Weak/SS_MPI_Sorted_Weak_comm.png" alt="MPI_Sorted_Weak_comm" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Small_Weak/SS_MPI_Sorted_small_Weak_comm.png" alt="MPI_Sorted_small_Weak_comm" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Weak/SS_MPI_Reverse_Weak_comm.png" alt="MPI_Reverse_Weak_comm" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Small_Weak/SS_MPI_Reverse_small_Weak_comm.png" alt="MPI_Reverse_small_Weak_comm" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Weak/SS_MPI_Perturbed_Weak_comm.png" alt="MPI_Perturbed_Weak_comm" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Small_Weak/SS_MPI_Perturbed_small_Weak_comm.png" alt="MPI_Perturbed_small_Weak_comm" width="325"/>

#### comp_large

The graphs for comp_large show good weak scaling across the board for all input sizes as they have a steep drop at the beginning and then flatten out at the end. An interesting observation that can be seen on the graph of smaller input sizes is that sizes of 2^22 and under do not see as steep of an initial dropoff as the larger sizes do.

<img src="./Report_Images/SampleSort/MPI_Weak/SS_MPI_Random_Weak_comp_large.png" alt="MPI_Random_Weak_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Small_Weak/SS_MPI_Random_small_Weak_comp_large.png" alt="MPI_Random_small_Weak_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Weak/SS_MPI_Sorted_Weak_comp_large.png" alt="MPI_Sorted_Weak_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Small_Weak/SS_MPI_Sorted_small_Weak_comp_large.png" alt="MPI_Sorted_small_Weak_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Weak/SS_MPI_Reverse_Weak_comp_large.png" alt="MPI_Reverse_Weak_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Small_Weak/SS_MPI_Reverse_small_Weak_comp_large.png" alt="MPI_Reverse_small_Weak_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Weak/SS_MPI_Perturbed_Weak_comp_large.png" alt="MPI_Perturbed_Weak_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/MPI_Small_Weak/SS_MPI_Perturbed_small_Weak_comp_large.png" alt="MPI_Perturbed_small_Weak_comp_large" width="325"/>

### CUDA

#### main

The CUDA weak scaling graphs for main show fairly flat lines overall on the graphs with larger input sizes implying good weak scaling performance. When looking at just the smaller input sizes, while a bit more variation can be seen, the Y-axis scale is much more narrow so small ups and downs are not surprising.

<img src="./Report_Images/SampleSort/CUDA_Weak/SS_CUDA_Random_Weak_main.png" alt="CUDA_Random_Weak_main" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Small_Weak/SS_CUDA_Random_small_Weak_main.png" alt="CUDA_Random_small_Weak_main" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Weak/SS_CUDA_Sorted_Weak_main.png" alt="CUDA_Sorted_Weak_main" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Small_Weak/SS_CUDA_Sorted_small_Weak_main.png" alt="CUDA_Sorted_small_Weak_main" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Weak/SS_CUDA_Reverse_Weak_main.png" alt="CUDA_Reverse_Weak_main" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Small_Weak/SS_CUDA_Reverse_small_Weak_main.png" alt="CUDA_Reverse_small_Weak_main" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Weak/SS_CUDA_Perturbed_Weak_main.png" alt="CUDA_Perturbed_Weak_main" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Small_Weak/SS_CUDA_Perturbed_small_Weak_main.png" alt="CUDA_Perturbed_small_Weak_main" width="325"/>

#### comp_large

Weak scaling comp_large graphs echo the story told by the main graphs with the exception of the smaller input sizes. These maintain much more of a flat line which means the variation found in the graphs above is due to either data initialization or correctness checking runtimes fluctuating, not the actual sorting computation time.

<img src="./Report_Images/SampleSort/CUDA_Weak/SS_CUDA_Random_Weak_comp_large.png" alt="CUDA_Random_Weak_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Small_Weak/SS_CUDA_Random_small_Weak_comp_large.png" alt="CUDA_Random_small_Weak_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Weak/SS_CUDA_Sorted_Weak_comp_large.png" alt="CUDA_Sorted_Weak_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Small_Weak/SS_CUDA_Sorted_small_Weak_comp_large.png" alt="CUDA_Sorted_small_Weak_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Weak/SS_CUDA_Reverse_Weak_comp_large.png" alt="CUDA_Reverse_Weak_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Small_Weak/SS_CUDA_Reverse_small_Weak_comp_large.png" alt="CUDA_Reverse_small_Weak_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Weak/SS_CUDA_Perturbed_Weak_comp_large.png" alt="CUDA_Perturbed_Weak_comp_large" width="325"/>
<img src="./Report_Images/SampleSort/CUDA_Small_Weak/SS_CUDA_Perturbed_small_Weak_comp_large.png" alt="CUDA_Perturbed_small_Weak_comp_large" width="325"/>

# Merge Sort

## Weak Scaling

### MPI

One thing to note when I did my weak scaling is that I chose to measure the average time per rank over number of processors. The reason I chose average time over total time is that disussing with the TA about how these sorts work, total time is always going to grow as you increase processors because its an aggregate of all times over all processors. Additionally, I wanted to begin by focusing on the main function times for everything because I thought a good introduction to the analysis is how the program as a whole ran on average.

#### RANDOM INPUT ARRAY

The analysis begins with weak scaling, specifically with MPI. For weak scaling, I began with the input type of a random array, where each element is a randomly generated number from 0 to n, where n is the size of the array. The most obvious trend between all of the graphs in this section is that the as you increase the number of elements, the algorithm performs more slowly. This is shown best with the largest input size of 2^28 having the largest graph, as signified by the pink line. It is interesting to note that the computation time for all the times looks relatively the same as we parallelize, and that most of the variation in times comes from the communication. I meausred both the MPI_Scatter and MPI_Gather functions, and the MPI_Gather function takes much longer than the scatter. In the sorted array input type, we see very similar behavior, where larger input sizes seemed to have a much longer time to complete. The same behavior is present in the reverse sorted array as well. Finally, in the data initialized with 1% of the data perturbed, we see the similar data. We can assume this due to merge sort typically always breaking the arrays down into a single element subarray and merging them back together, so we should expect relatively consistent behavior.

<img src="./Report_Images/MergeSort/mpi-weak1.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak2.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak3.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak4.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak5.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak6.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak7.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak8.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak9.png" alt="Average-Time-main-Weak Scaling" width="400">

#### SORTED INPUT ARRAY

<img src="./Report_Images/MergeSort/mpi-weak10.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak11.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak12.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak13.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak14.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak15.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak16.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak17.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak18.png" alt="Average-Time-main-Weak Scaling" width="400">

#### REVERSE SORTED INPUT ARRAY

<img src="./Report_Images/MergeSort/mpi-weak19.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak20.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak21.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak22.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak23.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak24.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak25.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak26.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak27.png" alt="Average-Time-main-Weak Scaling" width="400">

#### 1% PERTURBED INPUT ARRAY

<img src="./Report_Images/MergeSort/mpi-weak28.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak29.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak30.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak31.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak32.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak33.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak34.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak35.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-weak36.png" alt="Average-Time-main-Weak Scaling" width="400">

### CUDA

For the CUDA weak scaling of the different input types, we had a lot of the same activity as the MPI implementation. As we increased the number of elements in the input array, there was more time needed for the merge sort algorithm to complete. It also seems thats the CUDA implementation takes a bit longer to complete than the MPI implementation. We can see this most notably with the 2^28th line (the pink), where it takes over 120 seconds to complete the main_function, whereas the MPI implementation had a maximum of 40 seconds to complete. We also sea a lot steadier increase in time for the CUDA implementation. It is interesting to see that all of the different performance regions increase quite steadily. Another big difference with CUDA is how it handles communication. Rather than speaking to multiple processors like in MPI, CUDA handles only cudamemcpy, so we see rather constant communciation times between all of the different input sizes.

#### Random Array

<img src="./Report_Images/MergeSort/cuda-weak1.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak2.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak3.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak4.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak5.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak6.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak7.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak8.png" alt="Average-Time-main-Weak Scaling" width="400">

#### Sorted Array

<img src="./Report_Images/MergeSort/cuda-weak9.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak10.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak11.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak12.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak13.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak14.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak15.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak16.png" alt="Average-Time-main-Weak Scaling" width="400">

#### Reverse Sorted Array

<img src="./Report_Images/MergeSort/cuda-weak17.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak18.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak19.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak20.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak21.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak22.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak23.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak24.png" alt="Average-Time-main-Weak Scaling" width="400">

### 1% Perturbed Array

<img src="./Report_Images/MergeSort/cuda-weak25.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak26.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak27.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak28.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak29.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak30.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak31.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-weak32.png" alt="Average-Time-main-Weak Scaling" width="400">

## Strong Scaling

### MPI

We see some similar behavior as the weak scaling here in terms of shape of the graph. The most notable one here is the comp and comp_large regions, which have the same exponentially decreasing graph, where each input type has relatively the same behavior. This is more or less expected given the nature of merge sort, where each input is eventually recursively broken down into single input subarrays. Especially considering that each graph has the same number of elements, we can understand why the computation is so much closer between the lines for this strong scaling. It is interesting because this strong scaling shows that MPI_Scatter is much slower than MPI_Gather, which is the opposite than from the weak scaling. We also see that as we increase the number of elements, our behavior becomes more erratic, which is most easily seen in the main_function graph of each input size. We're also seeing that the time peaks around 128 processors before decreasing and then further increasing again. For communication, we are seeing some steady increase in time as well, but overall pretty fast across the ranks.

### 2^16 Elements

<img src="./Report_Images/MergeSort/mpi-strong1.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong2.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong3.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong4.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong5.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong6.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong7.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong8.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong9.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2 ^18 Elements

<img src="./Report_Images/MergeSort/mpi-strong10.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong11.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong12.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong13.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong14.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong15.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong16.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong17.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong18.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^20 Elements

<img src="./Report_Images/MergeSort/mpi-strong19.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong20.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong21.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong22.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong23.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong24.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong25.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong26.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong27.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^22 Elements

<img src="./Report_Images/MergeSort/mpi-strong28.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong29.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong30.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong31.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong32.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong33.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong34.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong35.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong36.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^24 Elements

<img src="./Report_Images/MergeSort/mpi-strong37.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong38.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong39.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong40.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong41.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong42.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong43.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong44.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong45.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^26 Elements

<img src="./Report_Images/MergeSort/mpi-strong46.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong47.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong48.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong49.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong50.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong51.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong52.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong53.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong54.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^28 Elements

<img src="./Report_Images/MergeSort/mpi-strong55.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong56.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong57.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong58.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong59.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong60.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong61.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong62.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/mpi-strong63.png" alt="Average-Time-main-Strong Scaling" width="400">

### CUDA

Looking at the graphs for CUDA strong scaling before, the data seems to be much more erratic. Rather than steady increases, we have many fluctuations as we increase the number of threads at a certain size, regardless of the input type. There doesn't seem to be an input type that outperforms the others. This is rather expected because merge sort will continue to break down the elements, regardless of the input type. It does look like the random array input does take a little more time than the other inputs, as it generally sits above the rest of the input types. This is followed by the 1% perturbed array. However, it is important to note that the scale of these graphs tend to be very close together, which can signify that there is no real tradeoff here based on input type. We definietly see that the comm region of the code, which again is around the CUDAMemcpys, fluctuates quite a bit, regardless of input type. There is no definite winner here. We get the main differences in computation, but again due to scale, isn't very significant. It is interesting to see though what could potentially cause this. Random taking the longest makes sense because there are a lot of comparisons, but it may be concerning that the reverse sorted array is one of the quicker ones, since this should theoretically have the maximum number of comparisons to make. One thing to note is that these two slowest, the random and perturbed array, have the slowest data initialization timees as well, so this could be a cause.

Looking at the graphs for CUDA strong scaling before, the data seems to be much more erratic. Rather than steady increases, we have many fluctuations as we increase the number of threads at a certain size, regardless of the input type. There doesn't seem to be an input type that outperforms the others. This is rather expected because merge sort will continue to break down the elements, regardless of the input type. It does look like the random array input does take a little more time than the other inputs, as it generally sits above the rest of the input types. This is followed by the 1% perturbed array. However, it is important to note that the scale of these graphs tend to be very close together, which can signify that there is no real tradeoff here based on input type. We definietly see that the comm region of the code, which again is around the CUDAMemcpys, fluctuates quite a bit, regardless of input type. There is no definite winner here. We get the main differences in computation, but again due to scale, isn't very significant. It is interesting to see though what could potentially cause this. Random taking the longest makes sense because there are a lot of comparisons, but it may be concerning that the reverse sorted array is one of the quicker ones, since this should theoretically have the maximum number of comparisons to make. One thing to note is that these two slowest, the random and perturbed array, have the slowest data initialization timees as well, so this could be a cause. We also see that the merge step was done on the CPU, which introduces a bit of a bottleneck as we increase the problem size and number of threads.

### 2^16 Elements

<img src="./Report_Images/MergeSort/cuda-strong1.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong2.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong3.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong4.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong5.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong6.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong7.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong8.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2 ^18 Elements

<img src="./Report_Images/MergeSort/cuda-strong9.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong10.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong11.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong12.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong13.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong14.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong15.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong16.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^20 Elements

<img src="./Report_Images/MergeSort/cuda-strong17.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong18.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong19.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong20.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong21.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong22.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong23.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong24.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^22 Elements

<img src="./Report_Images/MergeSort/cuda-strong25.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong26.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong27.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong28.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong29.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong30.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong31.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong32.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^24 Elements

<img src="./Report_Images/MergeSort/cuda-strong33.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong34.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong35.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong36.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong37.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong38.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong39.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong40.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^26 Elements

<img src="./Report_Images/MergeSort/cuda-strong41.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong42.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong43.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong44.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong45.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong46.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong47.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong48.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^28 Elements

<img src="./Report_Images/MergeSort/cuda-strong49.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong50.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong51.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong52.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong53.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong54.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong55.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/MergeSort/cuda-strong56.png" alt="Average-Time-main-Strong Scaling" width="400">

## Speedup

### MPI

When looking at the computation speedups, we see a linear trend between the input sizes. This shows the importance of parallelism. For all sizes, as we increase the number of processors, the pure computation time gets better and better, this showing the higher speedup. We do see that speedup for the 2^16 elements begins to taper off for the comp regions, indiating that the parallelism is good for very large sizes, but smaller sizes aren't as impacts by increasing processors. This makes sense because you would spend more percentage of your time increasing your processors and have more communication and face more overhead for the smaller values. If we look at the entire main region as a whole, we can see that the larger the input the size, the more speedup there is. This further reinforces that there is a need for parallelism, and a definite benefit. For all of the input sizes, there seems to be a peak speedup at 256 processors before decreasing again. After 256 processors, there may be so much overhead that we lose overall performance. It is a little difficult to look at it easily for the smaller sizes, but the speedup graphs tend to all show that a bigger input size benefits from this parallelism. We also see that for communication, speedup decreases for more processsors, again reiterating that we are likely facing some signifcant overhead.

### Random Array

<img src="./Report_Images/MergeSort/scale-mpi1.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/MergeSort/scale-mpi2.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/MergeSort/scale-mpi3.png" alt="speedup-mpi" width="400">

### Sorted Array

<img src="./Report_Images/MergeSort/scale-mpi4.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/MergeSort/scale-mpi5.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/MergeSort/scale-mpi6.png" alt="speedup-mpi" width="400">

### Reverse Sorted Array

<img src="./Report_Images/MergeSort/scale-mpi7.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/MergeSort/scale-mpi8.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/MergeSort/scale-mpi9.png" alt="speedup-mpi" width="400">

# 1% Perturbed Array

<img src="./Report_Images/MergeSort/scale-mpi10.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/MergeSort/scale-mpi11.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/MergeSort/scale-mpi12.png" alt="speedup-mpi" width="400">

### CUDA

In the CUDA speedup graphs, there doesn't seem to be any inherent benefit to paralleism. When we look at the main_function as a whole, we see that the speedup values tend to hover around 1.0. In fact, most of the lines tend to be below a value of 1, indicating that it is performing worse by being parallel. There also seems to be no pattern with the varying input size. We see that at the highest number of processors, the worst speedups are 2^16 elements and 2^28 elements, which are both the smallest and largest values respectively. It seems like as we increase processors, speedup for the communication is worst for the 2^16 element, but the computation is worst for 2^28 elements. Communication is hard to guage completely becuase there is quite a bit of fluctuation here, where computation speedup hovers around 1.0 and steadily decreases for most of the input sizes. This could also be because theres no real inter-process communication for CUDA like there is for MPI.

### Random Array

<img src="./Report_Images/MergeSort/scale-cuda1.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/MergeSort/scale-cuda2.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/MergeSort/scale-cuda3.png" alt="speedup-cuda" width="400">

### Sorted Array

<img src="./Report_Images/MergeSort/scale-cuda4.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/MergeSort/scale-cuda5.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/MergeSort/scale-cuda6.png" alt="speedup-cuda" width="400">

### Reverse Sorted Array

<img src="./Report_Images/MergeSort/scale-cuda7.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/MergeSort/scale-cuda8.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/MergeSort/scale-cuda9.png" alt="speedup-cuda" width="400">

### 1% Perturbed Array

<img src="./Report_Images/MergeSort/scale-cuda10.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/MergeSort/scale-cuda11.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/MergeSort/scale-cuda12.png" alt="speedup-cuda" width="400">

# Bitonic Sort

## Weak Scaling

### MPI

Like Shawn for merge sort, I also opted to assess the average time per rank in relation to the number of processors instead of total time since total time invariably increases with the addition of processors, as it represents an aggregate of all times across all processors.

For this analysis, I examined weak scaling using 2^16 elements across all four input types (sorted, reverse sorted, randomized, and 1% perturbed). It is evident that as the number of processors increases, the sorted array consistently exhibits the fastest runtime. This scenario represents the optimal condition for bubble sort, as it involves zero swaps for elements. The efficiency of this input type remains largely unaffected by the increase in processors, given its inherently efficient nature. The next fastest is the 1% perturbed input, which demonstrates a noticeable increase in runtime as the number of processors grows. The subsequent rankings alternate between reverse sorted and randomized inputs, as expected, considering they require a greater number of swaps. Reverse sorted input is relatively insensitive to the number of processors, as it consistently involves the maximum number of swaps.
FIX ME!

#### RANDOM INPUT ARRAY

<img src="./Report_Images/BitonicSort/mpi-weak-1.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-2.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-3.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-4.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-5.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-6.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-7.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-8.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-9.png" alt="Average-Time-main-Weak Scaling" width="400">

#### SORTED INPUT ARRAY

<img src="./Report_Images/BitonicSort/mpi-weak-10.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-11.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-12.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-13.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-14.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-15.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-16.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-17.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-18.png" alt="Average-Time-main-Weak Scaling" width="400">

#### REVERSE SORTED INPUT ARRAY

<img src="./Report_Images/BitonicSort/mpi-weak-19.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-20.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-21.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-22.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-23.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-24.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-25.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-26.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-27.png" alt="Average-Time-main-Weak Scaling" width="400">

#### 1% PERTURBED INPUT ARRAY

<img src="./Report_Images/BitonicSort/mpi-weak-28.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-29.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-30.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-31.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-32.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-33.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-34.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-35.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-weak-36.png" alt="Average-Time-main-Weak Scaling" width="400">

### CUDA

For this analysis, I examined weak scaling using 2^16 elements across three input types (sorted, reverse sorted, and randomized). The results are unexpected, as all average times increase with the number of threads used. One would anticipate a decrease for some time, given that parallelization should enhance efficiency. Furthermore, similar to MPI, we observe that "Sorted" consistently ranks as the fastest, as expected, since no swaps are required. "Reverse sorted" and "randomized" exhibit variations in their next fastest rankings, which aligns with expectations as they entail more swaps. The graph suggests that for this input size (65536), parallelization is not beneficial. With additional graphs featuring different input sizes, we can discern the general trend.
FIX ME!

#### RANDOM INPUT ARRAY

<img src="./Report_Images/BitonicSort/cuda-weak-1.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-2.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-3.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-4.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-5.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-6.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-7.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-8.png" alt="Average-Time-main-Weak Scaling" width="400">

#### SORTED INPUT ARRAY

<img src="./Report_Images/BitonicSort/cuda-weak-9.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-10.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-11.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-12.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-13.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-14.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-15.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak1-6.png" alt="Average-Time-main-Weak Scaling" width="400">

#### REVERSE SORTED INPUT ARRAY

<img src="./Report_Images/BitonicSort/cuda-weak-17.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-18.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-19.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-20.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-21.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-22.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-23.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-24.png" alt="Average-Time-main-Weak Scaling" width="400">

#### 1% PERTURBED INPUT ARRAY

<img src="./Report_Images/BitonicSort/cuda-weak-25.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-26.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-27.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-28.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-29.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-30.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-31.png" alt="Average-Time-main-Weak Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-weak-32.png" alt="Average-Time-main-Weak Scaling" width="400">

## Weak Scaling

### MPI

For the analysis, weak scaling was examined across the sorted input type. 2, 4, and 8 processors were run and it showed that there was an increase in average time. However, this was interesting as usually as the number of processors increases, the time should decrease due to parallelization and efficiency. However, since smaller numbers were run it might not be as easy to tell. Therefore, as more trials with more data is run, then the graph will show parallelization. The rest of the input types with the number of processors and threads will be run as the next time in order to gather more accurate data creating the various graphs.
FIX ME!

### 2^16 Elements

<img src="./Report_Images/BitonicSort/mpi-strong-1.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-2.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-3.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-4.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-5.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-6.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-7.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-8.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-9.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2 ^18 Elements

<img src="./Report_Images/BitonicSort/mpi-strong-10.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-11.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-12.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-13.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-14.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-15.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-16.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-17.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-18.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^20 Elements

<img src="./Report_Images/BitonicSort/mpi-strong-19.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-20.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-21.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-22.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-23.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-24.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-25.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-26.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-27.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^22 Elements

<img src="./Report_Images/BitonicSort/mpi-strong-28.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-29.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-30.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-31.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-32.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-33.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-34.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-35.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-36.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^24 Elements

<img src="./Report_Images/BitonicSort/mpi-strong-37.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-38.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-39.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-40.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-41.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-42.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-43.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-44.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-45.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^26 Elements

<img src="./Report_Images/BitonicSort/mpi-strong-46.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-47.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-48.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-49.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-50.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-51.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-52.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-53.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-54.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^28 Elements

<img src="./Report_Images/BitonicSort/mpi-strong-55.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-56.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-57.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-58.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-59.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-60.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-61.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-62.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/mpi-strong-63.png" alt="Average-Time-main-Strong Scaling" width="400">

### CUDA

For the analysis, for CUDA only two cali files with 16 were run so only two points are showing. It does seem that one of the points has a higher average time. Due to the small number of trials, not much can be observed from the graph. The jobs were queued on grace portal so the data was not fully able to be collected for this implementation. However, as the number of processors increases, the average time should decrease as the the paralleization of the tasks has been implemented. Therefore, the rest of the input types will be run
FIX ME!

### 2^16 Elements

<img src="./Report_Images/BitonicSort/cuda-strong-1.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-2.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-3.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-4.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-5.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-6.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-7.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-8.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2 ^18 Elements

<img src="./Report_Images/BitonicSort/cuda-strong-9.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-10.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-11.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-12.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-13.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-14.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-15.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-16.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^20 Elements

<img src="./Report_Images/BitonicSort/cuda-strong-17.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-18.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-19.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-20.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-21.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-22.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-23.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-24.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^22 Elements

<img src="./Report_Images/BitonicSort/cuda-strong-25.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-26.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-27.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-28.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-29.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-30.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-31.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-32.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^24 Elements

<img src="./Report_Images/BitonicSort/cuda-strong-33.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-34.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-35.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-36.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-37.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-38.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-39.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-40.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^26 Elements

<img src="./Report_Images/BitonicSort/cuda-strong-41.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-42.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-43.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-44.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-45.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-46.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-47.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-48.png" alt="Average-Time-main-Strong Scaling" width="400">

### 2^28 Elements

<img src="./Report_Images/BitonicSort/cuda-strong-49.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-50.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-51.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-52.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-53.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-54.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-55.png" alt="Average-Time-main-Strong Scaling" width="400">
<img src="./Report_Images/BitonicSort/cuda-strong-56.png" alt="Average-Time-main-Strong Scaling" width="400">

## Speedup

### MPI

When looking at the computation speedups, we see a linear trend between the input sizes. This shows the importance of parallelism. For all sizes, as we increase the number of processors, the pure computation time gets better and better, this showing the higher speedup. We do see that speedup for the 2^16 elements begins to taper off for the comp regions, indiating that the parallelism is good for very large sizes, but smaller sizes aren't as impacts by increasing processors. This makes sense because you would spend more percentage of your time increasing your processors and have more communication and face more overhead for the smaller values. If we look at the entire main region as a whole, we can see that the larger the input the size, the more speedup there is. This further reinforces that there is a need for parallelism, and a definite benefit. For all of the input sizes, there seems to be a peak speedup at 256 processors before decreasing again. After 256 processors, there may be so much overhead that we lose overall performance. It is a little difficult to look at it easily for the smaller sizes, but the speedup graphs tend to all show that a bigger input size benefits from this parallelism. We also see that for communication, speedup decreases for more processsors, again reiterating that we are likely facing some signifcant overhead.
FIX ME!

### RANDOM INPUT ARRAY

<img src="./Report_Images/BitonicSort/mpi-speed-1.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/BitonicSort/mpi-speed-2.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/BitonicSort/mpi-speed-3.png" alt="speedup-mpi" width="400">

#### SORTED INPUT ARRAY

<img src="./Report_Images/BitonicSort/mpi-speed-4.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/BitonicSort/mpi-speed-5.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/BitonicSort/mpi-speed-6.png" alt="speedup-mpi" width="400">

#### REVERSE SORTED INPUT ARRAY

<img src="./Report_Images/BitonicSort/mpi-speed-7.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/BitonicSort/mpi-speed-8.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/BitonicSort/mpi-speed-9.png" alt="speedup-mpi" width="400">

#### 1% PERTURBED INPUT ARRAY

<img src="./Report_Images/BitonicSort/mpi-speed-10.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/BitonicSort/mpi-speed-11.png" alt="speedup-mpi" width="400">
<img src="./Report_Images/BitonicSort/mpi-speed-12.png" alt="speedup-mpi" width="400">

### CUDA

In the CUDA speedup graphs, there doesn't seem to be any inherent benefit to paralleism. When we look at the main_function as a whole, we see that the speedup values tend to hover around 1.0. In fact, most of the lines tend to be below a value of 1, indicating that it is performing worse by being parallel. There also seems to be no pattern with the varying input size. We see that at the highest number of processors, the worst speedups are 2^16 elements and 2^28 elements, which are both the smallest and largest values respectively. It seems like as we increase processors, speedup for the communication is worst for the 2^16 element, but the computation is worst for 2^28 elements. Communication is hard to guage completely becuase there is quite a bit of fluctuation here, where computation speedup hovers around 1.0 and steadily decreases for most of the input sizes. This could also be because theres no real inter-process communication for CUDA like there is for MPI.
FIX ME!

#### RANDOM INPUT ARRAY

<img src="./Report_Images/BitonicSort/cuda-speed-1.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/BitonicSort/cuda-speed-2.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/BitonicSort/cuda-speed-3.png" alt="speedup-cuda" width="400">

#### SORTED INPUT ARRAY

<img src="./Report_Images/BitonicSort/cuda-speed-4.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/BitonicSort/cuda-speed-5.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/BitonicSort/cuda-speed-6.png" alt="speedup-cuda" width="400">

#### REVERSE SORTED INPUT ARRAY

<img src="./Report_Images/BitonicSort/cuda-speed-7.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/BitonicSort/cuda-speed-8.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/BitonicSort/cuda-speed-9.png" alt="speedup-cuda" width="400">

#### 1% PERTURBED INPUT ARRAY

<img src="./Report_Images/BitonicSort/cuda-speed-10.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/BitonicSort/cuda-speed-11.png" alt="speedup-cuda" width="400">
<img src="./Report_Images/BitonicSort/cuda-speed-12.png" alt="speedup-cuda" width="400">

# Comparison Analysis

For the comparison analysis, we analyzed an input size of 2^16 elements since selection sort was having issues at larger sizes.
Starting with CUDA, we noticed that selection sort was much slower for all aspects of total (main), communication, and computation time. To conduct a more thorough analysis, we looked at both the graphs with all of the sorting algorithms, as well as graphs that excluded selection sort. This was needed as selection sort's time was on such a large scale that it would wash out any discernable graph shape for the other three sorts making it impossible to tell which ones performed better. We can see the main and comp graphs are completely overtaken by the selection sort because it is not inherently a parallel algorithm and therefore performs the worst by far. A good comparison with selection sort is easiest to see in the communication graph since this was mainly cudaMemcpy and is relatively quick. Sample sort did not include a comm region, and therefore is not present on the graph.  

## CUDA - All

<img src="./Report_Images/Comparisons/comp3.png" alt="comp-cuda" width="400">
<img src="./Report_Images/Comparisons/comp2.png" alt="comp-cuda" width="400">
<img src="./Report_Images/Comparisons/comp1.png" alt="comp-cuda" width="400">

## CUDA - Bitonic, Merge, and Sample Sort

When looking at the main and computation graphs without selection sort, we see they are relatively constant, despite the rises and falls in the graphs, as the y-axis scale is to the order of tenths of a second. We see a spike for bitonic sort at 512 threads in the main region, which could likely be an outlier considering its other data points. When looking at just computation, we see that it remains fairly constant for all three graphs. From this analysis, we can conclude that for CUDA, the order from best to worst scaling sorting algorithms for our implementations is bitonic, sample, merge, and then selection sort.

<img src="./Report_Images/Comparisons/comp6.png" alt="comp-cuda" width="400">
<img src="./Report_Images/Comparisons/comp5.png" alt="comp-cuda" width="400">

## MPI

The graph of main shows mostly similar behavior across algorithms outside of merge sort, with the other three sorts hovering well below one for most number processes. Sample sort has a spike in the comm time at higher process counts, which explains the upward jump on the far right. Merge sort's graph shape for main isn't explained by either comp or comm, so other areas such as data initialization likely played a role in its poor performance. Looking at the computation graph, we see that selection sort is by far the worst at lower process counts, which indicates it is a poor-performing algorithm. However, it quickly converges towards 0 starting around 32 processes so it seems to take better advantage of more resources. It is difficult to see on the comp graph, but merge sort is the best performing of our sorting algorithm in MPI. When looking at the comm graph, we see most of the sorting algorithms slightly increase as more processes are added, except for sample sort which jumps up at 1024 processes, as previously discussed, due to the communication overhead of sending both the data and samples between processes.

<img src="./Report_Images/Comparisons/mpi1.png" alt="comp-mpi" width="400">
<img src="./Report_Images/Comparisons/mpi2.png" alt="comp-mpi" width="400">
<img src="./Report_Images/Comparisons/mpi3.png" alt="comp-mpi" width="400">
