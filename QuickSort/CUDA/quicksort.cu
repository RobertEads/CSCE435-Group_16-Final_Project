#include <stdio.h>
#include <cstdlib>
#include <string>

__device__ void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

__device__ int partition(int* arr, int low, int high) {
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

__global__ void quicksort(int* arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        if (pi - low < high - pi) {
            quicksort<<<1, 1>>>(arr, low, pi - 1);
            quicksort<<<1, 1>>>(arr, pi + 1, high);
        } else {
            quicksort<<<1, 1>>>(arr, pi + 1, high);
            quicksort<<<1, 1>>>(arr, low, pi - 1);
        }
    }
}

/* Verification */
// CUDA kernel to check if the array is sorted
__global__ void checkArraySorted(int *array, bool *isSorted, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1)
    {
        isSorted[idx] = (array[idx] <= array[idx + 1]);
    }
}

int main() {
    int n = 10;
    int h_arr[n] = {4, 1, 7, 3, 9, 8, 2, 5, 6, 0};

    int* d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    quicksort<<<1, 1>>>(d_arr, 0, n - 1);

    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    return 0;
}