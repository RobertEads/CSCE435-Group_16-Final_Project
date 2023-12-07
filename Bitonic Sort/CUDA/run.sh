#!/bin/bash

elems=(65536 262144 1048576 4194304 16777216 67108864 268435456)
# elems=(16 64 256 1024 4096 16384)
threads=(64 128 256 512 1024)
inps=(0 1 2 3)


for elem in "${elems[@]}"; do
    for thread in "${threads[@]}"; do
        for inp in "${inps[@]}"; do
            sbatch CUDA.grace_job $inp $thread $elem
        done
    done
done