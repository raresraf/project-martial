/**
 * @file gpu_hashtable.cu
 * @brief CUDA-based concurrent Hash Table with open addressing and linear probing.
 * @details Leverages GPU parallelism for batch key-value operations. Employs 
 * atomic Compare-And-Swap (CAS) for thread-safe state transitions in global memory.
 * 
 * Domain: HPC, Parallel Data Structures, CUDA.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "gpu_hashtable.hpp"

/**
 * @brief Device-side hash generator using bitwise mixing.
 * Logic: Implements a non-cryptographic hash based on bit-shifts and XORs 
 * to distribute integer keys across N buckets.
 * 
 * @param x The input key.
 * @param N The total number of buckets (table capacity).
 * @return long long The computed bucket index.
 */
__device__ long long getHash(int x, int N) {
    if (x < 0) x = -x;
    x = ((x >> 16) ^ x) * HASH_NO;
    x = ((x >> 16) ^ x) * HASH_NO;
    x = (x >> 16) ^ x;
    return (long long)(x % N);
}

/**
 * @brief CUDA Kernel for re-hashing existing entries during a table reshape.
 * Logic: Transfers key-value pairs from an old hash table (h1) to a new, 
 * larger one (h2) using linear probing.
 */
__global__ void reshape_hashT(hashTable h1, long siz1, hashTable h2, long siz2) {
    int vall, idx = blockIdx.x * blockDim.x + threadIdx.x, key1, key2;
    bool ok = false;
    
    // Boundary and Validity check: Ignore empty slots or out-of-bounds threads.
    if (idx >= siz1 || h1.pairs[idx].key == KEY_INVALID)
        return;

    key2 = h1.pairs[idx].key;
    vall = (int)getHash(key2, h2.size);

    /**
     * Block Logic: Linear probing for empty slots in the new table.
     * Synchronization: atomicCAS ensures only one thread claims an empty bucket.
     */
    for (int i = vall; i < siz2; i++) {
        key1 = atomicCAS(&h2.pairs[i].key, KEY_INVALID, key2);
        if (key1 == KEY_INVALID) {
            h2.pairs[i].value = h1.pairs[idx].value;
            ok = true;
            break;
        } 
    }
    
    // Wrap-around search if no slot was found in the primary range.
    if (!ok) {
        for (int i = 0; i < vall; i++) {
            key1 = atomicCAS(&h2.pairs[i].key, KEY_INVALID, key2);
            if (key1 == KEY_INVALID) {
                h2.pairs[i].value = h1.pairs[idx].value;
                break;
            }
        }
    }
}

/**
 * @brief CUDA Kernel for batch insertion of key-value pairs.
 * Logic: Each thread attempts to insert one pair from the input arrays.
 * Supports updating existing keys (if key matches input).
 */
__global__ void insert_hash(int k, int *keys, int *values, hashTable h, long siz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x, key, vall;
    
    if (idx >= k) return;
    
    vall = (int)getHash(keys[idx], (int)siz);

    /**
     * Block Logic: Linear probing with atomic state checking.
     * Invariant: Success occurs when a slot is either claimed (was INVALID) 
     * or found to already contain the target key (for updates).
     */
    for (int i = vall; i < siz; i++) {
        key = atomicCAS(&h.pairs[i].key, KEY_INVALID, keys[idx]);
        if (key == KEY_INVALID || key == keys[idx]) {
            h.pairs[i].value = values[idx];
            return;
        } 
    }
    for (int i = 0; i < vall; i++) {
        key = atomicCAS(&h.pairs[i].key, KEY_INVALID, keys[idx]);
        if (key == KEY_INVALID || key == keys[idx]) {
            h.pairs[i].value = values[idx];
            return;
        }
    }
}

/**
 * @brief CUDA Kernel for batch retrieval of values.
 * Logic: Linear probing search. In-place modification of 'values' array with results.
 */
__global__ void get_hash(int k, int *keys, int *values, hashTable h, long siz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x, vall;
    if (idx >= k) return;
    
    vall = (int)getHash(keys[idx], (int)siz);

    /**
     * Block Logic: Probes the table buckets sequentially starting from the hash index.
     */
    for (int i = vall; i < siz; i++) {
        if (h.pairs[i].key == keys[idx]) {
            values[idx] = h.pairs[i].value;
            return;
        } 
    }
    for (int i = 0; i < vall; i++) {
        if (h.pairs[i].key == keys[idx]) {
            values[idx] = h.pairs[i].value;
            return;
        }
    }
}

/**
 * @brief Constructor for the GpuHashTable class.
 * Memory Hierarchy: Allocates the pair structure array in Device Global Memory.
 */
GpuHashTable::GpuHashTable(int size) {
    hashT.size = size;
    cntPairs = 0;
    hashT.pairs = nullptr;
    cudaMalloc(&hashT.pairs, size * sizeof(pair));
    cudaMemset(hashT.pairs, 0, size * sizeof(pair));
}

GpuHashTable::~GpuHashTable() {
    cudaFree(hashT.pairs);
}

/**
 * @brief Dynamic capacity scaling for the hash table.
 * Logic: Creates a new table, launches the re-hashing kernel, and swaps pointers.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
    int num_blocks = (int)((hashT.size + THREADS_NO - 1) / THREADS_NO);
    hashTable newH;
    newH.size = numBucketsReshape;

    cudaMalloc(&newH.pairs, numBucketsReshape * sizeof(pair));
    cudaMemset(newH.pairs, 0, numBucketsReshape * sizeof(pair));
    
    // Execution: Parallel migration of elements to the new bucket space.
    reshape_hashT<<<num_blocks, THREADS_NO>>>(hashT, hashT.size, newH, newH.size);

    cudaDeviceSynchronize();
    cudaFree(hashT.pairs);
    hashT = newH;
}

/**
 * @brief High-level API for batch insertions.
 * Workflow: Threshold check -> Host to Device Copy -> Kernel Launch.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
    int *aKeys, *aValues;
    int num_blocks = (numKeys + THREADS_NO - 1) / THREADS_NO;
    long nr = cntPairs + numKeys;

    // Resizing Logic: Proactively expands the table if the load factor exceeds MAX.
    if ((float)nr / hashT.size >= LOADFACTOR_MAX) {
        reshape((int)(nr / LOADFACTOR_MIN));
    }

    cudaMalloc(&aKeys, numKeys * sizeof(int));
    cudaMalloc(&aValues, numKeys * sizeof(int));

    // Transfer: System RAM to GPU VRAM.
    cudaMemcpy(aKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(aValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    
    insert_hash<<<num_blocks, THREADS_NO>>>(numKeys, aKeys, aValues, hashT, hashT.size);

    cudaDeviceSynchronize();
    cudaFree(aKeys);
    cudaFree(aValues);
    cntPairs += numKeys;
    return true;
}

/**
 * @brief High-level API for batch value retrieval.
 * Workflow: HtoD Copy -> Kernel Launch -> DtoH Copy back to host.
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
    int *aKeys, *valls;
    int num_blocks = (numKeys + THREADS_NO - 1) / THREADS_NO;

    cudaMalloc(&aKeys, numKeys * sizeof(int));
    // Memory allocation for results (Unified Memory or Managed for convenience).
    cudaMallocManaged(&valls, numKeys * sizeof(int));

    cudaMemcpy(aKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    get_hash<<<num_blocks, THREADS_NO>>>(numKeys, aKeys, valls, hashT, hashT.size);
    
    cudaDeviceSynchronize();
    cudaFree(aKeys);
    return valls;
}

float GpuHashTable::loadFactor() {
    if (hashT.size == 0) return 0;
    return (float(cntPairs) / hashT.size);
}
