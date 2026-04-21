/**
 * @file gpu_hashtable.cu
 * @brief High-performance GPU-accelerated Hash Table implementation using CUDA.
 * 
 * Architectural Intent: Provides a massively parallel key-value store optimized for batch processing. 
 * Designed to reside entirely in GPU VRAM to minimize host-device data transfers during hot paths.
 * 
 * Domain-Awareness: Uses CUDA atomic operations to ensure thread-safe concurrency during 
 * collision resolution. Implements open addressing with linear probing.
 * 
 * Memory Strategy: Manages global GPU memory for the primary hash array. Utilizes 
 * atomicCompareAndSwap (atomicCAS) to orchestrate lock-free bucket acquisition across 
 * thousands of concurrent threads.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cuda.h>

#include "gpu_hashtable.hpp"


/**
 * @brief Constructs a GpuHashTable and initializes GPU memory.
 * @param size Initial bucket capacity.
 */
GpuHashTable::GpuHashTable(int size)
{
	cudaError_t error;

	// Resource Management: Allocates contiguous physical memory on the compute device.
	error = cudaMalloc(&(GpuHashTable::hashtable), size * sizeof(hT)); 
	DIE(error != cudaSuccess || GpuHashTable::hashtable == NULL, "cudaMalloc hashtable error");
	// Initialization: Zeroes out the table to represent an empty state (key=0).
	error = cudaMemset(GpuHashTable::hashtable, 0, size * sizeof(hT));
	DIE(error != cudaSuccess, "cudaMemset hashtable error");

	
	GpuHashTable::currentTableSize = 0;
	GpuHashTable::tableSize = size;
}



GpuHashTable::~GpuHashTable()
{
	cudaError_t error;
	error = cudaFree(GpuHashTable::hashtable);
	DIE(error != cudaSuccess, "cudaFree hashtable error");

	GpuHashTable::currentTableSize = 0;
	GpuHashTable::tableSize = 0;
}


/**
 * @brief CUDA kernel for compacting the hash table before a resize operation.
 * Functional Utility: Identifies non-empty buckets and extracts their data into 
 * dense arrays using atomic indexing.
 */
__global__ void copyForReshape(hT *hashtable, int tableSize,
								int *device_keys, int *device_values,
								int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	/**
	 * Block Logic: Parallel data extraction.
	 * Invariant: Threads up to 'tableSize' participate. Counter ensures no write-collisions in destination.
	 */
	if(idx < tableSize) {
		if (hashtable[idx].key != 0) {
			// Synchronization: Atomically claims a destination index to preserve uniqueness.
			int index = atomicAdd(counter, 1);

			device_keys[index] = hashtable[idx].key;
			device_values[index] = hashtable[idx].value;
		}
	}
}



/**
 * @brief Dynamically resizes the hash table to maintain a healthy load factor.
 * Algorithm: Full Rehash. 
 * 1. Allocate new GPU buffer. 
 * 2. Parallel compaction of old data to host. 
 * 3. Atomic migration to new table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t error;
	hT *newHashTable = NULL;
	int new_size = 1.2f * numBucketsReshape;

	error = cudaMalloc(&newHashTable, new_size * sizeof(hT)); 
	DIE(error != cudaSuccess || newHashTable == NULL, "cudaMalloc new hashtable error");
	error = cudaMemset(newHashTable, 0, new_size * sizeof(hT));
	DIE(error != cudaSuccess, "cudaMemset new hashtable error");


	if(GpuHashTable::currentTableSize != 0) {
		const size_t block_size = 1024;
		size_t blocks_no = (GpuHashTable::tableSize + block_size - 1) / block_size;
 
		int *device_keys = NULL;
		int *counter = NULL;
		int *device_values = NULL;

		error = cudaMalloc(&device_keys, GpuHashTable::currentTableSize * sizeof(int)); 
		error = cudaMalloc(&device_values, GpuHashTable::currentTableSize * sizeof(int)); 
		error = cudaMalloc(&counter, sizeof(int)); 
		error = cudaMemset(counter, 0, sizeof(int));

		int *host_keys = (int *)malloc(GpuHashTable::currentTableSize * sizeof(int));
		int *host_values = (int *)malloc(GpuHashTable::currentTableSize * sizeof(int));

		copyForReshape<<<blocks_no, block_size>>>(GpuHashTable::hashtable, 
												GpuHashTable::tableSize,
												device_keys, device_values,
												counter);
		cudaDeviceSynchronize();

		error = cudaMemcpy(host_keys, device_keys, GpuHashTable::currentTableSize * sizeof(int), cudaMemcpyDeviceToHost);
		error = cudaMemcpy(host_values, device_values, GpuHashTable::currentTableSize * sizeof(int), cudaMemcpyDeviceToHost);

		GpuHashTable::~GpuHashTable();

		GpuHashTable::tableSize = new_size;
		GpuHashTable::hashtable = newHashTable;
		GpuHashTable::currentTableSize = 0;

		int numKeys = 0;
		error = cudaMemcpy(&numKeys, counter, sizeof(int), cudaMemcpyDeviceToHost); 
		
		insertBatch(host_keys, host_values, numKeys);

		cudaFree(device_keys); cudaFree(device_values); cudaFree(counter);
		free(host_keys); free(host_values);
		return;
	}

	GpuHashTable::~GpuHashTable();
	GpuHashTable::tableSize = new_size;
	GpuHashTable::hashtable = newHashTable;
}


/**
 * @brief CUDA kernel to compute multiplicative hashes for a set of keys.
 */
__global__ void getHashCode(int *keys, int *hashcodes, int numkeys, int tablesize)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < numkeys)
		hashcodes[idx] = (((long)keys[idx]) * 1402982055436147llu) % 452517535812813007llu % tablesize;
}



/**
 * @brief Parallel insertion kernel with linear probing and atomic Compare-And-Swap.
 * Logic: Implements lock-free synchronization for concurrent table updates. 
 * If a thread encounters a collision (claimed slot by another key), it probes the next 
 * slot in a cyclical pattern until insertion succeeds.
 */
__global__ void insertKeysandValues(int *hashcodes, int *keys, int *values,
									int numKeys, hT *hashtable,
									int *currentTableSize, int tablesize,
									int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numKeys) {
		if (keys[idx] == 0) return;

		// Synchronization: Ensures atomic ownership of a hash bucket.
		int key = atomicCAS(&hashtable[hashcodes[idx]].key, 0, keys[idx]); 
		
		/**
		 * Block Logic: Slot ownership evaluation.
		 */
		if (key == 0) { // Success: First-time insertion.
			hashtable[hashcodes[idx]].value = values[idx];
			keys[idx] = 0;
			atomicAdd(currentTableSize, 1);
			atomicAdd(counter, 1);
		} else if (key == keys[idx]) { // Success: Update existing key.
			hashtable[hashcodes[idx]].value = values[idx];
			atomicAdd(counter, 1);
			keys[idx] = 0;
		} else { // Collision: Structural retry.
			hashcodes[idx] = (hashcodes[idx] + 1) % tablesize;
		}
	}
}



/**
 * @brief Dispatches parallel insertion of multiple key-value pairs.
 * Functional Utility: Orchestrates the GPU compute grid and manages host-device data sync.
 * Retries: Iteratively re-launches the kernel until all collisions are resolved.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	// ... (Buffer allocation and data transfer) ...

	/**
	 * Block Logic: Collision resolution loop.
	 * Invariant: Re-triggers execution until the global 'counter' matches 'numKeys'.
	 */
	while(1) {
		insertKeysandValues<<<blocks_no, block_size>>>(hashcodes, device_keys, 
													device_values, numKeys,
													GpuHashTable::hashtable,
													device_current,
													GpuHashTable::tableSize,
													device_counter);
		cudaDeviceSynchronize();
		// ... (Check completion status) ...
		if(host_counter == numKeys) break;
	}
	
	// ... (Cleanup) ...
	return true;
}

// ... (Rest of implementation) ...
