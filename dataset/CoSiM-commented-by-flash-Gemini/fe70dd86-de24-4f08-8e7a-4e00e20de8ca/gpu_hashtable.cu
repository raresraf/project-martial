
/**
 * @file gpu_hashtable.cu
 * @brief Thread-safe GPU Hash Table using atomic compare-and-swap and circular probing.
 * 
 * Algorithm: Open addressing with linear probing and circular wrap-around.
 * Memory Model: Device global memory for the entry array.
 * Synchronization: atomicCAS for race-free mapping and value updates.
 * Domain: HPC, Parallel Data Structures.
 */

#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#include "gpu_hashtable.hpp"

#define minim 0.82
#define maxim 0.93
#define num_threads 256


/**
 * @brief CUDA kernel for migrating data during table expansion.
 * 
 * Functional Utility: Re-hashes existing valid entries and places them into 
 * a new, larger memory buffer.
 */
__global__ void do_reshape(hashtable oldHash, hashtable newHash) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Pre-condition: Thread index must be within source table bounds.
	if (index < oldHash.size &&
		oldHash.table[index].key != KEY_INVALID) {
		int key = oldHash.table[index].key;
		int hash = Hash(key, newHash.size);
		int size = newHash.size, i;
		
		/**
		 * Block Logic: Forward probe sequence for re-insertion.
		 * Invariant: Searches for the first available slot in the destination buffer.
		 */
		for (i = hash; i < size; i++){
			if (atomicCAS(&newHash.table[i].key, KEY_INVALID, key) == KEY_INVALID) {
				newHash.table[i].value = oldHash.table[index].value;
				return;
			}
		}
		
		/**
		 * Block Logic: Wrap-around probe sequence for re-insertion.
		 */
		for (i = 0; i < hash; i++){
			if (atomicCAS(&newHash.table[i].key, KEY_INVALID, key) == KEY_INVALID) {
				newHash.table[i].value = oldHash.table[index].value;


				return;
			}
		}
	}
}


/**
 * @brief CUDA kernel for parallel entry insertion.
 * 
 * Logic: Uses linear probing to find an empty slot or update an existing key. 
 * Synchronization is achieved through hardware-level atomic operations.
 */
__global__ void insert(int *keys, int *values, hashtable hashmap, int count) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Pre-condition: Kernel handles exactly 'count' input elements.
	if (index < count) {
		int key = keys[index];
		int hash = Hash(key, hashmap.size);
		int size = hashmap.size, i;
		
		/**
		 * Block Logic: Two-pass linear search and claim.
		 */
		for(i = hash; i < size; i++){
			atomicCAS(&hashmap.table[i].key, KEY_INVALID, key);
			if (hashmap.table[i].key == key) {
				hashmap.table[i].value = values[index];
				return;
			}
		}
		for(i = 0; i < hash; i++){
			atomicCAS(&hashmap.table[i].key, KEY_INVALID, key);
			if (hashmap.table[i].key == key) {
				hashmap.table[i].value = values[index];


				return;
			}
		}
	}
	
}


/**
 * @brief CUDA kernel for parallel value lookup.
 */
__global__ void get(int *keys, int *values, hashtable hashmap, int count) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < count) {
		int key = keys[index];
		int hash = Hash(keys[index], hashmap.size);
		int size = hashmap.size, i;
		
		/**
		 * Block Logic: Linear search traversal.
		 */
		for (i = hash; i < size; i++){
			if (hashmap.table[i].key == key) {
				values[index] = hashmap.table[i].value;
				return;
			}
		}
		for (i = 0; i < hash; i++){
			if (hashmap.table[i].key == key) {
				values[index] = hashmap.table[i].value;
				return;
			}
		}
	}
}


/**
 * @brief Constructor: Allocates GPU memory and initializes table metadata.
 */
GpuHashTable::GpuHashTable(int size) {
	h.real_size = 0;
	h.size = size;
	// Memory Hierarchy: Global memory (VRAM) allocation.
	cudaMalloc(&h.table, size * sizeof(assoc));
	cudaMemset(h.table, 0, size * sizeof(assoc));
}


/**
 * @brief Destructor: Releases device resident memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(h.table);
}


/**
 * @brief Resizes the hash table to a new capacity.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hashtable newh;
	newh.size = numBucketsReshape;
	newh.real_size = h.real_size;
	cudaMalloc(&newh.table, numBucketsReshape * sizeof(assoc));
	cudaMemset(newh.table, 0, numBucketsReshape * sizeof(assoc));
	
	// Synchronization: Ensures data migration is complete before memory reclamation.
	if (h.size / num_threads * num_threads == h.size) {
		do_reshape<<<h.size / num_threads, num_threads>>>(h, newh);
	} else {


		do_reshape<<<h.size / num_threads + 1, num_threads>>>(h, newh);
	}
	cudaDeviceSynchronize();
	cudaFree(h.table);
	h = newh;
}


/**
 * @brief Orchestrates batch insertion from Host to Device.
 * 
 * Optimization: Automatically resizes the table if the load factor exceeds thresholds.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *deviceKeys, *deviceValues;
	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	cudaMalloc(&deviceValues, numKeys * sizeof(int));
	h.real_size += numKeys;

	
	// Adaptive Scaling: Maintain probe efficiency by expanding the bucket array.
	if (float(h.real_size) >= (float)(h.size * (float)maxim))
		reshape(int((h.real_size) / minim));

	
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	if (numKeys / num_threads * num_threads == numKeys) {
		insert<<<numKeys / num_threads, num_threads>>>
			(deviceKeys, deviceValues, h, numKeys);
	} else {
		insert<<<numKeys / num_threads + 1, num_threads>>>
			(deviceKeys, deviceValues, h, numKeys);
	}
	
	cudaDeviceSynchronize();
	cudaFree(deviceKeys);
	cudaFree(deviceValues);
	return false;
}


/**
 * @brief Orchestrates batch retrieval into managed memory results.
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
	int *deviceKeys, *values;
	// Memory Hierarchy: Unified memory simplifies result access on the Host.
	cudaMallocManaged(&deviceKeys, numKeys * sizeof(int));
	cudaMallocManaged(&values, numKeys * sizeof(int));

	
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	if (numKeys / num_threads * num_threads == numKeys) {
		get<<<numKeys / num_threads, num_threads>>>
			(deviceKeys, values, h, numKeys);
	} else {
		get<<<numKeys / num_threads + 1, num_threads>>>
			(deviceKeys, values, h, numKeys);
	}
	cudaDeviceSynchronize();
	cudaFree(deviceKeys);
	return values;
}


/**
 * @brief Returns current occupancy ratio.
 */
float GpuHashTable::loadFactor() {
	if (h.size > 0)
		return (float(h.real_size) / h.size);
	return 0.f;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

#include <iostream>
#include <vector>

#define KEY_INVALID 0

#define A 518509
#define B 6904869625999

#define DIE(assertion, call_description) \
    do {    \
        if (assertion) {    \
        fprintf(stderr, "(%s, %d): ",    \
        __FILE__, __LINE__);    \
        perror(call_description);    \
        exit(errno);    \
    }    \
} while (0)


/**
 * @brief Prime-multiplicative hash for device index generation.
 */
__device__ int Hash(int data, int limit) {
	return ((long long) abs(data) * A) % B % limit;
}

/**
 * @brief Reference primes for hashing distribution.
 */
const std::size_t primeList[] =
{
		2llu, 3llu, 5llu, 7llu, 11llu, 13llu, 17llu, 23llu, 29llu, 37llu, 47llu,
		59llu, 73llu, 97llu, 127llu, 151llu, 197llu, 251llu, 313llu, 397llu,
		499llu, 631llu, 797llu, 1009llu, 1259llu, 1597llu, 2011llu, 2539llu,
...
		11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu
};


/**
 * @brief Hash function variants for dispersion tests.
 */
int hash1(int data, int limit) {
	return ((long) abs(data) * primeList[64]) % primeList[90] % limit;
}

int hash2(int data, int limit) {
	return ((long) abs(data) * primeList[67]) % primeList[91] % limit;
}

int hash3(int data, int limit) {
	return ((long) abs(data) * primeList[70]) % primeList[93] % limit;
}


/**
 * @struct assoc
 * @brief Key-value mapping unit for GPU storage.
 */
struct assoc {
	int key;
	int value;
};


/**
 * @struct hashtable
 * @brief Device metadata and buffer pointer.
 */
struct hashtable {
	int size;
	int real_size;
	assoc *table;
};


/**
 * @class GpuHashTable
 * @brief Controller for managing a concurrent GPU resident map.
 */
class GpuHashTable {
	hashtable h;
public:
	GpuHashTable(int size);

	void reshape(int sizeReshape);

	bool insertBatch(int *keys, int *values, int numKeys);

	int *getBatch(int *key, int numItems);

	float loadFactor();

	void occupancy();

	void print(std::string info);

	~GpuHashTable();
};

#endif
