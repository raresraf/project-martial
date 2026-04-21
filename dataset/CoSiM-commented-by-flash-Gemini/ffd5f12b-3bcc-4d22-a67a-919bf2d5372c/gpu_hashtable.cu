
/**
 * @file gpu_hashtable.cu
 * @brief High-performance GPU Hash Table with linear probing and load-aware resizing.
 * 
 * Algorithm: Open addressing with linear probing and prime-based multiplicative hashing.
 * Memory Model: Global memory for entry storage, host-managed batch transfers.
 * Synchronization: Uses atomicCAS for thread-safe concurrent insertion and key updates.
 * Domain: HPC, Parallel Data Structures.
 */

#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#include "gpu_hashtable.hpp"


/**
 * @brief CUDA kernel for parallel entry insertion.
 * 
 * Functional Utility: Maps keys to buckets and resolves collisions via circular 
 * linear probing. Atomically claims empty slots to prevent data loss.
 * 
 * @param info Device pointer for tracking successful new insertions.
 */
__global__ void kernelInsert(Entry *entries, int size, int *keys, 
	int *values, int N, int *info) {

	int hash, oldKey, i;

	int index = threadIdx.x + blockDim.x * blockIdx.x;

	// Pre-condition: Thread index must be within the batch input range.
	if (index >= N)
		return;
	
	hash = hashKey(keys[index], size);
	i = hash;

	/**
	 * Block Logic: Linear probing search and claim loop.
	 * Invariant: Terminates when an empty slot is claimed, the key is updated, 
	 * or the entire table is searched.
	 */
	do {
		oldKey = atomicCAS(&entries[i].key, 0, keys[index]);
		
		if (oldKey == 0) {
			// Logic: Found empty slot, increment occupancy and set value.
			atomicAdd(info, 1);
			entries[i].value = values[index];
			return;
		}
		
		if (oldKey == keys[index]) {
			// Logic: Key already exists, perform in-place value update.
			entries[i].value = values[index];
			return;
		}
		// Move to next slot in circular buffer.
		i = (i + 1) % size;
	} while (i != hash);
}


/**
 * @brief CUDA kernel for parallel entry retrieval.
 */
__global__ void kernelGet(Entry *entries, int size, int *keys, 
	int *values, int N) {

	int hash, i;

	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= N)
		return;
	
	hash = hashKey(keys[index], size);
	i = hash;

	/**
	 * Block Logic: Linear search traversal.
	 */
	do {
		if (entries[i].key == keys[index]) {
			values[index] = entries[i].value;
			return;
		}
		i = (i + 1) % size;
	} while (i != hash);
}


/**
 * @brief CUDA kernel for re-hashing data during table expansion.
 */
__global__ void kernelRehash(Entry *newEntries, int newSize, 
	Entry *oldEntries, int oldSize) {

	int hash, oldKey, i;

	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= oldSize)
		return;

	// Optimization: Skip inactive buckets.
	if (oldEntries[index].key == 0)
		return;
	
	hash = hashKey(oldEntries[index].key, newSize);
	i = hash;

	/**
	 * Block Logic: Migration re-insertion loop.
	 */
	do {
		oldKey = atomicCAS(&newEntries[i].key, 0, oldEntries[index].key);
		
		if (oldKey == 0) {
			newEntries[i].value = oldEntries[index].value;
			return;
		}
		i = (i + 1) % newSize;
	} while (i != hash);
}


/**
 * @brief Constructor: Prepares the GPU resident hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	cudaError_t err;

	this->size = size;
	this->occupied = 0;
	this->entries = NULL;

	// Memory Hierarchy: VRAM allocation for bucket array.
	err = cudaMalloc((void **) &this->entries, size * sizeof(Entry));

	if (err != cudaSuccess) {
		std::cout << "[INIT] Couldn't allocate memory\n";
		return;
	}

	cudaMemset((void *) this->entries, 0, size * sizeof(Entry));
}


/**
 * @brief Destructor: Frees allocated device memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(this->entries);
}


/**
 * @brief Resizes the hash table to accommodate more entries.
 * 
 * Algorithm: Spawns a re-hashing kernel to migrate data to a larger buffer.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int totalSize;
	cudaError_t err;
	Entry *newEntries;
	size_t numBlocks = this->size / BLOCK_SIZE;

	// Optimization: Adds a buffer margin to keep load factor stable.
	numBucketsReshape *= SCALE_FACTOR;

	if (this->size % BLOCK_SIZE) {
		numBlocks++;
	}

	totalSize = numBucketsReshape * sizeof(Entry);
	
	err = cudaMalloc((void **) &newEntries, totalSize);

	if (err != cudaSuccess) {
		std::cout << "[RESHAPE] Couldn't allocate memory\n";
		return;
	}

	cudaMemset((void *) newEntries, 0, totalSize);

	// Synchronization: Kernel synchronization point ensuring safe buffer swap.
	kernelRehash<<<numBlocks, BLOCK_SIZE>>>(newEntries, numBucketsReshape,
		this->entries, this->size);
	cudaDeviceSynchronize();

	cudaFree(this->entries);
	this->entries = newEntries;
	this->size = numBucketsReshape;
}


/**
 * @brief Performs host-initiated batch insertion.
 * 
 * Logic: Manages host-to-device transfers and triggers proactive resizing 
 * to maintain high probe performance.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *deviceKeys = NULL;
	int *deviceValues = NULL;
	int *info = NULL;
	int copy;
	float load;
	size_t totalSize = numKeys * sizeof(int);
	size_t numBlocks = numKeys / BLOCK_SIZE;
	cudaError_t err1, err2, err3;

	if (numKeys % BLOCK_SIZE) {
		numBlocks++;
	}

	err1 = cudaMalloc((void **) &deviceKeys, totalSize);
	err2 = cudaMalloc((void **) &deviceValues, totalSize);
	err3 = cudaMalloc((void **) &info, sizeof(int));

	if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
		std::cout << "[INSERT] Couldn't allocate memory\n";
		return false;
	}

	cudaMemset((void *) info, 0, sizeof(int));

	cudaMemcpy(deviceKeys, keys, totalSize, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, totalSize, cudaMemcpyHostToDevice);
	
	// Optimization: Expansion check prior to execution.
	if (this->size - this->occupied < numKeys) {
		reshape(this->size + numKeys);
	}

	kernelInsert<<<numBlocks, BLOCK_SIZE>>>(this->entries, this->size,
		deviceKeys, deviceValues, numKeys, info);

	cudaDeviceSynchronize();
	// Synchronization: Aggregates update status back to the host.
	cudaMemcpy(&copy, info, sizeof(int), cudaMemcpyDeviceToHost);
	this->occupied += copy;

	// Invariant: Maintains a target load factor range for probe consistency.
	load = loadFactor();
	if (load < MIN_LOAD_FACTOR) {
		// reshape(load * this->size); // Optional shrinking logic.
	}

	cudaFree(info);
	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	return true;
}


/**
 * @brief Performs host-initiated batch retrieval.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *deviceKeys = NULL;
	int *deviceValues = NULL;
	int *hostValues = NULL;
	size_t totalSize = numKeys * sizeof(int);
	size_t numBlocks = numKeys / BLOCK_SIZE;
	cudaError_t err1, err2;

	if (numKeys % BLOCK_SIZE) {
		numBlocks++;
	}

	hostValues = (int *)malloc(totalSize);
	err1 = cudaMalloc((void **) &deviceKeys, totalSize);
	err2 = cudaMalloc((void **) &deviceValues, totalSize);

	cudaMemset((void *) deviceValues, 0, totalSize);

	if (err1 != cudaSuccess || err2 != cudaSuccess || hostValues == NULL) {
		std::cout << "[GET] Couldn't allocate memory\n";
		return NULL;
	}

	cudaMemcpy(deviceKeys, keys, totalSize, cudaMemcpyHostToDevice);
	
	kernelGet<<<numBlocks, BLOCK_SIZE>>>(this->entries, this->size,
		deviceKeys, deviceValues, numKeys);

	cudaDeviceSynchronize();
	
	cudaMemcpy(hostValues, deviceValues, totalSize, cudaMemcpyDeviceToHost);

	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	return hostValues;
}


/**
 * @brief Returns the ratio of occupied buckets to total capacity.
 */
float GpuHashTable::loadFactor() {
	return (float)this->occupied / this->size; 
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0
#define	SCALE_FACTOR	1.05
#define MIN_LOAD_FACTOR	0.8
#define BLOCK_SIZE		1024
#define PRIME_1			653267llu
#define PRIME_2			10703903591llu

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)
	
/**
 * @brief Hashing primes for dispersion.
 */
const size_t primeList[] =
{
	2llu, 3llu, 5llu, 7llu, 11llu, 13llu, 17llu, 23llu, 29llu, 37llu, 47llu,
	59llu, 73llu, 97llu, 127llu, 151llu, 197llu, 251llu, 313llu, 397llu,
	499llu, 631llu, 797llu, 1009llu, 1259llu, 1597llu, 2011llu, 2539llu,
...
	5746614499066534157llu, 7240280573005008577llu, 9122181901073924329llu,
	11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu
};


/**
 * @brief Simple hash variations for dispersion tests.
 */
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}


/**
 * @brief Prime-multiplicative hash for device index generation.
 */
__device__ int hashKey(int data, int limit) {
	return ((long)abs(data) * PRIME_1) % PRIME_2 % limit;
}


/**
 * @struct entry
 * @brief Unit of storage for a key-value mapping on the GPU.
 */
typedef struct entry {
	int key;
	int value;
} Entry;


/**
 * @class GpuHashTable
 * @brief Host manager for the GPU resident hash table state.
 */
class GpuHashTable
{
	Entry *entries;
	int size;
	int occupied;

	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		
		float loadFactor();
		void occupancy();
		void print(string info);
	
		~GpuHashTable();
};

#endif
