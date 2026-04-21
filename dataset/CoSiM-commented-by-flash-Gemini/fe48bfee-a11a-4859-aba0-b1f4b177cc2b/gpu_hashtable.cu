
/**
 * @file gpu_hashtable.cu
 * @brief GPGPU Hash Table with linear probing and duplicate tracking.
 * 
 * Algorithm: Open addressing with linear probing and manual circularity.
 * Memory Model: Global memory for entry buffers, Managed memory for result retrieval.
 * Synchronization: Uses atomicCAS for thread-safe mapping and concurrency control.
 * Domain: HPC, Parallel Data Structures.
 */

#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#include "gpu_hashtable.hpp"


/**
 * @brief Multiplicative hash function for device-side index calculation.
 */
__device__ int myHash(int data, int limit) {
	return ((long)abs(data) * 2654435761llu) % 4294967296llu % limit;
}


/**
 * @brief CUDA kernel for parallel entry insertion.
 * 
 * Functional Utility: Claims empty slots or updates existing entries using 
 * atomic compare-and-swap. Implements a two-pass linear probe to handle wrap-around.
 */
__global__ void kernel_insert(int *keys, int *values, int limitBound, hashtable hashmap, int *duplicate) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	// Pre-condition: Thread index must be within the batch input range.
	if (i >= limitBound) {
		return;
	}

	int extractKey = keys[i], size = hashmap.size, oldK, check = 0;
	int hash = myHash(extractKey, size);

	/**
	 * Block Logic: Forward probe sequence.
	 * Invariant: Probes from the initial hash index to the end of the global buffer.
	 */
	for (int k = hash; k < size; ++k) {
		oldK = atomicCAS(&hashmap.list[k].key, KEY_INVALID, extractKey);
		if (oldK == KEY_INVALID || oldK == extractKey) {
			hashmap.list[k].value  = values[i];
			check = 1;
			if (oldK == extractKey)
				*duplicate++; // Note: Pointer increment (potential original bug retained).
			break;
		}
	}

	/**
	 * Block Logic: Wrap-around probe sequence.
	 * Invariant: Probes from the start of the table back to the original hash index.
	 */
	if (check == 0) {
		for (int k = 0; k < hash; ++k) {
			oldK = atomicCAS(&hashmap.list[k].key, KEY_INVALID, extractKey);
			if (oldK == KEY_INVALID || oldK == extractKey) {
				hashmap.list[k].value  = values[i];
				if (oldK == extractKey)
					*duplicate++;
				return;
			}
		}
	}
 
	return;
}


/**
 * @brief CUDA kernel for parallel entry retrieval.
 * 
 * Logic: Performs a two-pass linear search to locate keys and extract values.
 */
__global__ void kernel_get(int *keys, int *values, int limitBound, hashtable hashmap) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= limitBound) {
		return;
	}

	int extractKey = keys[i], size = hashmap.size, check = 0;
	int hash = myHash(extractKey, size);

	/**
	 * Block Logic: Two-phase linear search sweep.
	 */
	for (int k = hash; k < size; ++k) {
		if (hashmap.list[k].key == extractKey) {
			values[i] = hashmap.list[k].value;
			check = 1;
			return;
		}
	}

	if (check == 0) {
		for (int k = 0; k < hash; ++k) {
			if (hashmap.list[k].key == extractKey) {
				values[i] = hashmap.list[k].value;
				return;
			}
		}
	}

}


/**
 * @brief CUDA kernel for re-hashing entries during table resizing.
 */
__global__ void kernel_reshape(hashtable hashmap, hashtable newHashmap) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= hashmap.size) {
		return;
	}

	// Logic: Migrates existing valid mappings to the new address space.
	if (hashmap.list[i].key != KEY_INVALID) {
		int extractKey = hashmap.list[i].key, isInserted = 0, size = newHashmap.size;
		int hash = myHash(extractKey, size);

		for (int k = hash; k < size && !isInserted; ++k) {
			int oldK = atomicCAS(&newHashmap.list[k].key, KEY_INVALID, extractKey);
			if (oldK == KEY_INVALID) {
				newHashmap.list[k].value  = hashmap.list[i].value;
				isInserted = 1;
				break;
			}
		}

		if (isInserted == 0) {
			for (int k = 0; k < hash && !isInserted; ++k) {
				int oldK = atomicCAS(&newHashmap.list[k].key, KEY_INVALID, extractKey);
				if (oldK == KEY_INVALID) {
					newHashmap.list[k].value  = hashmap.list[i].value;
					isInserted = 1;
					break;
				}
			}
		}
	}

}


/**
 * @brief Constructor: Initializes hash map metadata and allocates device memory.
 */
GpuHashTable::GpuHashTable(int size) {
	length = 0;
	hashmap.size = size;
	hashmap.list = nullptr;
	
	// Memory Hierarchy: Global memory (VRAM) allocation for entry array.
	if (cudaMalloc(&hashmap.list, size * sizeof(entry)) != cudaSuccess) {
		std::cerr << "Memory allocation error\n";


		return;
	}
	cudaMemset(hashmap.list, 0, size * sizeof(entry));  
}


/**
 * @brief Destructor: Frees device resident resources.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashmap.list);
}


/**
 * @brief Dynamically resizes the hash table and re-inserts current mappings.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hashtable newHashmap;
	newHashmap.size = numBucketsReshape;
	if (cudaMalloc(&newHashmap.list, numBucketsReshape * sizeof(entry)) != cudaSuccess) {
		std::cerr << "Memory allocation error\n";
		return;
	}
	cudaMemset(newHashmap.list, 0, numBucketsReshape * sizeof(entry));

	int numBlocks = ((hashmap.size % 1024 == 0) ? (hashmap.size / 1024) : (hashmap.size / 1024 + 1));
	// Synchronization: Kernel execution blocks host until migration completes.
	kernel_reshape<<<numBlocks, 1024>>>(hashmap, newHashmap);

	cudaDeviceSynchronize();
	cudaFree(hashmap.list);
	hashmap = newHashmap;
}


/**
 * @brief Orchestrates batch parallel insertion and adaptive resizing.
 * 
 * Logic: Transfers batch to GPU, resizes if capacity is threatened, 
 * and reconciles host-side occupancy after execution.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *deviceKeys, *deviceValues, *duplicate;
	int dupl;
	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	cudaMalloc(&deviceValues, numKeys * sizeof(int));
	cudaMalloc(&duplicate, sizeof(int));
	if (!deviceKeys || !deviceValues) {
		std::cerr << "Memory allocation error\n";
		return false;
	}
	
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemset(duplicate, 0, sizeof(int));

	 // Optimization: Automatic resizing to maintain O(1) performance.
	 if (float(length + numKeys) >= hashmap.size) {                                             
                reshape(int(((length + numKeys) / 0.83)));                                                                                                                                                                  }
	int numBlocks = ((numKeys % 1024 == 0) ? (numKeys / 1024) : (numKeys / 1024 + 1));
	
	kernel_insert<<<numBlocks, 1024>>>(deviceKeys, deviceValues, numKeys, hashmap, duplicate);
	
	// Synchronization: Captures update count to maintain accurate occupancy tracking.
	cudaMemcpy(&dupl, duplicate, sizeof(int), cudaMemcpyDeviceToHost);
	int dup = dupl;

	cudaDeviceSynchronize();
	length += numKeys;
	length -= dup;

	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	return true;
}


/**
 * @brief Orchestrates batch parallel retrieval into managed memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *deviceKeys, *values;
	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	// Memory Hierarchy: Managed memory (Unified) simplifies host access to GPU search results.
	cudaMallocManaged(&values, numKeys * sizeof(int));

	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int numBlocks = ((numKeys % 1024 == 0) ? (numKeys / 1024) : (numKeys / 1024 + 1));
	kernel_get<<<numBlocks, 1024>>>(deviceKeys, values, numKeys, hashmap);

	cudaDeviceSynchronize();

	cudaFree(deviceKeys);
	return values;
}


/**
 * @brief Returns current occupancy density.
 */
float GpuHashTable::loadFactor() {
	
	if (hashmap.size == 0) {
		return 0.f;
	}
	return (float(length) / hashmap.size);
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
 * @brief Prime numbers for hashing dispersion.
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
 * @brief Hash function variations for distribution testing.
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
 * @struct entry
 * @brief Internal storage unit for a key-value pair on the GPU.
 */
typedef struct entry {
	int key;
	int value;
} entry;


/**
 * @struct hashtable
 * @brief Metadata and storage reference for the GPU-resident map.
 */
typedef struct hashtable {
	int size;
	entry *list;
} hashtable;


/**
 * @class GpuHashTable
 * @brief Host manager for the GPU hash table lifecycle.
 */
class GpuHashTable
{	
	int length;
	hashtable hashmap;
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
