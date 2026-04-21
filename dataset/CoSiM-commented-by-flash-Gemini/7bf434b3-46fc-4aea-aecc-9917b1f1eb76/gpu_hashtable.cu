
/**
 * @file gpu_hashtable.cu
 * @brief High-throughput GPU Hash Table with dynamic load-factor management.
 * 
 * Algorithm: Open addressing with linear probing and incremental step dispersion. 
 * Implements multiplicative hashing with prime constants for high key dispersion.
 * Memory Model: Device global memory for the core hashtable structure and entry pools.
 * Synchronization: Uses atomicCAS for thread-safe mapping and atomicAdd for occupancy tracking.
 * Domain: HPC, Parallel Data Structures.
 */

#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "gpu_hashtable.hpp"

#define MIN_LOAD_FACTOR 85


/**
 * @brief Prime-multiplicative hash for device-side index generation.
 * 
 * Time Complexity: O(1)
 */
__device__ int hashFunc(int data, int size) {
	return ((long)abs(data) * 905035071625626043llu) % 5746614499066534157llu % size;
}


/**
 * @brief CUDA kernel for initializing or updating device-resident metadata.
 */
__global__ void kernel_update_hash(hashtable *hash, hashelement *table, int capacity, int items) {
	hash->table = table;
	hash->capacity = capacity;
	hash->numElements = items;
}


/**
 * @brief CUDA kernel for parallel entry insertion.
 * 
 * Functional Utility: Claims empty slots or updates existing keys using 
 * atomic compare-and-swap. Employs a linear probe with an incremental 
 * step factor 'f' to resolve collisions.
 * 
 * @param hash Pointer to the device-resident hashtable metadata.
 */
__global__ void kernel_insert(int *keys, int *values, int numElements, hashtable *hash) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int f = 0;

	// Pre-condition: Thread index must be within batch input range.
	if (index >= numElements)
		return;

	int key = keys[index];
	int value = values[index];

	// Logic: Skip invalid input pairs.
	if (key == 0 || value == 0)
		return;

	int valueHash = hashFunc(key, hash->capacity);
	int tableIndex = valueHash;
	
	/**
	 * Block Logic: Collision resolution loop.
	 * Invariant: Probes until an empty slot is atomically claimed or 
	 * the key is matched for an update.
	 */
	for (int i = 0; i < hash->capacity; ++i) {
		int oldKey = atomicCAS(&hash->table[tableIndex].key, 0, key);
		
		if (oldKey == 0) {
			// Logic: Successfully claimed new slot, update occupancy count.
			atomicAdd(&hash->numElements, 1);
			hash->table[tableIndex].value = value;
			return;
		}
		else if (oldKey == key) {
			// Logic: Key already exists, update value in-place.
			hash->table[tableIndex].value = value;
			return;
		}
		
		// Optimization: Incremental linear probing with dynamic step adjustment.
		tableIndex = (valueHash + i * f) % hash->capacity;
		f += 1;
	}
}


/**
 * @brief CUDA kernel for migrating entries during table expansion.
 */
__global__ void kernel_rehash(hashtable *oldHash, hashelement *newTable, int newCapacity) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int f = 0;

	if (index >= oldHash->capacity)
		return;

	hashelement elementToInsert = oldHash->table[index];
	
	// Optimization: Skip empty source slots.
	if (elementToInsert.key == 0) {
		return;
	}

	int valueHash = hashFunc(elementToInsert.key, newCapacity);
	int tableIndex = valueHash;

	/**
	 * Block Logic: Migration re-insertion probe.
	 */
	for (int i = 0; i < newCapacity; ++i) {
		int oldKey = atomicCAS(&newTable[tableIndex].key, 0, elementToInsert.key);
		if (oldKey == 0) {
			newTable[tableIndex].value = elementToInsert.value;
			return;
		}
		tableIndex = (valueHash + i * f) % newCapacity;
		f += 1;
	}
}


/**
 * @brief CUDA kernel for parallel value lookup.
 */
__global__ void kernel_get(int *keys, int *values, int numElements, hashtable *hash) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int f = 0;

	if (index >= numElements)
		return;

	int key = keys[index];
	int valueHash = hashFunc(key, hash->capacity);
	int tableIndex = valueHash;
	
	/**
	 * Block Logic: Search probe traversal.
	 * Pre-condition: Hashed index used as search starting point.
	 * Invariant: Terminates upon key match or exhaustion of capacity.
	 */
	for (int i = 0; i < hash->capacity; ++i) {
		if (hash->table[tableIndex].key == key) {
			values[index] = hash->table[tableIndex].value;
			return;
		}
		tableIndex = (valueHash + i * f) % hash->capacity;
		f += 1;
	}
}


/**
 * @brief Constructor: Allocates GPU memory for both table structure and data pool.
 */
GpuHashTable::GpuHashTable(int size) {
	hashelement *table = NULL;
	cudaMalloc((void **)&table, size * sizeof(hashelement));
	cudaMemset(table, 0, size * sizeof(hashelement));

	cudaMalloc((void **)&hash, sizeof(hashtable));

	// Synchronization: Updates metadata on device before first operation.
	kernel_update_hash<<<1, 1>>>(hash, table, size, 0);

	cudaDeviceSynchronize();
}


/**
 * @brief Destructor: Releases device resources.
 */
GpuHashTable::~GpuHashTable() {
	hashtable *hostTable = (hashtable *)calloc(1, sizeof(hashtable));
	cudaMemcpy(hostTable, hash, sizeof(hashtable), cudaMemcpyDeviceToHost);
	cudaFree(hostTable->table);
	cudaFree(hash);
	free(hostTable);
}


/**
 * @brief Dynamically resizes the table if the load factor exceeds 90%.
 */
void GpuHashTable::reshape(int batchSize) {
	hashtable *hostTable = (hashtable *)calloc(1, sizeof(hashtable));
	cudaMemcpy(hostTable, hash, sizeof(hashtable), cudaMemcpyDeviceToHost);
	
	float currentLoad = ((float)hostTable->numElements + (float)batchSize) / ((float)hostTable->capacity);
	
	// Optimization: Proactive resizing to maintain O(1) expected complexity.
	if (currentLoad > 0.9) {
		int newCapacity = ((float)hostTable->numElements + (float)batchSize) * 100 / (float)MIN_LOAD_FACTOR;
		hashelement *newTable = NULL;
		cudaMalloc((void **)&newTable, newCapacity * sizeof(hashelement));
		cudaMemset(newTable, 0, newCapacity * sizeof(hashelement));
		
		int threadsPerBlock = 1024;
		int blocks = hostTable->capacity / threadsPerBlock + 1;

		// Synchronization: Kernel barrier ensuring migration is complete.
		kernel_rehash<<<blocks, threadsPerBlock>>>(hash, newTable, newCapacity);
		cudaDeviceSynchronize();
		
		cudaFree(hostTable->table);
		hostTable->table = newTable;
		hostTable->capacity = newCapacity;
		
		// Logic: Update device-side metadata with new buffer address and capacity.
		cudaMemcpy(hash, hostTable, sizeof(hashtable), cudaMemcpyHostToDevice);
	}
	free(hostTable);
}


/**
 * @brief Orchestrates batch insertion from Host to Device.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	reshape(numKeys);
	int threadsPerBlock = 1024; 
	int blocks = numKeys / threadsPerBlock + 1;
	int *deviceKeys = NULL;
	int *deviceValues = NULL;
	
	cudaMalloc((void **)&deviceKeys, numKeys * sizeof(int));
	cudaMalloc((void **)&deviceValues, numKeys * sizeof(int));
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	kernel_insert<<<blocks, threadsPerBlock>>>(deviceKeys, deviceValues, numKeys, hash);
	
	cudaDeviceSynchronize();
	cudaFree(deviceKeys);
	cudaFree(deviceValues);
	return true;
}


/**
 * @brief Orchestrates batch retrieval from Device to Host.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *getValues = NULL;
	int *deviceKeys = NULL;
	cudaMalloc((void **)&getValues, numKeys * sizeof(int));
	cudaMalloc((void **)&deviceKeys, numKeys * sizeof(int));
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	int threadsPerBlock = 1024; 
	int blocks = numKeys / threadsPerBlock + 1;
	kernel_get<<<blocks, threadsPerBlock>>>(deviceKeys, getValues, numKeys, hash);
	
	int *hostValues = (int *)calloc(numKeys, sizeof(int));
	cudaMemcpy(hostValues, getValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	cudaFree(deviceKeys);
	cudaFree(getValues);
	return hostValues;
}


/**
 * @brief Calculates current utilization density.
 */
float GpuHashTable::loadFactor() {
	hashtable *hostTable = (hashtable *)calloc(1, sizeof(hashtable));
	cudaMemcpy(hostTable, hash, sizeof(hashtable), cudaMemcpyDeviceToHost);
	float loadFactor = ((float)hostTable->numElements) / ((float)hostTable->capacity);
	free(hostTable);
	return loadFactor;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <ctime>
#include <random>

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

using namespace std;

/**
 * @brief Test utility for populating vectors with randomized keys and values.
 */
void fillRandom(vector<int> &vecKeys, vector<int> &vecValues, int numEntries) {
	vecKeys.reserve(numEntries);
	vecValues.reserve(numEntries);

	int interval = (numeric_limits<int>::max() / numEntries) - 1;
	default_random_engine generator;
	uniform_int_distribution<int> distribution(1, interval);

	for (int i = 0; i < numEntries; i++) {
		vecKeys.push_back(interval * i + distribution(generator));
		vecValues.push_back(interval * i + distribution(generator));
	}

	random_shuffle(vecKeys.begin(), vecKeys.end());
	random_shuffle(vecValues.begin(), vecValues.end());
}

int main(int argc, char **argv)
{
	clock_t begin;
	double elapsedTime;

	int numKeys = 0;
	int numChunks = 0;
	vector<int> vecKeys;
	vector<int> vecValues;
	int *valuesGot = NULL;

	DIE(argc != 3,
		"ERR, args num, call ./bin test_numKeys test_numChunks");

	numKeys = stoll(argv[1]);
	numChunks = stoll(argv[2]);

	fillRandom(vecKeys, vecValues, numKeys);

	HASH_INIT;

	int chunkSize = numKeys / numChunks;
	HASH_RESERVE(chunkSize);

	/**
	 * Block Logic: Throughput profiling for parallel insertions.
	 */
	for (int chunkStart = 0; chunkStart < numKeys; chunkStart += chunkSize) {

		int* keysStart = &vecKeys[chunkStart];
		int* valuesStart = &vecValues[chunkStart];

		begin = clock();
		
		HASH_BATCH_INSERT(keysStart, valuesStart, chunkSize);
		elapsedTime = double(clock() - begin) / CLOCKS_PER_SEC;

		cout << "HASH_BATCH_INSERT, " << chunkSize
			<< ", " << chunkSize / elapsedTime / 1000000
			<< ", " << 100.f * HASH_LOAD_FACTOR << endl;
	}

	// Logic: Test for value update on existing keys.
	int chunkSizeUpdate = min(64, numKeys);
	for (int chunkStart = 0; chunkStart < chunkSizeUpdate; chunkStart++) {
		vecValues[chunkStart] += 1111111 + chunkStart;
	}
	HASH_BATCH_INSERT(&vecKeys[0], &vecValues[0], chunkSizeUpdate);

	/**
	 * Block Logic: Throughput profiling for parallel lookups.
	 */
	for (int chunkStart = 0; chunkStart < numKeys; chunkStart += chunkSize) {

		int* keysStart = &vecKeys[chunkStart];

		begin = clock();
		
		valuesGot = HASH_BATCH_GET(keysStart, chunkSize);
		elapsedTime = double(clock() - begin) / CLOCKS_PER_SEC;

		cout << "HASH_BATCH_GET, " << chunkSize
			<< ", " << chunkSize / elapsedTime / 1000000
			<< ", " << 100.f * HASH_LOAD_FACTOR << endl;

		DIE(valuesGot == NULL, "ERR, ptr valuesCheck cannot be NULL");

		// Validation: Confirms correctness of retrieved values against source data.
		int mistmatches = 0;
		for (int i = 0; i < chunkSize; i++) {
			if (vecValues[chunkStart + i] != valuesGot[i]) {
				mistmatches++;
			}
		}

		if (mistmatches > 0) {
			cout << "ERR, mistmatches: " << mistmatches << " / " << numKeys << endl;
			exit(1);
		}
		free(valuesGot);
	}

	return 0;
}


#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0

/**
 * @struct hashelement
 * @brief Base storage unit for a key-value mapping on the GPU.
 */
typedef struct {
	int key;
	int value;
} hashelement;


/**
 * @struct hashtable
 * @brief Metadata container for the GPU resident map.
 */
typedef struct {
	hashelement *table;
	int capacity;
	int numElements;
} hashtable;


/**
 * @class GpuHashTable
 * @brief Host manager for the GPU hash table lifecycle and batch operations.
 */
class GpuHashTable
{
	hashtable *hash;
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
