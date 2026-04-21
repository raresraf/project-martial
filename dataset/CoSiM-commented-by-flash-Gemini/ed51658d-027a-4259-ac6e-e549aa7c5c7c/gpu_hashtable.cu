
#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "gpu_hashtable.hpp"

/**
 * @file gpu_hashtable.cu
 * @brief Thread-safe parallel hash table with dynamic load-factor management on GPU.
 * 
 * Functional Intent: Implements a high-performance hash table using Open Addressing 
 * and linear probing. It utilizes `atomicCAS` to manage concurrent slot claims 
 * and `atomicAdd` for global element counting. The implementation features an 
 * automatic `reshape` mechanism that expands capacity when the load factor 
 * exceeds a defined threshold, ensuring efficient O(1) performance for large 
 * batch operations.
 * 
 * Domain: HPC, Parallel Data Structures, CUDA.
 */

#define MIN_LOAD_FACTOR 85

/**
 * hashFunc - Device-side hash generation using a large prime multiplier.
 */
__device__ int hashFunc(int data, int size) {
	return ((long)abs(data) * 905035071625626043llu) % 5746614499066534157llu % size;
}

/**
 * @kernel kernel_update_hash
 * @brief Synchronizes host-side metadata with the device-side hashtable state.
 */
__global__ void kernel_update_hash(hashtable *hash, hashelement *table, int capacity, int items) {
	hash->table = table;
	hash->capacity = capacity;
	hash->numElements = items;
}

/**
 * @kernel kernel_insert
 * @brief Parallel insertion kernel using atomic compare-and-swap for slot reservation.
 * 
 * Algorithm: Open Addressing with Linear Probing.
 * 1. Threads attempt to claim an empty slot (key=0) via `atomicCAS`.
 * 2. On success, the element count is incremented and the value is written.
 * 3. On collision with the same key, the value is updated in-place.
 * 4. On collision with a different key, linear probing continues until a slot is found.
 */
__global__ void kernel_insert(int *keys, int *values, int numElements, hashtable *hash) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int f = 0;
	if (index >= numElements)
		return;
	
	int key = keys[index];
	int value = values[index];
	if (key == 0 || value == 0)
		return;

	int valueHash = hashFunc(key, hash->capacity);
	int tableIndex = valueHash;

	/**
	 * Block Logic: Insertion Probe Loop.
	 * Invariant: Loop persists until the key is successfully placed or the table is exhausted.
	 */
	for (int i = 0; i < hash->capacity; ++i) {
		// Synchronization: atomicCAS ensures exclusive access to the bucket's key field.
		int oldKey = atomicCAS(&hash->table[tableIndex].key, 0, key);
		
		if (oldKey == 0) {
			// Case: Successfully claimed empty slot.
			atomicAdd(&hash->numElements, 1);
			hash->table[tableIndex].value = value;
			return;
		}
		else if (oldKey == key) {
			// Case: Key already exists, performing update.
			hash->table[tableIndex].value = value;
			return;
		}
		// Case: Collision; update index via linear probing.
		tableIndex = (valueHash + i) % hash->capacity;
	}
}

/**
 * @kernel kernel_rehash
 * @brief Efficiently migrates entries from the old table to a new, larger buffer.
 */
__global__ void kernel_rehash(hashtable *oldHash, hashelement *newTable, int newCapacity) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= oldHash->capacity)
		return;

	hashelement elementToInsert = oldHash->table[index];
	if (elementToInsert.key == 0) {
		return;
	}
	
	int valueHash = hashFunc(elementToInsert.key, newCapacity);
	int tableIndex = valueHash;

	// Block Logic: Migration Probe.
	for (int i = 0; i < newCapacity; ++i) {
		int oldKey = atomicCAS(&newTable[tableIndex].key, 0, elementToInsert.key);
		if (oldKey == 0) {
			newTable[tableIndex].value = elementToInsert.value;
			return;
		}
		tableIndex = (valueHash + i) % newCapacity;
	}
}

/**
 * @kernel kernel_get
 * @brief Parallel retrieval kernel using linear probing for key lookup.
 */
__global__ void kernel_get(int *keys, int *values, int numElements, hashtable *hash) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numElements)
		return;

	int key = keys[index];
	int valueHash = hashFunc(key, hash->capacity);
	int tableIndex = valueHash;

	for (int i = 0; i < hash->capacity; ++i) {
		hashelement element = hash->table[tableIndex];
		if (element.key == key) {
			values[index] = element.value;
			return;
		}
		if (element.key == 0) return; // Optimization: Early exit on empty bucket (linear probing invariant).
		tableIndex = (valueHash + i) % hash->capacity;
	}
}

/**
 * GpuHashTable constructor - Orchestrates dual-allocation for metadata and storage.
 */
GpuHashTable::GpuHashTable(int size) {
	hashelement *table = NULL;
	cudaMalloc((void **)&table, size * sizeof(hashelement));
	cudaMemset(table, 0, size * sizeof(hashelement));

	// Logic: Persistence of table metadata in device memory for kernel access.
	cudaMalloc((void **)&hash, sizeof(hashtable));
	kernel_update_hash<<<1, 1>>>(hash, table, size, 0);
	cudaDeviceSynchronize();
}

GpuHashTable::~GpuHashTable() {
	hashtable hostMetadata;
	cudaMemcpy(&hostMetadata, hash, sizeof(hashtable), cudaMemcpyDeviceToHost);
	cudaFree(hostMetadata.table);
	cudaFree(hash);
}

/**
 * reshape - Dynamically resizes the table based on load factor thresholds.
 * 
 * Logic: Triggers a rehash if the projected load factor (including batch) 
 * exceeds 90%. Ensures the hash table maintains optimal search density.
 */
void GpuHashTable::reshape(int batchSize) {
	hashtable *hostTable = (hashtable *)calloc(1, sizeof(hashtable));
	cudaMemcpy(hostTable, hash, sizeof(hashtable), cudaMemcpyDeviceToHost);
	
	float loadFactor = ((float)hostTable->numElements + (float)batchSize) / ((float)hostTable->capacity);
	if (loadFactor > 0.9) {
		int newCapacity = ((float)hostTable->numElements + (float)batchSize) * 100 / (float)MIN_LOAD_FACTOR;
		hashelement *newTable = NULL;
		cudaMalloc((void **)&newTable, newCapacity * sizeof(hashelement));
		cudaMemset(newTable, 0, newCapacity * sizeof(hashelement));

		int threadsPerBlock = 1024;
		int blocks = (hostTable->capacity + threadsPerBlock - 1) / threadsPerBlock;

		// Block Logic: Migration execution.
		kernel_rehash<<<blocks, threadsPerBlock>>>(hash, newTable, newCapacity);
		cudaDeviceSynchronize();
		
		cudaFree(hostTable->table);
		hostTable->table = newTable;
		hostTable->capacity = newCapacity;
		
		// Synchronization: Commit new metadata to device memory.
		cudaMemcpy(hash, hostTable, sizeof(hashtable), cudaMemcpyHostToDevice);
	}
	free(hostTable);
}

/**
 * insertBatch - Public interface for parallel data ingestion.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	reshape(numKeys);
	int threadsPerBlock = 1024; 
	int blocks = (numKeys + threadsPerBlock - 1) / threadsPerBlock;
	
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
 * getBatch - Public interface for parallel retrieval.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *deviceValues = NULL;
	int *deviceKeys = NULL;
	cudaMalloc((void **)&deviceValues, numKeys * sizeof(int));
	cudaMalloc((void **)&deviceKeys, numKeys * sizeof(int));
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	int threadsPerBlock = 1024; 
	int blocks = (numKeys + threadsPerBlock - 1) / threadsPerBlock;
	
	kernel_get<<<blocks, threadsPerBlock>>>(deviceKeys, deviceValues, numKeys, hash);
	
	int *hostValues = (int *)calloc(numKeys, sizeof(int));
	cudaMemcpy(hostValues, deviceValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	cudaFree(deviceKeys);
	cudaFree(deviceValues);
	return hostValues;
}

float GpuHashTable::loadFactor() {
	hashtable hostMetadata;
	cudaMemcpy(&hostMetadata, hash, sizeof(hashtable), cudaMemcpyDeviceToHost);
	return ((float)hostMetadata.numElements) / ((float)hostMetadata.capacity);
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

	int chunkSizeUpdate = min(64, numKeys);
	for (int chunkStart = 0; chunkStart < chunkSizeUpdate; chunkStart++) {
		vecValues[chunkStart] += 1111111 + chunkStart;
	}
	HASH_BATCH_INSERT(&vecKeys[0], &vecValues[0], chunkSizeUpdate);

	for (int chunkStart = 0; chunkStart < numKeys; chunkStart += chunkSize) {

		int* keysStart = &vecKeys[chunkStart];

		begin = clock();
		
		valuesGot = HASH_BATCH_GET(keysStart, chunkSize);
		elapsedTime = double(clock() - begin) / CLOCKS_PER_SEC;

		cout << "HASH_BATCH_GET, " << chunkSize
			<< ", " << chunkSize / elapsedTime / 1000000
			<< ", " << 100.f * HASH_LOAD_FACTOR << endl;

		DIE(valuesGot == NULL, "ERR, ptr valuesCheck cannot be NULL");

		int mistmatches = 0;
		for (int i = 0; i < chunkSize; i++) {
			if (vecValues[chunkStart + i] != valuesGot[i]) {
				mistmatches++;
				if (mistmatches < 32) {
					cout << "Expected " << vecValues[chunkStart + i]
						<< ", but got " << valuesGot[i] << " for key:" << keysStart[i] << endl;
				}
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

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)

typedef struct {
	int key;
	int value;
} hashelement;

typedef struct {
	hashelement *table;
	int capacity;
	int numElements;
} hashtable;

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
