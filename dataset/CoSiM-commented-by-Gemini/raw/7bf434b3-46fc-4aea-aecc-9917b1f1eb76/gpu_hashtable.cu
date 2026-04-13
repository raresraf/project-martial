/**
 * @file gpu_hashtable.cu
 * @brief Implements a hash table on the GPU using CUDA.
 *
 * This file contains the implementation of a GPU-accelerated hash table. It includes
 * CUDA kernels for insertion, retrieval, and rehashing, as well as a host-side C++
 * class (`GpuHashTable`) to manage the data structure. The implementation uses
 * atomic operations to handle concurrent access from multiple threads on the GPU,
 * ensuring thread-safe operations.
 */
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <chrono>

#include "gpu_hashtable.hpp"
#define MIN_LOAD_FACTOR 85

/**
 * @brief Computes the hash of a key.
 * @param data The key to be hashed.
 * @param size The size of the hash table, used for the modulo operation.
 * @return The hash value.
 * @note This is a device function, intended to be called from CUDA kernels.
 */
__device__ int hashFunc(int data, int size) {
	return ((long)abs(data) * 905035071625626043llu) % 5746614499066534157llu % size;
}

/**
 * @brief CUDA kernel to update the hash table's metadata on the device.
 * @param hash A pointer to the hashtable structure on the device.
 * @param table A pointer to the hash table's data array on the device.
 * @param capacity The capacity of the hash table.
 * @param items The number of elements currently in the hash table.
 */
__global__ void kernel_update_hash(hashtable *hash, hashelement *table, int capacity, int items) {
	hash->table = table;
	hash->capacity = capacity;
	hash->numElements = items;
}

/**
 * @brief CUDA kernel to insert key-value pairs into the hash table.
 * @param keys An array of keys to be inserted.
 * @param values An array of values corresponding to the keys.
 * @param numElements The number of key-value pairs to insert.
 * @param hash A pointer to the hashtable structure on the device.
 * @note Each thread handles the insertion of one key-value pair.
 *       Uses atomicCAS for thread-safe insertion and linear probing for collision resolution.
 */
__global__ void kernel_insert(int *keys, int *values, int numElements, hashtable *hash) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numElements)
		return;

	int key = keys[index];
	int value = values[index];
	if (key == 0 || value == 0)
		return;

	int valueHash = hashFunc(key, hash->capacity);
	int tableIndex = valueHash;
	hashelement elementToInsert;
	elementToInsert.key = key;
	elementToInsert.value = value;
	
	// Linear probing for collision resolution.
	for (int i = 0; i < hash->capacity; ++i) {
		// atomicCAS (Compare-And-Swap) ensures that only one thread can insert into an empty slot.
		if (atomicCAS(&hash->table[tableIndex].key, 0, elementToInsert.key) == 0) {
			atomicAdd(&hash->numElements, 1);
			hash->table[tableIndex].value = value;
			return;
		}
		// If the key already exists, update the value.
		if (hash->table[tableIndex].key == key) {
			hash->table[tableIndex].value = value;
			return;
		}
		tableIndex = (tableIndex + 1) % hash->capacity;
	}
}

/**
 * @brief CUDA kernel to rehash the elements from an old table to a new, larger one.
 * @param oldHash A pointer to the old hashtable structure.
 * @param newTable A pointer to the new, larger hash table data array.
 * @param newCapacity The capacity of the new hash table.
 * @note Each thread is responsible for rehashing one element from the old table.
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

	for (int i = 0; i < newCapacity; ++i) {
		if (atomicCAS(&newTable[tableIndex].key, 0, elementToInsert.key) == 0) {
			newTable[tableIndex].value = elementToInsert.value;
			return;
		}
		tableIndex = (tableIndex + 1) % newCapacity;
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param keys An array of keys to look up.
 * @param values An array where the retrieved values will be stored.
 * @param numElements The number of keys to look up.
 * @param hash A pointer to the hashtable structure on the device.
 * @note Each thread handles the retrieval of one value.
 */
__global__ void kernel_get(int *keys, int *values, int numElements, hashtable *hash) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numElements)
		return;

	int key = keys[index];
	int valueHash = hashFunc(key, hash->capacity);
	int tableIndex = valueHash;

	for (int i = 0; i < hash->capacity; ++i) {
		hashelement elementToGet = hash->table[tableIndex];
		if (elementToGet.key == key) {
			values[index] = elementToGet.value;
			return;
		}
		if (elementToGet.key == 0) {
			// Key not found
			values[index] = 0;
			return;
		}
		tableIndex = (tableIndex + 1) % hash->capacity;
	}
	values[index] = 0; // Key not found after full scan
}

/**
 * @brief Constructor for the GpuHashTable class.
 * @param size The initial capacity of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	hashelement *table = NULL;
	cudaMalloc((void **)&table, size * sizeof(hashelement));
	cudaMemset(table, 0, size * sizeof(hashelement));
	cudaMalloc((void **)&hash, sizeof(hashtable));
	kernel_update_hash<<<1, 1>>>(hash, table, size, 0);
	cudaDeviceSynchronize();
}

/**
 * @brief Destructor for the GpuHashTable class.
 */
GpuHashTable::~GpuHashTable() {
	// To get the table pointer, we need to copy the hashtable struct back to the host
	hashtable host_hash;
	cudaMemcpy(&host_hash, hash, sizeof(hashtable), cudaMemcpyDeviceToHost);
	cudaFree(host_hash.table);
	cudaFree(hash);
}

/**
 * @brief Reshapes the hash table if the load factor exceeds a threshold.
 * @param batchSize The size of the incoming batch, used to predict the new load factor.
 */
void GpuHashTable::reshape(int batchSize) {
	hashtable hostTable;
	cudaMemcpy(&hostTable, hash, sizeof(hashtable), cudaMemcpyDeviceToHost);

	float loadFactor = ((float)hostTable.numElements + (float)batchSize) / ((float)hostTable.capacity);
	if (loadFactor > 0.9) {
		int newCapacity = ((float)hostTable.numElements + (float)batchSize) * 100 / (float)MIN_LOAD_FACTOR;
		
		hashelement *newTable = NULL;
		cudaMalloc((void **)&newTable, newCapacity * sizeof(hashelement));
		cudaMemset(newTable, 0, newCapacity * sizeof(hashelement));

		int threadsPerBlock = 1024;
		int blocks = (hostTable.capacity + threadsPerBlock - 1) / threadsPerBlock;

		kernel_rehash<<<blocks, threadsPerBlock>>>(hash, newTable, newCapacity);
		cudaDeviceSynchronize();

		hashelement* oldDeviceTable;
		cudaMemcpy(&oldDeviceTable, &hash->table, sizeof(hashelement*), cudaMemcpyDeviceToHost);
		cudaFree(oldDeviceTable);
		
		hostTable.table = newTable;
		hostTable.capacity = newCapacity;
		
		cudaMemcpy(hash, &hostTable, sizeof(hashtable), cudaMemcpyHostToDevice);
	}
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys A pointer to the array of keys on the host.
 * @param values A pointer to the array of values on the host.
 * @param numKeys The number of key-value pairs to insert.
 * @return True on successful insertion.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	reshape(numKeys);

	int *deviceKeys = NULL;
	int *deviceValues = NULL;
	cudaMalloc((void **)&deviceKeys, numKeys * sizeof(int));
	cudaMalloc((void **)&deviceValues, numKeys * sizeof(int));
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int threadsPerBlock = 1024; 
	int blocks = (numKeys + threadsPerBlock - 1) / threadsPerBlock;

	kernel_insert<<<blocks, threadsPerBlock>>>(deviceKeys, deviceValues, numKeys, hash);
	cudaDeviceSynchronize();

	cudaFree(deviceKeys);
	cudaFree(deviceValues);
	return true;
}

/**
 * @brief Retrieves the values for a batch of keys.
 * @param keys A pointer to the array of keys on the host.
 * @param numKeys The number of keys to retrieve.
 * @return A pointer to an array on the host containing the retrieved values.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *getValues = NULL;
	int *deviceKeys = NULL;
	cudaMalloc((void **)&getValues, numKeys * sizeof(int));
	cudaMalloc((void **)&deviceKeys, numKeys * sizeof(int));
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int threadsPerBlock = 1024; 
	int blocks = (numKeys + threadsPerBlock - 1) / threadsPerBlock;

	kernel_get<<<blocks, threadsPerBlock>>>(deviceKeys, getValues, numKeys, hash);

	int *hostValues = (int *)calloc(numKeys, sizeof(int));
	cudaMemcpy(hostValues, getValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree(deviceKeys);
	cudaFree(getValues);
	return hostValues;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor (numElements / capacity).
 */
float GpuHashTable::loadFactor() {
	int numElements = 0;
	int capacity = 0;
	// We need to access members of a struct on the device, so we copy them individually.
	cudaMemcpy(&numElements, &hash->numElements, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&capacity, &hash->capacity, sizeof(int), cudaMemcpyDeviceToHost);
	if (capacity == 0) return 0;
	return ((float)numElements) / ((float)capacity);
}

// -- Main function for benchmarking and testing --

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#define	KEY_INVALID		0
#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)

using namespace std;

/**
 * @brief Fills vectors with random integer keys and values.
 * @param vecKeys The vector to fill with keys.
 * @param vecValues The vector to fill with values.
 * @param numEntries The number of entries to generate.
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

/**
 * @brief Main entry point for the test program.
 *
 * This program benchmarks the GPU hash table implementation. It takes the number of keys
 * and the number of chunks as command-line arguments. It then fills the hash table
 * with random data and measures the performance of insertion and retrieval operations.
 */
int main(int argc, char **argv)
{
	clock_t begin;
	double elapsedTime;

	int numKeys = 0;
	int numChunks = 0;
	vector<int> vecKeys;
	vector<int> vecValues;
	int *valuesGot = NULL;

	DIE(argc != 3, "ERR, args num, call ./bin test_numKeys test_numChunks");

	numKeys = stoll(argv[1]);
	DIE((numKeys <= 0 || numKeys >= numeric_limits<int>::max()), "ERR, numKeys should be a positive integer less than maxint");

	numChunks = stoll(argv[2]);
	DIE((numChunks <= 0 || numChunks > numKeys), "ERR, numChunks should be greater than 0 and less than or equal to numKeys");

	fillRandom(vecKeys, vecValues, numKeys);

	HASH_INIT;

	int chunkSize = numKeys / numChunks;
	HASH_RESERVE(chunkSize);

	// Benchmark insertion
	for (int chunkStart = 0; chunkStart < numKeys; chunkStart += chunkSize) {
		int currentChunkSize = min(chunkSize, numKeys - chunkStart);
		int* keysStart = &vecKeys[chunkStart];
		int* valuesStart = &vecValues[chunkStart];

		begin = clock();
		HASH_BATCH_INSERT(keysStart, valuesStart, currentChunkSize);
		elapsedTime = double(clock() - begin) / CLOCKS_PER_SEC;

		cout << "HASH_BATCH_INSERT, " << currentChunkSize
			<< ", " << currentChunkSize / elapsedTime / 1000000
			<< ", " << 100.f * HASH_LOAD_FACTOR << endl;
	}

	// Update a small chunk of values to test update functionality
	int chunkSizeUpdate = min(64, numKeys);
	for (int i = 0; i < chunkSizeUpdate; i++) {
		vecValues[i] += 1111111 + i;
	}
	HASH_BATCH_INSERT(&vecKeys[0], &vecValues[0], chunkSizeUpdate);

	// Benchmark retrieval and verify correctness
	for (int chunkStart = 0; chunkStart < numKeys; chunkStart += chunkSize) {
		int currentChunkSize = min(chunkSize, numKeys - chunkStart);
		int* keysStart = &vecKeys[chunkStart];

		begin = clock();
		valuesGot = HASH_BATCH_GET(keysStart, currentChunkSize);
		elapsedTime = double(clock() - begin) / CLOCKS_PER_SEC;

		cout << "HASH_BATCH_GET, " << currentChunkSize
			<< ", " << currentChunkSize / elapsedTime / 1000000
			<< ", " << 100.f * HASH_LOAD_FACTOR << endl;

		DIE(valuesGot == NULL, "ERR, ptr valuesCheck cannot be NULL");

		int mistmatches = 0;
		for (int i = 0; i < currentChunkSize; i++) {
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
