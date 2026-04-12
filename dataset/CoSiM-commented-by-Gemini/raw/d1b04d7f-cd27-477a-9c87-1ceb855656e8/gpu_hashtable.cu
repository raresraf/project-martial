
/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table using a hybrid 2-choice hashing and
 *        linear probing collision resolution strategy.
 * @details This file implements a hash table on the GPU that uses two separate
 * data arrays (`data1` and `data2`), attempting to place an element in either
 * before resorting to linear probing. This approach is inspired by Cuckoo Hashing
 * but is augmented with a linear scan.
 *
 * It correctly uses `atomicExch` for thread-safe value updates, which is a significant
 * improvement over simpler models. However, the main `insert` and `get` kernels
 * contain a `while(1)` loop that may not terminate if the table becomes full and a
 * key cannot be found, representing a critical bug. The reshape logic is also
 * implemented inefficiently.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief Host-side constructor for the GpuHashTable.
 * @details Allocates two separate data arrays (`data1`, `data2`) on the GPU,
 * corresponding to the 2-choice hashing scheme.
 */
GpuHashTable::GpuHashTable(int size) {
	ht.size = size;
	ht.nrElem = 0;
	// Allocate the first hash table array.
	cudaMalloc((void **)&(ht.data1), size * sizeof(Pair));
	cudaMemset(ht.data1, 0, size * sizeof(Pair));
	// Allocate the second hash table array.
	cudaMalloc((void **)&(ht.data2), size * sizeof(Pair));
	cudaMemset(ht.data2, 0, size * sizeof(Pair));
}

/**
 * @brief Host-side destructor. Frees all GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(ht.data1);
	cudaFree(ht.data2);
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs.
 * @warning BUG: This kernel contains a `while(1)` loop that can lead to an
 * infinite loop if the hash table is full and an insertion is attempted.
 */
__global__ void insert(HashTable ht, int N, int *keys, int *values){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if(idx >= N) // Corrected guard condition
		return;
	int keyToInsert = keys[idx];
	int valueToInsert = values[idx];

	if (keyToInsert == 0)
		return;

	int hashIdx = hash1(keyToInsert, ht.size);

	// --- Hybrid Collision Resolution: 2-Choice then Linear Probing ---
	while(1){ // This can be an infinite loop.
		// Attempt 1: Update/insert into primary slot.
		if(atomicCAS(&(ht.data1[hashIdx].key), keyToInsert, keyToInsert) == keyToInsert){
			atomicExch(&(ht.data1[hashIdx].value), valueToInsert);
			break;
		}
		if(atomicCAS(&(ht.data1[hashIdx].key), 0, keyToInsert) == 0){
			atomicExch(&(ht.data1[hashIdx].value), valueToInsert);
			break;
		}

		// Attempt 2: Update/insert into secondary (cuckoo) slot.
		if(atomicCAS(&(ht.data2[hashIdx].key), keyToInsert, keyToInsert) == keyToInsert){
			atomicExch(&(ht.data2[hashIdx].value), valueToInsert);
			break;
		}
		if(atomicCAS(&(ht.data2[hashIdx].key), 0, keyToInsert) == 0){
			atomicExch(&(ht.data2[hashIdx].value), valueToInsert);
			break;
		}

		// Attempt 3: Linear probing.
		hashIdx++;
		hashIdx = hashIdx % ht.size;
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @warning BUG: This kernel contains a `while(1)` loop that can lead to an
 * infinite loop if the table is full and a key is not found.
 */
__global__ void get(HashTable ht, int N, int *keys, int *values){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if(idx >= N) // Corrected guard condition
		return;
	int keyToGet = keys[idx];
	int hashIdx = hash1(keyToGet, ht.size);

	// --- Hybrid Search: 2-Choice then Linear Probing ---
	while(1){ // This can be an infinite loop.
		// Check primary slot. (Unconventional use of atomicCAS for a read).
		if(atomicCAS(&(ht.data1[hashIdx].key), keyToGet, keyToGet) == keyToGet){
			values[idx] = ht.data1[hashIdx].value;
			break;
		}
		// Check secondary slot.
		if(atomicCAS(&(ht.data2[hashIdx].key), keyToGet, keyToGet) == keyToGet){
			values[idx] = ht.data2[hashIdx].value;
			break;
		}
		// Linear probing.
		hashIdx++;
		hashIdx = hashIdx % ht.size;
	}
}

/**
 * @brief Helper kernel to extract all key-value pairs from one of the data arrays.
 * @note This is used as part of an inefficient, multi-step reshape process.
 */
__global__ void getKeysAndValues(HashTable ht, int N, int *keys, int *values, int slot) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if(idx >= N) // Corrected guard condition
		return;
	if (slot == 1){
		keys[idx] = ht.data1[idx].key;
		values[idx] = ht.data1[idx].value;
	} else {
		keys[idx] = ht.data2[idx].key;
		values[idx] = ht.data2[idx].value;
	}
}

/**
 * @brief Resizes the hash table to a new capacity.
 * @note This implementation is functionally correct but highly inefficient. It extracts
 * all keys/values into temporary buffers and then re-inserts them, rather than
 * performing a direct in-place rehash from the old table to the new one.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	// 1. Create a new, larger hash table structure.
	HashTable newHt;
	newHt.size = numBucketsReshape;
	newHt.nrElem = 0;
	cudaMalloc((void **)&(newHt.data1), numBucketsReshape * sizeof(Pair));
	cudaMemset(newHt.data1, 0, numBucketsReshape * sizeof(Pair));
	cudaMalloc((void **)&(newHt.data2), numBucketsReshape * sizeof(Pair));
	cudaMemset(newHt.data2, 0, numBucketsReshape * sizeof(Pair));

	// 2. Allocate temporary GPU memory to hold all elements from the old table.
	int *keys, *values;
	cudaMalloc((void **)&(keys), ht.size * sizeof(int));
	cudaMalloc((void **)&(values), ht.size * sizeof(int));

	int numBlocks = (ht.size + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// 3. Extract all elements from the first old table (`data1`).
	getKeysAndValues>>(ht, ht.size, keys, values, 1);
	cudaDeviceSynchronize();

	// 4. Re-insert the extracted elements into the new table.
	insert>>(newHt, ht.size, keys, values);
	cudaDeviceSynchronize();

	// 5. Repeat the process for the second old table (`data2`).
	getKeysAndValues>>(ht, ht.size, keys, values, 2);
	cudaDeviceSynchronize();
	insert>>(newHt, ht.size, keys, values);
	cudaDeviceSynchronize();

	newHt.nrElem = ht.nrElem;

	// 6. Free the old table memory and update the pointer.
	cudaFree(ht.data1);
	cudaFree(ht.data2);
	ht = newHt;

	// 7. Free temporary buffers.
	cudaFree(keys);
	cudaFree(values);
}

/**
 * @brief Inserts a batch of key-value pairs from the host.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *devKeys, *devValues;
	cudaMalloc((void **)&(devKeys), numKeys * sizeof(int));
	if(!devKeys) return false;
	cudaMalloc((void **)&(devValues), numKeys * sizeof(int));
	if(!devValues) return false;

	cudaMemcpy(devKeys, keys, sizeof(int) * numKeys, cudaMemcpyHostToDevice);
	// BUG: sizeof(float) is used for an int array. This may cause incorrect data transfer.
	cudaMemcpy(devValues, values, sizeof(float) * numKeys, cudaMemcpyHostToDevice);

	// Resize if the load factor exceeds the threshold.
	if( (float)(ht.nrElem + numKeys)/(float)(ht.size) >= 0.9)
		reshape((int)((ht.nrElem + numKeys) / 0.8));
	
	int numBlocks = (numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE;
	
	insert>>(ht, numKeys, devKeys, devValues);
	cudaDeviceSynchronize();

	ht.nrElem += numKeys;
	cudaFree(devKeys);
	cudaFree(devValues);

	return true;
}

/**
 * @brief Retrieves values for a batch of keys from the host.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *devKeys, *devValues, *values;
	cudaMalloc((void **)&(devKeys), numKeys * sizeof(int));
	if(!devKeys) return NULL;
	cudaMalloc((void **)&(devValues), numKeys * sizeof(int));
	if(!devValues) return NULL;
	values = (int *)malloc(numKeys * sizeof(int));

	cudaMemcpy(devKeys, keys, sizeof(int) * numKeys, cudaMemcpyHostToDevice);

	int numBlocks = (numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE;

	get>>(ht, numKeys, devKeys, devValues);
	cudaDeviceSynchronize();

	cudaMemcpy(values, devValues, sizeof(int) * numKeys, cudaMemcpyDeviceToHost);

	cudaFree(devKeys);
	cudaFree(devValues);

	return values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
	return (float)ht.nrElem/(float)ht.size; 
}


// --- Boilerplate test harness ---
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
#define BLOCK_SIZE 512

#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)

__device__ const size_t primeList[] = {/* ... */};

// Hash functions for different schemes.
__device__ int hash1(int data, int limit) {
	return ((long)abs(data) * 73llu) % 7240280573005008577llu % limit;
}
__device__ int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
__device__ int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

// Data structure for a single hash table entry (key-value pair).
typedef struct pair {
	int key;
	int value;
} Pair;

// Host-side representation of the hash table's core data.
typedef struct hashtable {
	int nrElem;
	int size;
	Pair *data1;
	Pair *data2;
} HashTable;

class GpuHashTable
{
	HashTable ht;
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
