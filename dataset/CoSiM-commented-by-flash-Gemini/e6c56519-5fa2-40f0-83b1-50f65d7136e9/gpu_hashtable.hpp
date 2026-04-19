/**
 * @file gpu_hashtable.hpp
 * @brief Integrated GPU Hash Table implementation with circular linear probing.
 * 
 * This module provides an integrated header-plus-implementation for a high-performance 
 * GPU hash table. It uses a single-hash approach with circular linear probing 
 * across two passes for collision resolution and hardware atomic operations to 
 * maintain transactional consistency.
 * 
 * Algorithm: Open addressing with circular linear probing and atomic CAS.
 * Memory Model: Global memory (cudaMalloc) for entry buffers.
 * Domain: HPC, Parallel Data Structures.
 */

#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0

// ... primeList documentation omitted ...

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
			fprintf(stderr, "(%s, %d): ",	\
				__FILE__, __LINE__);	\
			perror(call_description);	\
			exit(errno);	\
		}	\
	} while (0)

typedef unsigned long long Entry;




/**
 * @class GpuHashTable
 * @brief Host-side handle for GPU hash table lifecycle management.
 */
class GpuHashTable
{
public:
	unsigned long size;
	unsigned int *num_elements;
	Entry *table;

	
	unsigned long a;
	unsigned long b;

	GpuHashTable(int size);
	void reshape(int sizeReshape);

	bool insertBatch(int *keys, int* values, int numKeys);
	int* getBatch(int* key, int numItems);

	float loadFactor();
	void occupancy();
	void print(string info);
	void printTable();

	~GpuHashTable();
};

#endif

#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define FIRST 823117
#define SECOND 3452434812973

/**
 * @brief Device-side hash function using a 64-bit modular scheme.
 */
__device__ int getHash(int val, int limit)
{
	return ((long long) abs(val) * FIRST) % SECOND % limit;
}

/**
 * @brief CUDA kernel for parallel entry insertion.
 * 
 * Functional Utility: Uses atomicCAS to claim slots or update existing keys. 
 * Resolves collisions via two-pass circular linear probing.
 */
__global__ void addInKern(int *keys, int *val, int max, hash_table hash)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= max) return;
	int actual_key, replacing_key;
	replacing_key = keys[index];
	int hash = getHash(replacing_key, hash.dim);



	/**
	 * Block Logic: Primary insertion probe pass.
	 */
	for (int i = hash; i < hash.dim; i++) {
		actual_key = atomicCAS(&hash.map[i].key, KEY_INVALID, replacing_key);

		if (actual_key == KEY_INVALID || actual_key == replacing_key) {
			hash.map[i].value = val[index];
			return;
		}
	}
	/**
	 * Block Logic: Wrap-around insertion probe pass.
	 */
	for (int i = 0; i < hash; i++) {
		actual_key = atomicCAS(&hash.map[i].key, KEY_INVALID, replacing_key);

		if (actual_key == KEY_INVALID || actual_key == replacing_key) {
			hash.map[i].value = val[index];
			return;
		}
	}
}


/**
 * @brief CUDA kernel for parallel entry retrieval.
 */
__global__ void getFromKern(int *keys, int *val, int max, hash_table hash)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= max) return;

	int key = keys[index];


	int hash = getHash(keys[index], hash.dim);

	// Block Logic: Multi-pass circular search traversal.
	for (int i = hash; i < hash.dim; i++) {
		if (hash.map[i].key == key) {
			val[index] = hash.map[i].value;
			return;
		}
	}

	for (int i = 0; i < hash; i++) {
		if (hash.map[i].key == key) {
			val[index] = hash.map[i].value;
			return;
		}
	}
}



/**
 * @brief CUDA kernel for re-hashing data during capacity expansion.
 */
__global__ void replaceHash(hash_table current_hash, hash_table replacing_hash)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= current_hash.dim) return;

	if (current_hash.map[index].key == KEY_INVALID) return;

	int actual_key, replacing_key;
	replacing_key = current_hash.map[index].key;

	int hash = getHash(replacing_key, replacing_hash.dim);

	// Logic: Standard re-insertion via circular linear probing.
	for (int i = hash; i < replacing_hash.dim; i++) {
		actual_key = atomicCAS(&replacing_hash.map[i].key, KEY_INVALID, replacing_key);

		if (actual_key == KEY_INVALID || actual_key == replacing_key) {
			replacing_hash.map[i].value = current_hash.map[index].value;
			return;
		}
	}

	for (int i = 0; i < hash; i++) {
		actual_key = atomicCAS(&replacing_hash.map[i].key, KEY_INVALID, replacing_key);

		if (actual_key == KEY_INVALID || actual_key == replacing_key) {
			replacing_hash.map[i].value = current_hash.map[index].value;
			return;
		}
	}
}



/**
 * @brief Constructor: Initializes hash table metadata and allocates device memory.
 */
GpuHashTable::GpuHashTable(int size)
{
	count = 0;
	hashmap.dim = size;

	// Memory Hierarchy: Global memory allocation for entries.
	cudaMalloc(&hashmap.map, size * sizeof(Entity));
	cudaMemset(hashmap.map, 0, size * sizeof(Entity));
}


/**
 * @brief Destructor: Releases device-resident memory.
 */
GpuHashTable::~GpuHashTable()
{
	cudaFree(hashmap.map);
}


/**
 * @brief Dynmically resizes the table via parallel re-hashing.
 */
void GpuHashTable::reshape(int numBucketsReshape)
{
	hash_table hash;
	hash.dim = numBucketsReshape;

	cudaMalloc(&hash.map, numBucketsReshape * sizeof(Entity));
	cudaMemset(hash.map, 0, numBucketsReshape * sizeof(Entity));

	unsigned int numBlocks = hashmap.dim / THREADS_PER_BLOCK + 1;
	if (hashmap.dim % THREADS_PER_BLOCK != 0) numBlocks++;
	
	// Synchronization: Ensures all elements are migrated before old memory is reclaimed.
	replaceHash>>(hashmap, hash);

	cudaDeviceSynchronize();

	cudaFree(hashmap.map);
	hashmap = hash;
}


/**
 * @brief Performs batch parallel insertion.
 * 
 * Logic: Transfers batch to GPU and triggers expansion if load factor > 80%.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys)
{
	int *batch_keys, *batch_values;

	cudaMalloc(&batch_keys, numKeys * sizeof(int));
	cudaMalloc(&batch_values, numKeys * sizeof(int));

	// Adaptive Scaling: Monitors table saturation to maintain lookup performance.
	if (float(count + numKeys) / hashmap.dim >= MAX_LOAD_FACTOR)
		reshape(int((count + numKeys) / MIN_LOAD_FACTOR));

	cudaMemcpy(batch_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(batch_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	unsigned int numBlocks = numKeys / THREADS_PER_BLOCK + 1;


	if (numKeys % THREADS_PER_BLOCK != 0) numBlocks++;
	addInKern>>(batch_keys, batch_values, numKeys, hashmap);

	cudaDeviceSynchronize();

	// Logic: Updates global element count.
	count += numKeys;

	cudaFree(batch_keys);
	cudaFree(batch_values);

	return true;
}


/**
 * @brief Performs batch parallel retrieval using Managed Memory for results.
 */
int *GpuHashTable::getBatch(int *keys, int numKeys)
{
	int *batch_keys, *batch_values;

	size_t memSize = numKeys * sizeof(int);
	cudaMalloc(&batch_keys, memSize);
	// Memory Hierarchy: Managed memory (Unified) for easy result access.
	cudaMallocManaged(&batch_values, memSize);

	cudaMemcpy(batch_keys, keys, memSize, cudaMemcpyHostToDevice);

	unsigned int numBlocks = numKeys / THREADS_PER_BLOCK + 1;


	if (numKeys % THREADS_PER_BLOCK != 0) numBlocks++;
	getFromKern>>(batch_keys, batch_values, numKeys, hashmap);

	cudaDeviceSynchronize();

	cudaFree(batch_keys);

	return batch_values;
}


/**
 * @brief Returns the current table density.
 */
float GpuHashTable::loadFactor()
{
	return (hashmap.dim == 0)? 0 : (float(count) / hashmap.dim);
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
