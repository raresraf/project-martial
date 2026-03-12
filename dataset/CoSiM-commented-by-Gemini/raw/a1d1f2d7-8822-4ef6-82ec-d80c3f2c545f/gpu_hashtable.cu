/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA.
 * @details This file provides a hash table implementation that leverages the parallelism
 *          of the GPU for batch operations like insert and get. It uses open addressing
 *          with linear probing for collision resolution and atomic operations for
 *          thread-safe access to the hash table data structures in global memory.
 *
 * @note The file structure is unconventional, seemingly combining implementation
 *       (.cu) and declaration (.hpp) parts, and including a .cpp file directly.
 *       The code also contains several syntax errors (e.g., in kernel launch syntax
 *       and conditional statements) which are preserved as per instructions.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include <string>

#include "gpu_hashtable.hpp"




/**
 * @brief CUDA kernel for inserting a batch of key-value pairs into the hash table.
 * @details Each thread is responsible for inserting one key-value pair. It uses
 *          linear probing to resolve hash collisions. `atomicCAS` is used to
 *          ensure that only one thread can claim an empty slot, which is crucial
 *          for correctness in a concurrent environment. All data resides in global GPU memory.
 *
 * @param keys      (Device Pointer) Input keys to insert.
 * @param values    (Device Pointer) Input values corresponding to the keys.
 * @param numKeys   The number of key-value pairs in the batch.
 * @param h_keys    (Device Pointer) The keys array of the hash table.
 * @param h_values  (Device Pointer) The values array of the hash table.
 * @param limit     The total capacity of the hash table.
 */
__global__ void m_insert(int* keys, int* values, int numKeys,
	int *h_keys, int* h_values, int limit)
{
	// Thread-to-data mapping: Each thread processes one element from the input batch.
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int index;
	int old;

	
	// Boundary check to ensure the thread is within the bounds of the input array.
	if (i < numKeys && keys[i] != 0) { // KEY_INVALID is 0
		
		// Calculate the initial hash index.
		index = hash_func(keys[i], limit);

		while (true) {
			
			// If the key already exists, update its value and return.
			if (h_keys[index] == keys[i]) {
				h_values[index] = values[i];
				return;
			}

			
			// Attempt to claim an empty slot (where key is 0) using an atomic Compare-And-Swap.
			// This is the core synchronization mechanism for concurrent inserts.
			old = atomicCAS(&h_keys[index], 0, keys[i]);
			if (old == 0) {
				h_values[index] = values[i];
				return;
			} else {
				// Collision: move to the next slot (linear probing).
				index = (index + 1) % limit;
			}
		}
	}
}


/**
 * @brief CUDA kernel to check for the existence of a batch of keys.
 * @details Each thread checks for one key. It uses linear probing to traverse the
 *          hash table. `atomicAdd` is used to safely increment a counter in global
 *          memory when a key is found.
 *
 * @param keys      (Device Pointer) Input keys to check.
 * @param numKeys   The number of keys in the batch.
 * @param h_keys    (Device Pointer) The keys array of the hash table.
 * @param limit     The total capacity of the hash table.
 * @param num       (Device Pointer) A counter in global memory to be incremented when a key is found.
 */
__global__ void m_has(int* keys, int numKeys,
	int* h_keys, int limit, int* num)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int index;

	if (i < numKeys) {
		
		index = hash_func(keys[i], limit);

		// Traverse the table using linear probing until an empty slot is found.
		while (h_keys[index]) {
			
			// If the key is found, atomically increment the shared counter and exit.
			if (h_keys[index] == keys[i]) {
				atomicAdd(num, 1);
				return;
			}
			index = (index + 1) % limit;
		}
	}
}




/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @details Each thread retrieves the value for one key. It uses linear probing
 *          to find the key. The found value is written back into the input `keys` array
 *          at the corresponding position, which serves as the output mechanism.
 *
 * @param keys      (Device Pointer) Input keys to search for. Also serves as the output
 *                  array for the found values.
 * @param numKeys   The number of keys in the batch.
 * @param h_keys    (Device Pointer) The keys array of the hash table.
 * @param h_values  (Device Pointer) The values array of the hash table.
 * @param limit     The total capacity of the hash table.
 */
__global__ void m_get(int* keys, int numKeys,
	int* h_keys, int* h_values, int limit)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int index;

	if (i < numKeys) {
		
		index = hash_func(keys[i], limit);

		while (true) {
			
			if (h_keys[index] == keys[i]) {
				// Overwrite the input key with the corresponding value from the hash table.
				keys[i] = h_values[index];
				return;
			} else {
				index = (index + 1) % limit;
			}
		}
	}
}


GpuHashTable::GpuHashTable(int size) {
	this->size = size;
	this->curr_size = 0;
	this->resize = 0;
	// Allocate storage for keys and values in global GPU memory.
	cudaMalloc((void **)&this->keys, size * sizeof(int));
	cudaMalloc((void **)&this->values, size * sizeof(int));
	// Allocate a managed memory integer for sharing a counter between host and device.
	cudaMallocManaged((void **)&this->num, sizeof(int));
}


GpuHashTable::~GpuHashTable() {
	// Free all allocated GPU memory.
	cudaFree(this->keys);
	cudaFree(this->values);
	cudaFree(this->num);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	int *aux_keys = this->keys;
	int *aux_values = this->values;
	int old_size = this->size;

	
	// Note: The original code has a syntax error here ('curr_size' missing an operator).
	if (numBucketsReshape < this->curr_size)
		return;

	
	// Allocate new, larger arrays in GPU memory.
	cudaMalloc((void **)&this->keys, numBucketsReshape * sizeof(int));
	cudaMalloc((void **)&this->values, numBucketsReshape * sizeof(int));
	this->size = numBucketsReshape;

	
	// Re-insert all existing elements into the new, larger table.
	if (this->curr_size) {
		this->resize = 1; // Set resize flag to alter insertBatch behavior.
		
		// Call insertBatch to re-hash and insert old elements into the new table.
		insertBatch(aux_keys, aux_values, old_size);
		this->resize = 0;
	}

	// Free the old, smaller GPU memory arrays.
	cudaFree(aux_keys);
	cudaFree(aux_values);
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *aux_keys, *aux_values;
	// Calculate the number of thread blocks needed for the kernel launch.
	int num_blocks = (numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE;

	
	// For a normal insert (not a resize), copy data from host to device.
	if (!this->resize) {
		cudaMalloc((void **)&aux_keys, numKeys * sizeof(int));
		cudaMalloc((void **)&aux_values, numKeys * sizeof(int));

		cudaMemcpy(aux_keys, keys,
			numKeys * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(aux_values, values,
			numKeys * sizeof(int), cudaMemcpyHostToDevice);
	} else {
		// For a resize, the data is already on the device.
		aux_keys = keys;
		aux_values = values;
	}

	
	// Before inserting, find how many keys already exist to update curr_size correctly.
	if (!this->resize) {
		*this->num = 0;
		// Note: The original code has a syntax error in the kernel launch.
		m_has>>(aux_keys,
			numKeys, this->keys, this->size, this->num);
		cudaDeviceSynchronize(); // Wait for the kernel to finish.
		this->curr_size += (numKeys - *this->num);
	}

	
	// If the load factor exceeds the threshold, trigger a reshape.
	if (this->loadFactor() >= 0.01f * LOADFACTOR) {
		// Note: The original code has a syntax error (extra newlines) in this calculation.
		this->reshape((this->curr_size * 100) / LOADFACTOR


			+ 0.001f * this->curr_size);
	}

	
	// Launch the insertion kernel.
	// Note: The original code has a syntax error in the kernel launch.
	m_insert>>(aux_keys,
		aux_values, numKeys, this->keys, this->values, this->size);
	cudaDeviceSynchronize(); // Wait for insertion to complete.

	
	if (!this->resize) {
		cudaFree(aux_keys);
		cudaFree(aux_values);
	}
	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *batch;
	int *aux_key_values;
	int num_blocks = (numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE;

	
	// Allocate host memory for the results.
	batch = (int *)malloc(numKeys * sizeof(int));
	if (batch == NULL)
		return NULL;

	
	// Allocate device memory for keys and copy them from host.
	cudaMalloc((void **)&aux_key_values, numKeys * sizeof(int));
	cudaMemcpy(aux_key_values, keys,
		numKeys * sizeof(int), cudaMemcpyHostToDevice);

	
	// Launch the get kernel.
	// Note: The original code has a syntax error in the kernel launch.
	m_get>>(aux_key_values,
		numKeys, this->keys, this->values, this->size);
	cudaDeviceSynchronize();

	// Copy the results (which are in aux_key_values) back to host memory.
	cudaMemcpy(batch, aux_key_values,
		numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(aux_key_values);
	return batch;
}


float GpuHashTable::loadFactor() {
	return (1.0f * this->curr_size) / (1.0f * this->size);
}

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

// This is a highly unconventional practice and can lead to build issues.
// It suggests that test code is being directly included into the source file.
#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0
#define BLOCK_SIZE		256
#define LOADFACTOR		85

// A macro for simple error checking and exiting.
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
 * @brief A simple integer hash function executed on the device.
 * @param data The integer key to hash.
 * @param limit The size of the hash table, used for the final modulo.
 * @return The calculated hash index.
 */
__device__ int hash_func(int data, int limit) {
	return ((long)abs(data) * 1009llu) % 8495693897llu % limit;
}

/**
 * @class GpuHashTable
 * @brief Host-side interface for managing a GPU-accelerated hash table.
 *
 * This class encapsulates the memory management and kernel launch logic
 * for operating the hash table on the GPU.
 */
class GpuHashTable
{
	private:
		int size, curr_size;
		int *keys, *values, *num;
		char resize;

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
