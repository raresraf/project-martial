/**
 * @file gpu_hashtable.cu
 * @brief High-performance GPU-accelerated Hash Table using CUDA.
 * @details Implements a thread-safe (via atomic operations) hash table with linear probing 
 * collision resolution. Supports batch operations and dynamic resizing (reshaping).
 * 
 * Domain: Parallel Computing, GPGPU, Data Structures.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "gpu_hashtable.hpp"

/**
 * @brief CUDA Kernel for batch insertion of key-value pairs.
 * 
 * Algorithm: Open addressing with linear probing.
 * Synchronization: Uses atomicCAS (Compare-And-Swap) to ensure thread-safe slot reservation 
 * in global memory.
 * 
 * @param keys Array of source keys to insert.
 * @param values Array of source values to insert.
 * @param numKeys Number of elements in the input batch.
 * @param h_keys Device-side hash table key array.
 * @param h_values Device-side hash table value array.
 * @param limit Total capacity of the hash table.
 */
__global__ void m_insert(int* keys, int* values, int numKeys,
	int *h_keys, int* h_values, int limit)
{
	// Global thread index mapping to input batch element.
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int index;
	int old;

	/**
	 * Block Logic: Conditional insertion for valid keys within bounds.
	 * Pre-condition: Thread index is within batch size and key is non-zero (valid).
	 */
	if (i < numKeys && keys[i] != 0) {
		// Functional Utility: Computes the initial hash bucket.
		index = hash_func(keys[i], limit);

		while (true) {
			/**
			 * Case 1: Key already exists. Update the value in-place.
			 * Invariant: Key-level atomicity is assumed for existing entries.
			 */
			if (h_keys[index] == keys[i]) {
				h_values[index] = values[i];
				return;
			}

			/**
			 * Case 2: Empty slot found. Attempt to claim it atomically.
			 * Logic: atomicCAS ensures only one thread can transition a slot from 0 to keys[i].
			 */
			old = atomicCAS(&h_keys[index], 0, keys[i]);
			if (old == 0) {
				// Slot claimed successfully. Write the value.
				h_values[index] = values[i];
				return;
			} else {
				// Collision: Apply linear probing (step = 1) with wrap-around.
				index = (index + 1) % limit;
			}
		}
	}
}

/**
 * @brief CUDA Kernel for identifying new unique keys in a batch.
 * Functional Utility: Used to calculate the updated 'curr_size' of the table.
 */
__global__ void m_has(int* keys, int numKeys,
	int* h_keys, int limit, int* num)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int index;

	if (i < numKeys) {
		index = hash_func(keys[i], limit);

		/**
		 * Block Logic: Linear search for the key until an empty slot is encountered.
		 */
		while (h_keys[index]) {
			if (h_keys[index] == keys[i]) {
				// Key exists: Increment the match counter atomically.
				atomicAdd(num, 1);
				return;
			}
			index = (index + 1) % limit;
		}
	}
}

/**
 * @brief CUDA Kernel for batch retrieval of values associated with input keys.
 * Functional Utility: Implements the 'Get' side of the Map interface.
 */
__global__ void m_get(int* keys, int numKeys,
	int* h_keys, int* h_values, int limit)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int index;

	if (i < numKeys) {
		index = hash_func(keys[i], limit);

		/**
		 * Block Logic: Probes the table to find the requested key.
		 */
		while (true) {
			if (h_keys[index] == keys[i]) {
				// Functional Utility: In-place replacement of keys with retrieved values.
				keys[i] = h_values[index];
				return;
			} else {
				// Logic: Linear probing for retrieval must follow the same path as insertion.
				index = (index + 1) % limit;
			}
		}
	}
}

/**
 * @brief Constructor for the GPU-based hash table.
 * Memory Hierarchy: Allocates key/value buffers in Device Global Memory.
 */
GpuHashTable::GpuHashTable(int size) {
	this->size = size;
	this->curr_size = 0;
	this->resize = 0;
	cudaMalloc((void **)&this->keys, size * sizeof(int));
	cudaMalloc((void **)&this->values, size * sizeof(int));
	// Zero-copy or managed memory for shared counters.
	cudaMallocManaged((void **)&this->num, sizeof(int));
}

GpuHashTable::~GpuHashTable() {
	cudaFree(this->keys);
	cudaFree(this->values);
	cudaFree(this->num);
}

/**
 * @brief Dynamic reconfiguration of the hash table capacity.
 * Logic: Allocates a new larger buffer and re-inserts all existing elements (re-hashing).
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int *aux_keys = this->keys;
	int *aux_values = this->values;
	int old_size = this->size;

	if (numBucketsReshape < curr_size)
		return;

	// Memory Allocation: New buffers for expanded capacity.
	cudaMalloc((void **)&this->keys, numBucketsReshape * sizeof(int));
	cudaMalloc((void **)&this->values, numBucketsReshape * sizeof(int));
	this->size = numBucketsReshape;

	if (this->curr_size) {
		this->resize = 1;
		// Re-hashing: Batch insertion into the new table structure.
		insertBatch(aux_keys, aux_values, old_size);
		this->resize = 0;
	}

	cudaFree(aux_keys);
	cudaFree(aux_values);
}

/**
 * @brief High-level API for inserting a batch of data from Host to Device.
 * Workflow: Memory transfer -> Load factor check -> Potential Reshape -> Kernel Launch.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *aux_keys, *aux_values;
	int num_blocks = (numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE;

	if (!this->resize) {
		// HtoD Transfer: Copies batch data from system RAM to GPU VRAM.
		cudaMalloc((void **)&aux_keys, numKeys * sizeof(int));
		cudaMalloc((void **)&aux_values, numKeys * sizeof(int));

		cudaMemcpy(aux_keys, keys,
			numKeys * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(aux_values, values,
			numKeys * sizeof(int), cudaMemcpyHostToDevice);
	} else {
		aux_keys = keys;
		aux_values = values;
	}

	// Logic: Pre-calculates the expected size to determine if a resize is mandatory.
	if (!this->resize) {
		*this->num = 0;
		m_has<<<num_blocks, BLOCK_SIZE>>>(aux_keys,
			numKeys, this->keys, this->size, this->num);
		cudaDeviceSynchronize();
		this->curr_size += (numKeys - *this->num);
	}

	// Functional Utility: Proactive resizing based on the LOADFACTOR threshold.
	if (this->loadFactor() >= 0.01f * LOADFACTOR) {
		this->reshape((this->curr_size * 100) / LOADFACTOR
			+ 0.001f * this->curr_size);
	}

	// Execution: Launches the parallel insertion kernel.
	m_insert<<<num_blocks, BLOCK_SIZE>>>(aux_keys,
		aux_values, numKeys, this->keys, this->values, this->size);
	cudaDeviceSynchronize();

	if (!this->resize) {
		cudaFree(aux_keys);
		cudaFree(aux_values);
	}
	return true;
}

/**
 * @brief Retrieves values for a batch of keys from the GPU.
 * @return Dynamically allocated host array containing the results.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *batch;
	int *aux_key_values;
	int num_blocks = (numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE;

	batch = (int *)malloc(numKeys * sizeof(int));
	if (batch == NULL)
		return NULL;

	cudaMalloc((void **)&aux_key_values, numKeys * sizeof(int));
	cudaMemcpy(aux_key_values, keys,
		numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Execution: Launches the parallel lookup kernel.
	m_get<<<num_blocks, BLOCK_SIZE>>>(aux_key_values,
		numKeys, this->keys, this->values, this->size);
	cudaDeviceSynchronize();

	// DtoH Transfer: Copies the found values back to the host.
	cudaMemcpy(batch, aux_key_values,
		numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(aux_key_values);
	return batch;
}

float GpuHashTable::loadFactor() {
	return (1.0f * this->curr_size) / (1.0f * this->size);
}
