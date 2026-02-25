/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA.
 *
 * @details
 * This file provides the implementation for a hash table that leverages the
 * parallel processing capabilities of a GPU. It supports batch insertion and
 * retrieval of key-value pairs. The collision resolution strategy used is
 * open addressing with linear probing. All hash table data is stored in
 * GPU device memory.
 *
 * The implementation consists of:
 * 1. CUDA kernels (`__global__` functions) that perform the parallel
 *    insert, get, and copy operations.
 * 2. A host-side C++ class (`GpuHashTable`) that encapsulates the hash table,
 *    manages GPU memory, and orchestrates kernel launches.
 *
 * Concurrency during insertion is managed using atomic operations (`atomicCAS`)
 * to ensure that threads safely claim slots in the hash table during linear
 * probing.
 */

#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cerrno>

// It's assumed that "gpu_hashtable.hpp" provides the definition for the
// GpuHashTable class, HashTableEntry struct, and other necessary declarations.
#include "gpu_hashtable.hpp"

// Forward declaration of the device-side hash function.
__device__ int hash_func(int data, int limit);

/**
 * @brief CUDA kernel for batch insertion of key-value pairs.
 * @param hashtable_vec Pointer to the hash table in global GPU memory.
 * @param size The total number of buckets in the hash table.
 * @param keys Device pointer to the array of keys to insert.
 * @param values Device pointer to the array of values to insert.
 * @param numKeys The number of key-value pairs in the batch.
 * @param entries_added Device pointer to a counter for newly added entries.
 *
 * @details
 * Each thread in the grid is responsible for inserting one key-value pair.
 * It uses linear probing to resolve collisions. The `atomicCAS` instruction
 * is used to thread-safely claim an empty bucket. If a key already exists,
 * its value is updated. If a new key is inserted, a global counter is
 * atomically incremented.
 */
__global__ void kernel_insert(HashTableEntry *hashtable_vec, int size,
		int *keys, int *values, int numKeys, unsigned int *entries_added)
{
	// Map global thread ID to the index of the key-value pair to process.
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= numKeys)
		return;

	// Calculate the initial bucket index for the key.
	int hash_index = hash_func(keys[index], size);

	// Linear probing to find a slot.
	// The loop continues until an empty slot is claimed (atomicCAS succeeds)
	// or a slot with the same key is found.
	while(hashtable_vec[hash_index].key != keys[index] &&
		atomicCAS(&(hashtable_vec[hash_index].occupied), 0, 1)) {
		hash_index++;
		hash_index %= size;
	}

	// If we claimed a new slot (the key was not already present),
	// set the key and increment the count of new entries.
	if (hashtable_vec[hash_index].key != keys[index]) {
		hashtable_vec[hash_index].key = keys[index];
		atomicInc(entries_added, numKeys + 1);
	}
	// Set or update the value for the key.
	hashtable_vec[hash_index].value = values[index];
}

/**
 * @brief CUDA kernel to copy and rehash entries into a new, larger table.
 * @param hashtable_vec Pointer to the old hash table in global memory.
 * @param size The size of the old hash table.
 * @param new_hashtable_vec Pointer to the new hash table in global memory.
 * @param new_size The size of the new hash table.
 *
 * @details
 * This kernel is launched during a `reshape` operation. Each thread is
 * responsible for one bucket in the old table. If the bucket is occupied,
 * the thread re-hashes the key-value pair and inserts it into the new table
 * using the same atomic linear probing logic as `kernel_insert`.
 */
__global__ void kernel_copy(HashTableEntry *hashtable_vec, int size,
		HashTableEntry *new_hashtable_vec, int new_size)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= size)
		return;

	HashTableEntry current_entry = hashtable_vec[index];

	// If the bucket in the old table was occupied, copy it to the new one.
	if (current_entry.occupied == 1) {
		int hash_index = hash_func(current_entry.key, new_size);

		// Probe for an empty slot in the new table.
		while (atomicCAS(&(new_hashtable_vec[hash_index].occupied), 0, 1)) {
			hash_index++;
			hash_index %= new_size;
		}

		// Copy the key and value to the new slot.
		new_hashtable_vec[hash_index].key = current_entry.key;
		new_hashtable_vec[hash_index].value = current_entry.value;
	}
}

/**
 * @brief CUDA kernel for batch retrieval of values for given keys.
 * @param hashtable_vec Pointer to the hash table in global memory.
 * @param size The size of the hash table.
 * @param keys Device pointer to the array of keys to look up.
 * @param values Device pointer to the array where found values will be stored.
 * @param numKeys The number of keys in the batch.
 *
 * @details
 * Each thread is responsible for looking up one key. It uses linear probing
 * to scan the table starting from the key's hash index. The scan stops when
 * the key is found or an empty bucket is encountered. This operation is
 * read-only, so no atomic operations are needed.
 */
__global__ void kernel_get(HashTableEntry *hashtable_vec, int size, int *keys,
		int *values, int numKeys)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= numKeys)
		return;
	
	int hash_index = hash_func(keys[index], size);

	// Linearly probe until the key is found or an empty slot is encountered.
	while (hashtable_vec[hash_index].occupied == 1 &&
		keys[index] != hashtable_vec[hash_index].key) {
		hash_index++;
		hash_index %= size;
	}

	// If the key was found in an occupied slot, write the value to the output array.
	if (hashtable_vec[hash_index].occupied == 1)
		values[index] = hashtable_vec[hash_index].value;
}


// --- Host-side GpuHashTable Class Implementation ---

GpuHashTable::GpuHashTable(int initial_size)
{
	// Note: Hardcoding the device ID is inflexible.
	HANDLE_ERROR(cudaSetDevice(1));

	HANDLE_ERROR(
		cudaMalloc(
			&(this->hashtable_vec),
			initial_size * sizeof(HashTableEntry)
		)
	);
	
	// Zero out the newly allocated GPU memory to ensure all 'occupied' flags are 0.
	HANDLE_ERROR(
		cudaMemset(this->hashtable_vec, 0, initial_size * sizeof(HashTableEntry))
	);

	this->size = initial_size;
	this->num_elems = 0;
}

GpuHashTable::~GpuHashTable()
{
	HANDLE_ERROR(cudaFree(this->hashtable_vec));
}

void GpuHashTable::reshape(int numBucketsReshape)
{
	HashTableEntry *new_hashtable_vec;
	// Calculate new size based on a target load factor of 0.8.
	unsigned long new_size = numBucketsReshape / 0.8;
	
	HANDLE_ERROR(
		cudaMalloc(
			&new_hashtable_vec,
			new_size * sizeof(HashTableEntry)
		)
	);
	HANDLE_ERROR(
		cudaMemset(new_hashtable_vec, 0, new_size * sizeof(HashTableEntry))
	);

	// Configure grid dimensions for the copy kernel.
	const size_t block_size = 16;
	size_t blocks_no = this->size / block_size + (this->size % block_size == 0 ? 0 : 1);

	// Launch kernel to rehash and copy all elements to the new table.
	kernel_copy<<>>(
		this->hashtable_vec,
		this->size,
		new_hashtable_vec,
		new_size
	);
	HANDLE_ERROR(cudaDeviceSynchronize());

	// Free the old table and update the object's state.
	HANDLE_ERROR(cudaFree(this->hashtable_vec));
	this->hashtable_vec = new_hashtable_vec;
	this->size = new_size;
}

bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys)
{
	const size_t block_size = 16;
	size_t blocks_no = numKeys / block_size + (numKeys % block_size == 0 ? 0 : 1);
	unsigned int *new_entries_no;
	int *device_keys, *device_values;

	// Resize the table if the new batch would exceed the load factor.
	if (this->num_elems + numKeys >= this->size)
		reshape((this->num_elems + numKeys));

	// Allocate temporary GPU memory for the batch data.
	HANDLE_ERROR(cudaMalloc(&device_keys, numKeys * sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&device_values, numKeys * sizeof(int)));
	// Use managed memory for the counter so it's accessible by both CPU and GPU.
	HANDLE_ERROR(cudaMallocManaged(&new_entries_no, sizeof(unsigned int)));
	*new_entries_no = 0;

	// Copy data from host to device.
	HANDLE_ERROR(cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice));
	
	// Launch the insertion kernel.
	kernel_insert<<>>(
		this->hashtable_vec,
		this.size,
		device_keys,
		device_values,
		numKeys,
		new_entries_no
	);
	HANDLE_ERROR(cudaDeviceSynchronize());

	// Update the host-side element count.
	this->num_elems += *new_entries_no;

	// Clean up temporary device memory.
	HANDLE_ERROR(cudaFree(device_keys));
	HANDLE_ERROR(cudaFree(device_values));
	HANDLE_ERROR(cudaFree(new_entries_no));

	return true;
}

int* GpuHashTable::getBatch(int* keys, int numKeys)
{
	const size_t block_size = 16;
	size_t blocks_no = numKeys / block_size + (numKeys % block_size == 0 ? 0 : 1);
	int *device_keys, *device_values, *host_values;

	// Allocate GPU memory for keys and results.
	HANDLE_ERROR(cudaMalloc(&device_keys, numKeys * sizeof(int)));
	HANDLE_ERROR(cudaMalloc(&device_values, numKeys * sizeof(int)));

	// Copy keys from host to device.
	HANDLE_ERROR(cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice));

	// Allocate host memory for the final results.
	host_values = (int *)calloc(numKeys, sizeof(int));
	if(host_values == NULL) {
		fprintf(stderr, "values get calloc\n");
		exit(ENOMEM);
	}

	// Launch the retrieval kernel.
	kernel_get<<>>(
		this->hashtable_vec,
		this->size,
		device_keys,
		device_values,
		numKeys
	);
	HANDLE_ERROR(cudaDeviceSynchronize());

	// Copy results from device back to host.
	HANDLE_ERROR(cudaMemcpy(host_values, device_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost));

	// Clean up temporary device memory.
	HANDLE_ERROR(cudaFree(device_keys));
	HANDLE_ERROR(cudaFree(device_values));
	
	return host_values;
}

float GpuHashTable::loadFactor()
{
	return (float)this->num_elems / this->size;
}
