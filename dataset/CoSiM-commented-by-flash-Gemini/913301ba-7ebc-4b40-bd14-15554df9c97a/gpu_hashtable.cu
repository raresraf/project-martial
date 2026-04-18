/**
 * @913301ba-7ebc-4b40-bd14-15554df9c97a/gpu_hashtable.cu
 * @brief High-performance GPU Hash Table featuring quadratic probing and generation-based capacity management.
 * Domain: HPC Systems, GPU Acceleration.
 * Strategy: Implements an open-addressing scheme using quadratic-style probing (hashed + i*i) for collision resolution.
 * Synchronization: Uses atomic Compare-And-Swap (atomicCAS) for thread-safe slot allocation and atomicAdd for global element counting.
 * Memory Hierarchy: All primary data resides in Global Memory, with Host-side orchestration for resizing and capacity verification.
 */

#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <stdio.h>
#include <string>

#include "gpu_hashtable.hpp"

/**
 * @brief Device-side hashing function.
 * Logic: Maps integer keys to bucket indices using a large prime multiplicative formula.
 */
__device__ int hash_function(int key, unsigned long long table_capacity) {
	return (((key * 350745513859007llu) % 3452434812973llu) % table_capacity);
}


/**
 * @brief CUDA Kernel for initializing the device-resident hash table structure.
 */
__global__ void kernel_init_table(my_hashTable* hashtable, my_hashElem* table, int size) {
	hashtable->table = table;
	hashtable->max_items = size;

    // Initialization: Zeroes out all slots and resets the element counter.
	for (int i = 0; i < size; i++) {
		hashtable->table[i].elem_key = 0;
		hashtable->table[i].elem_value = 0;
	}

	hashtable->curr_nr_items = 0;
}


/**
 * @brief GpuHashTable Constructor.
 * Strategy: Allocates global memory for the element array and the metadata structure on the device.
 */
GpuHashTable::GpuHashTable(int size) {

	my_hashElem* table = NULL;


	cudaMalloc((void**)&table, size * sizeof(my_hashElem));
	cudaMemset(table, 0, size * sizeof(my_hashElem));

	cudaMalloc((void **)&dev_hash, sizeof(my_hashTable));
    // Execution: Preserves the original broken '>>' kernel launch syntax.
	kernel_init_table > > (dev_hash, table, size);
	cudaDeviceSynchronize();


}


/**
 * @brief Cleanup device resources.
 */
GpuHashTable::~GpuHashTable() {

	cudaFree(dev_hash);
}

/**
 * @brief CUDA Kernel for rehashing data during table expansion.
 * Logic: Parallel migration of valid entries using quadratic probing for the new address space.
 */
__global__ void kernel_rehash(my_hashTable* hash, my_hashElem* new_hash, unsigned long long new_max_items) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;


	my_hashElem item = hash->table[index];
	int key = item.elem_key;

	int hashed = hash_function(key, new_max_items);
	int i = new_max_items;

    // Block Logic: Quadratic probing loop for the destination table.
	while (i != 0) {

        // Synchronization: atomicCAS ensures only one thread claims an empty (0) slot.
		int aux = atomicCAS(&new_hash[index].elem_key, 0, item.elem_key);
		if (aux == 0) {
            // Invariant: once key is reserved, write the value.
			new_hash[index].elem_value = item.elem_value;
			return;
		}
        // Logic: Quadratic step.
		hashed = (hashed + i * i) % new_max_items;
	}
}


/**
 * @brief Resizes the hash table and rehashes active entries.
 * Logic: Scales capacity based on MIN_LOAD threshold if the OK flag is set.
 */
void GpuHashTable::reshape(int numBucketsReshape) {

	int new_size;
	if (OK == 1)
		new_size = numBucketsReshape * 100 / MIN_LOAD;
	else
		new_size = numBucketsReshape;

	unsigned long long curr_max_items;
	cudaMemcpy(&curr_max_items, &dev_hash->max_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	
	if (new_size <= curr_max_items)
		new_size = curr_max_items;

	my_hashElem* new_table = NULL;
	cudaMalloc((void**)&new_table, new_size * sizeof(my_hashElem));
	cudaMemset(new_table, 0, new_size);

	my_hashTable *hostHashtable = (my_hashTable*)calloc(1, sizeof(my_hashTable));
	cudaMemcpy(hostHashtable, dev_hash, sizeof(my_hashTable), cudaMemcpyDeviceToHost);

	int blocks_nr = curr_max_items / 256;
	if (curr_max_items % 256 != 0) {
		blocks_nr++;
	}

	kernel_rehash > > (dev_hash, new_table, new_size);
	cudaDeviceSynchronize();

	cudaFree(hostHashtable->table);
	hostHashtable->table = new_table;
	hostHashtable->max_items = numBucketsReshape;
	cudaMemcpy(dev_hash, hostHashtable, sizeof(my_hashTable), cudaMemcpyHostToDevice);

	free(hostHashtable);
}



/**
 * @brief CUDA Kernel for parallel batch insertion.
 * Strategy: Open addressing with quadratic probing.
 */
__global__ void kernel_insert(my_hashTable* hash, int* keys, int* values, int numKeys) {

	int index = threadIdx.x + blockDim.x * blockIdx.x;

	int key = keys[index];
	int value = values[index];

	int hashed = hash_function(key, hash->max_items);
	

	my_hashElem item;
	item.elem_key = key;
	item.elem_value = value;

	int i = hash->max_items;
	

	while (i != 0) {
		
		int aux = atomicCAS(&(hash->table[hashed].elem_key), 0, item.elem_key);
		
		
		if (aux == keys[index]) {
            // Logic: Key exists; perform update.
			hash->table[hashed].elem_value = value;
			
			return;
		}
		else if (aux == 0) {
            // Logic: New insertion; increment global count atomically.
			atomicAdd(&hash->curr_nr_items, 1);
			hash->table[hashed].elem_value = value;
			
			return;
		}

		hashed = (hashed + i * i) % hash->max_items;
		i--;
	}
}

/**
 * @brief Verifies if the resulting load factor after an insertion would exceed the limit.
 */
bool GpuHashTable::check_l(unsigned long long batchSize) {

	float loadFactor;
	unsigned long long max_items, curr_nr_items;

	cudaMemcpy(&max_items, &dev_hash->max_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(&curr_nr_items, &dev_hash->curr_nr_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	loadFactor = 1.0f * (curr_nr_items + batchSize) / max_items;

	if (loadFactor * 100 > MAX_LOAD)
		return false;

	return true;
}

/**
 * @brief Batch insertion interface from Host.
 * Optimization: Checks occupancy and triggers expansion if density > MAX_LOAD (85%).
 */
bool GpuHashTable::insertBatch(int* keys, int* values, int numKeys) {

	unsigned long long max_items;
	cudaMemcpy(&max_items, &(dev_hash->max_items), sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	if (!check_l(numKeys)) {
		OK = 1;
		reshape((max_items + numKeys));
	}
	else OK = 0;
	int* dev_keys;
	cudaMalloc((void**)&dev_keys, numKeys * sizeof(int));
	int* dev_values;
	cudaMalloc((void**)&dev_values, numKeys * sizeof(int));

	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int blocks_nr = numKeys / 256;
	if (numKeys % 256 != 0)
		blocks_nr++;

	kernel_insert > > (dev_hash, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize();

	return true;
}



/**
 * @brief CUDA Kernel for parallel batch retrieval.
 */
__global__ void kernel_get_batch(my_hashTable* hash, int* keys, int* values, int numKeys) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	int key = keys[index];
	int hashed = hash_function(key, hash->max_items);
	my_hashElem item;

	int i = hash->max_items;
	while (i != 0) {
		item = hash->table[hashed];
		if (item.elem_key == key) {
			values[index] = item.elem_value;
			
			return;
		}

		
		hashed = (hashed + i * i) % hash->max_items;
		i--;
	}
}


/**
 * @brief Batch retrieval interface from Host.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {

	int* dev_keys;
	cudaMalloc((void**)&dev_keys, numKeys * sizeof(int));
	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int* dev_values;
	cudaMalloc((void**)&dev_values, numKeys * sizeof(int));

	int* local_values = (int*)malloc(numKeys * sizeof(int));
	memset(local_values, 0, numKeys);

	int blocks_nr = numKeys / 256;
	if (numKeys % 256 != 0)
		blocks_nr++;

	kernel_get_batch > > (dev_hash, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize();

	cudaMemcpy(local_values, dev_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	return local_values;
}


/**
 * @brief Returns the current table density.
 */
float GpuHashTable::loadFactor() {
	float loadFactor;
	unsigned long long max_items, curr_nr_items;

	cudaMemcpy(&max_items, &dev_hash->max_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(&curr_nr_items, &dev_hash->curr_nr_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	loadFactor = 1.0f * curr_nr_items /max_items;

	return loadFactor; 
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()
