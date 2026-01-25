/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA.
 *
 * This file contains the CUDA device code and C++ host interface for a hash table
 * that operates on the GPU. It uses CUDA kernels for parallel initialization,
 * insertion, retrieval, and reshaping of the hash table. The collision resolution
 * strategy is open addressing with quadratic probing.
 *
 * @b Architecture:
 * - `my_hashTable` and `my_hashElem` structs define the data structure for the hash table on the GPU.
 * - `GpuHashTable` is a C++ class that acts as a host-side wrapper, managing device memory
 *   and CUDA kernel launches.
 * - Kernels like `kernel_insert`, `kernel_get_batch`, and `kernel_rehash` execute in parallel,
 *   with each thread typically handling one element or operation.
 * - Atomic operations (`atomicCAS`, `atomicAdd`) are used to ensure thread-safe updates to the hash table.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_hashtable.hpp"

/**
 * @brief A device function to compute the hash value for a given key.
 * @param key The input key to hash.
 * @param table_capacity The total capacity of the hash table.
 * @return The calculated hash index.
 */
__device__ int hash_function(int key, unsigned long long table_capacity) {
	return (((key * 350745513859007llu) % 3452434812973llu) % table_capacity);
}


/**
 * @brief CUDA kernel to initialize the hash table on the device.
 * @param hashtable A device pointer to the main hash table struct.
 * @param table A device pointer to the array of hash elements.
 * @param size The total capacity of the hash table.
 */
__global__ void kernel_init_table(my_hashTable* hashtable, my_hashElem* table, int size) {
	// This kernel is intended to be launched with a single thread to initialize the main struct.
	hashtable->table = table;
	hashtable->max_items = size;

	printf("SUS DE TOT MAX : %d \n", hashtable->max_items);
	// The initialization of the table memory to zero should be done with cudaMemset for efficiency.
	// This loop is redundant if cudaMemset is used before launching the kernel.
	for (int i = 0; i < size; i++) {
		hashtable->table[i].elem_key = 0;
		hashtable->table[i].elem_value = 0;
	}



	hashtable->curr_nr_items = 0;
}


/**
 * @brief Constructor for the GpuHashTable class.
 * Allocates memory on the GPU for the hash table and initializes it.
 */
GpuHashTable::GpuHashTable(int size) {

	my_hashElem* table = NULL;
	cudaMalloc((void**)&table, size * sizeof(my_hashElem));
	cudaMemset(table, 0, size * sizeof(my_hashElem));

	cudaMalloc((void **)&dev_hash, sizeof(my_hashTable));
	// Launches a kernel with one thread to initialize the hash table structure on the device.
	kernel_init_table > > > (dev_hash, table, size);
	cudaDeviceSynchronize();


}


/**
 * @brief Destructor to free the device memory allocated for the hash table struct.
 * Note: The memory for the table itself (`dev_hash->table`) should also be freed here to avoid leaks.
 */
GpuHashTable::~GpuHashTable() {

	cudaFree(dev_hash);
}

/**
 * @brief CUDA kernel for rehashing the table contents into a new, larger table.
 * Each thread handles one element from the old table.
 */
__global__ void kernel_rehash(my_hashTable* hash, my_hashElem* new_hash, unsigned long long new_max_items) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	// This check prevents out-of-bounds access if not all threads have a valid element.
	if (index >= hash->max_items)
		return;

	my_hashElem item = hash->table[index];
	int key = item.elem_key;
	
	// Only rehash valid elements.
	if(key == 0)
		return;

	int hashed = hash_function(key, new_max_items);
	int i = new_max_items;

	// Quadratic probing to find an empty slot in the new table.
	while (i != 0) {

		// `atomicCAS` ensures that only one thread can claim a slot.
		int aux = atomicCAS(&new_hash[index].elem_key, 0, item.elem_key);
		if (aux == 0) {
			new_hash[index].elem_value = item.elem_value;
			return;
		}
		hashed = (hashed + i * i) % new_max_items; // This is not standard quadratic probing, but a variant.
	}
}


void GpuHashTable::reshape(int numBucketsReshape) {

	int new_size;
	if (OK == 1)
		new_size = numBucketsReshape * 100 / MIN_LOAD;
	else
		new_size = numBucketsReshape;

	unsigned long long curr_max_items;
	cudaMemcpy(&curr_max_items, &dev_hash->max_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	printf("fac reshape cu1 %d \n", new_size);
	
	if (new_size <= curr_max_items)
		new_size = curr_max_items;

	my_hashElem* new_table = NULL;
	cudaMalloc((void**)&new_table, new_size * sizeof(my_hashElem));
	cudaMemset(new_table, 0, new_size);
	printf("fac reshape cu2 %d \n", new_size);

	my_hashTable *hostHashtable = (my_hashTable*)calloc(1, sizeof(my_hashTable));
	cudaMemcpy(hostHashtable, dev_hash, sizeof(my_hashTable), cudaMemcpyDeviceToHost);

	int blocks_nr = curr_max_items / 256;
	if (curr_max_items % 256 != 0) {
		blocks_nr++;
	}



	// Launches the rehash kernel to move elements to the new table.
	kernel_rehash > > > (dev_hash, new_table, new_size);
	cudaDeviceSynchronize();

	cudaFree(hostHashtable->table);
	hostHashtable->table = new_table;
	hostHashtable->max_items = numBucketsReshape;
	cudaMemcpy(dev_hash, hostHashtable, sizeof(my_hashTable), cudaMemcpyHostToDevice);

	free(hostHashtable);
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * Each thread handles the insertion of one key-value pair.
 */
__global__ void kernel_insert(my_hashTable* hash, int* keys, int* values, int numKeys) {

	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index >= numKeys)
		return;

	int key = keys[index];
	int value = values[index];

	int hashed = hash_function(key, hash->max_items);
	

	my_hashElem item;
	item.elem_key = key;
	item.elem_value = value;

	int i = hash->max_items;
	

	// Quadratic probing to find a slot.
	while (i != 0) {
		
		// Use atomicCAS to handle race conditions during insertion.
		int aux = atomicCAS(&(hash->table[hashed].elem_key), 0, item.elem_key);
		
		
		// If the original key was the same, this is an update.
		if (aux == keys[index]) {
			hash->table[hashed].elem_value = value;
			
			return;
		}
		// If the slot was empty (0), the insertion is successful.
		else if (aux == 0) {
			atomicAdd(&hash->curr_nr_items, 1);
			hash->table[hashed].elem_value = value;
			
			return;
		}

		hashed = (hashed + i * i) % hash->max_items; // Variant of quadratic probing.
		i--;
	}
}

/**
 * @brief Checks if the current load factor plus a new batch size exceeds the maximum load factor.
 */
bool GpuHashTable::check_l(unsigned long long batchSize) {

	float loadFactor;
	unsigned long long max_items, curr_nr_items;

	cudaMemcpy(&max_items, &dev_hash->max_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(&curr_nr_items, &dev_hash->curr_nr_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	printf("CHECK %d \n", max_items);
	loadFactor = 1.0f * (curr_nr_items + batchSize) / max_items;

	if (loadFactor * 100 > MAX_LOAD)
		return false;

	return true;
}

/**
 * @brief Inserts a batch of keys and values into the hash table.
 * If the insertion causes the load factor to exceed a threshold, it triggers a reshape.
 */
bool GpuHashTable::insertBatch(int* keys, int* values, int numKeys) {

	unsigned long long max_items;
	cudaMemcpy(&max_items, &(dev_hash->max_items), sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	printf("ORIGINAL MAX ITEMS %d\n", max_items);

	// Pre-condition: Check if resizing is needed before batch insertion.
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

	kernel_insert > > > (dev_hash, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize();

	return true;
}

/**
 * @brief CUDA kernel to retrieve a batch of values corresponding to a given set of keys.
 * Each thread retrieves one value.
 */
__global__ void kernel_get_batch(my_hashTable* hash, int* keys, int* values, int numKeys) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index >= numKeys)
		return;

	int key = keys[index];
	int hashed = hash_function(key, hash->max_items);
	my_hashElem item;

	int i = hash->max_items;
	// Quadratic probing to find the key.
	while (i != 0) {


		item = hash->table[hashed];
		if (item.elem_key == key) {
			values[index] = item.elem_value;
			
			return;
		}
		
		// If we hit an empty slot, the key is not in the table.
		if(item.elem_key == 0) {
			return;
		}
		
		hashed = (hashed + i * i) % hash->max_items;
		i--;
	}
}


/**
 * @brief Retrieves a batch of values for a given set of keys from the GPU hash table.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {

	int* dev_keys;
	cudaMalloc((void**)&dev_keys, numKeys * sizeof(int));
	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int* dev_values;
	cudaMalloc((void**)&dev_values, numKeys * sizeof(int));
	cudaMemset(dev_values, 0, numKeys * sizeof(int));

	int* local_values = (int*)malloc(numKeys * sizeof(int));
	memset(local_values, 0, numKeys);

	int blocks_nr = numKeys / 256;
	if (numKeys % 256 != 0)
		blocks_nr++;

	kernel_get_batch > > > (dev_hash, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize();

	cudaMemcpy(local_values, dev_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	// Memory for dev_keys and dev_values should be freed here to avoid leaks.
	return local_values;
}


/**
 * @brief Calculates and returns the current load factor of the hash table.
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