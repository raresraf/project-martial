/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 * @details This file defines the device-side (kernel) and host-side logic for a hash table
 * that operates on the GPU. It supports initialization, batch insertion, batch retrieval,
 * and dynamic resizing (rehashing) to manage load factor. The hashing strategy
 * uses quadratic probing for collision resolution.
 *
 * @note The implementation relies on atomic operations (atomicCAS) to handle concurrent
 * insertions from multiple threads safely.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief Computes the hash for a given key.
 * @param key The input key to hash.
 * @param table_capacity The maximum capacity of the hash table, used for the modulo operation.
 * @return The computed hash index within the bounds of the table capacity.
 * @note This is a device function, intended to be called from GPU kernels.
 * It uses a multiplicative hashing scheme with large prime numbers to distribute keys.
 */
__device__ int hash_function(int key, unsigned long long table_capacity) {
	return (((key * 350745513859007llu) % 3452434812973llu) % table_capacity);
}

/**
 * @brief CUDA kernel to initialize the hash table in GPU memory.
 * @param hashtable A pointer to the main hash table structure in device memory.
 * @param table A pointer to the array of hash elements (the buckets) in device memory.
 * @param size The total capacity of the hash table.
 * @note This kernel is intended to be launched with a single thread, as it performs
 * a one-time setup of the hash table structure. It initializes all keys and values to 0.
 */
__global__ void kernel_init_table(my_hashTable* hashtable, my_hashElem* table, int size) {
	// Assign the device pointer for the bucket array to the main hash table struct.
	hashtable->table = table;
	// Set the maximum capacity of the table.
	hashtable->max_items = size;

	printf("SUS DE TOT MAX : %d 
", hashtable->max_items);
	// Invariant: The loop initializes all hash table entries to a default empty state.
	for (int i = 0; i < size; i++) {
		hashtable->table[i].elem_key = 0; // 0 is treated as an invalid/empty key.
		hashtable->table[i].elem_value = 0;
	}

	// Initialize the count of current items to zero.
	hashtable->curr_nr_items = 0;
}


/**
 * @brief Allocates and initializes a GpuHashTable object.
 * @param size The initial capacity of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {

	my_hashElem* table = NULL;

	// Allocate memory on the GPU for the hash table buckets.
	cudaMalloc((void**)&table, size * sizeof(my_hashElem));
	// Set the allocated memory to zero, ensuring all buckets are initially empty.
	cudaMemset(table, 0, size * sizeof(my_hashElem));

	// Allocate memory on the GPU for the main hash table management struct.
	cudaMalloc((void **)&dev_hash, sizeof(my_hashTable));
	// Launch the initialization kernel.
	kernel_init_table > > (dev_hash, table, size);
	// Synchronization: Wait for the initialization kernel to complete before proceeding.
	cudaDeviceSynchronize();


}

/**
 * @brief Destructor for the GpuHashTable, frees allocated GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	// Free the device memory associated with the main hash table struct.
	// Note: The memory for the table itself (`dev_hash->table`) should also be freed.
	// This appears to be a memory leak in the original code.
	cudaFree(dev_hash);
}

/**
 * @brief CUDA kernel to rehash elements from an old table to a new, larger table.
 * @param hash A pointer to the old hash table structure.
 * @param new_hash A pointer to the new, resized array of hash elements.
 * @param new_max_items The capacity of the new hash table.
 *
 * @details Each thread is responsible for moving one element from the old table to the new one.
 * It re-calculates the hash for the element based on the new table size and uses quadratic
 * probing with atomic operations to find an empty slot in the new table.
 */
__global__ void kernel_rehash(my_hashTable* hash, my_hashElem* new_hash, unsigned long long new_max_items) {
	// Thread Indexing: Each thread gets a unique index to process one element from the old table.
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	// Pre-condition: The thread index must be within the bounds of the old table's capacity.
	if (index >= hash->max_items) return;

	my_hashElem item = hash->table[index];
	int key = item.elem_key;
	
	// Pre-condition: Only process valid (non-empty) elements.
	if (key == 0) return;

	// Re-calculate the hash for the key using the new table's capacity.
	int hashed = hash_function(key, new_max_items);
	int i = 0;

	// Block Logic: Quadratic probing to find an empty slot in the new table.
	// Invariant: Continuously probe new slots until an empty one is found or all possibilities are exhausted.
	while (i < new_max_items) {
		// Synchronization: Use atomic Compare-And-Swap (CAS) to claim an empty slot.
		// This prevents race conditions where multiple threads try to write to the same new slot.
		int old_val = atomicCAS(&new_hash[hashed].elem_key, 0, item.elem_key);
		if (old_val == 0) {
			// If the slot was empty (0), we have successfully claimed it. Write the value.
			new_hash[hashed].elem_value = item.elem_value;
			return; // Exit the loop and thread.
		}
		// Collision detected, calculate the next probe index.
		i++;
		hashed = (hashed + i * i) % new_max_items;
	}
}


/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The target number of buckets for the new table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {

	int new_size;
	// Adjust the required size based on the minimum load factor to avoid immediate rehashing.
	if (OK == 1)
		new_size = numBucketsReshape * 100 / MIN_LOAD;
	else
		new_size = numBucketsReshape;

	unsigned long long curr_max_items;
	// Memory Transfer: Copy the current table capacity from device to host.
	cudaMemcpy(&curr_max_items, &dev_hash->max_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	printf("fac reshape cu1 %d 
", new_size);
	
	// Ensure the new size is not smaller than the current size.
	if (new_size <= curr_max_items)
		new_size = curr_max_items;

	// Allocate a new, larger table on the device.
	my_hashElem* new_table = NULL;
	cudaMalloc((void**)&new_table, new_size * sizeof(my_hashElem));
	cudaMemset(new_table, 0, new_size);
	printf("fac reshape cu2 %d 
", new_size);

	// Create a temporary host-side copy of the hash table struct to manage pointers.
	my_hashTable *hostHashtable = (my_hashTable*)calloc(1, sizeof(my_hashTable));
	cudaMemcpy(hostHashtable, dev_hash, sizeof(my_hashTable), cudaMemcpyDeviceToHost);

	// Calculate grid dimensions for the rehash kernel launch.
	int blocks_nr = curr_max_items / 256;
	if (curr_max_items % 256 != 0) {
		blocks_nr++;
	}

	// Launch the rehash kernel to move elements from the old table to the new one.
	kernel_rehash > > (dev_hash, new_table, new_size);
	// Synchronization: Wait for all threads of the rehash kernel to complete.
	cudaDeviceSynchronize();

	// Free the old table memory on the device.
	cudaFree(hostHashtable->table);
	// Update the host-side struct to point to the new table and capacity.
	hostHashtable->table = new_table;
	hostHashtable->max_items = new_size; // The size was potentially adjusted.
	// Memory Transfer: Copy the updated hash table struct back to the device.
	cudaMemcpy(dev_hash, hostHashtable, sizeof(my_hashTable), cudaMemcpyHostToDevice);

	free(hostHashtable);
}



/**
 * @brief CUDA kernel for batch insertion of key-value pairs.
 * @param hash A pointer to the hash table structure in device memory.
 * @param keys An array of keys to insert.
 * @param values An array of corresponding values to insert.
 * @param numKeys The number of keys/values in the batch.
 *
 * @details Each thread handles the insertion of one key-value pair. It uses quadratic probing
 * and atomicCAS to find an available slot or update an existing key.
 */
__global__ void kernel_insert(my_hashTable* hash, int* keys, int* values, int numKeys) {
	
	// Thread Indexing: Each thread gets a unique index corresponding to a key-value pair.
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	// Pre-condition: Ensure the thread index is within the bounds of the input batch.
	if (index >= numKeys) return;

	int key = keys[index];
	int value = values[index];

	// Calculate the initial hash position for the key.
	int hashed = hash_function(key, hash->max_items);
	
	my_hashElem item;
	item.elem_key = key;
	item.elem_value = value;

	int i = 0;
	
	// Block Logic: Loop using quadratic probing to find a slot.
	// Invariant: Probe until an empty slot is found, the key is found, or the table is full.
	while (i < hash->max_items) {
		
		// Synchronization: Atomically attempt to swap 0 with the new key.
		// This is the core of the concurrent insertion logic.
		int old_val = atomicCAS(&(hash->table[hashed].elem_key), 0, item.elem_key);
		
		// Case 1: The key already exists. Update its value.
		if (old_val == key) {
			atomicExch(&hash->table[hashed].elem_value, value); // Atomically update the value.
			return;
		}
		// Case 2: The slot was empty (0). We have successfully inserted the new key.
		else if (old_val == 0) {
			atomicAdd(&hash->curr_nr_items, 1); // Atomically increment the item counter.
			hash->table[hashed].elem_value = value; // Set the value.
			return;
		}
		
		// Case 3: Collision. The slot was occupied by a different key. Continue probing.
		i++;
		hashed = (hashed + i * i) % hash->max_items;
	}
}

/**
 * @brief Checks if the current load factor will exceed the maximum threshold after a batch insertion.
 * @param batchSize The size of the incoming batch.
 * @return `false` if inserting the batch would exceed MAX_LOAD, `true` otherwise.
 */
bool GpuHashTable::check_l(unsigned long long batchSize) {

	float loadFactor;
	unsigned long long max_items, curr_nr_items;

	// Memory Transfer: Copy metadata from device to host for the check.
	cudaMemcpy(&max_items, &dev_hash->max_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(&curr_nr_items, &dev_hash->curr_nr_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	printf("CHECK %d 
", max_items);
	loadFactor = 1.0f * (curr_nr_items + batchSize) / max_items;

	// Pre-condition: Check if adding the new batch exceeds the maximum load factor.
	if (loadFactor * 100 > MAX_LOAD)
		return false;

	return true;
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys A host pointer to an array of keys.
 * @param values A host pointer to an array of values.
 * @param numKeys The number of pairs to insert.
 * @return `true` on successful insertion.
 */
bool GpuHashTable::insertBatch(int* keys, int* values, int numKeys) {

	unsigned long long max_items;
	cudaMemcpy(&max_items, &(dev_hash->max_items), sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	printf("ORIGINAL MAX ITEMS %d
", max_items);

	// Block Logic: Check if a reshape is needed before insertion.
	if (!check_l(numKeys)) {
		OK = 1;
		reshape((max_items + numKeys));
	}
	else OK = 0;
	
	// Allocate device memory for the keys and values.
	int* dev_keys;
	cudaMalloc((void**)&dev_keys, numKeys * sizeof(int));
	int* dev_values;
	cudaMalloc((void**)&dev_values, numKeys * sizeof(int));

	// Memory Transfer: Copy keys and values from host to device.
	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Calculate grid dimensions for the kernel launch.
	int blocks_nr = numKeys / 256;
	if (numKeys % 256 != 0)
		blocks_nr++;

	// Launch the insertion kernel.
	kernel_insert > > (dev_hash, dev_keys, dev_values, numKeys);
	// Synchronization: Wait for the insertion to complete.
	cudaDeviceSynchronize();

	// Free the temporary device memory for keys and values.
	cudaFree(dev_keys);
	cudaFree(dev_values);

	return true;
}


/**
 * @brief CUDA kernel to retrieve a batch of values corresponding to a batch of keys.
 * @param hash A pointer to the hash table structure in device memory.
 * @param keys An array of keys to look up.
 * @param values An array where the retrieved values will be stored.
 * @param numKeys The number of keys to look up.
 *
 * @details Each thread is responsible for looking up one key. It uses the same quadratic probing
 * logic as the insertion kernel to find the key.
 */
__global__ void kernel_get_batch(my_hashTable* hash, int* keys, int* values, int numKeys) {
	// Thread Indexing: Each thread gets a unique index for a key to retrieve.
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	// Pre-condition: Ensure thread index is within bounds.
	if (index >= numKeys) return;

	int key = keys[index];
	// Calculate the initial hash position.
	int hashed = hash_function(key, hash->max_items);
	my_hashElem item;

	int i = 0;
	// Block Logic: Loop using quadratic probing to find the key.
	// Invariant: Probe until the key is found or an empty slot (key=0) is encountered,
	// which signifies the key is not in the table.
	while (i < hash->max_items) {
		// Memory Access: Read one element from global device memory.
		item = hash->table[hashed];
		if (item.elem_key == key) {
			// Key found, store its value and exit.
			values[index] = item.elem_value;
			return;
		}
		if (item.elem_key == 0) {
			// Encountered an empty slot, so the key is not in the table.
			values[index] = 0; // Assuming 0 indicates 'not found'.
			return;
		}

		// Collision, probe next slot.
		i++;
		hashed = (hashed + i * i) % hash->max_items;
	}
	values[index] = 0; // Key not found after exhausting all probes.
}


/**
 * @brief Retrieves values for a batch of keys from the hash table.
 * @param keys A host pointer to an array of keys to look up.
 * @param numKeys The number of keys to retrieve.
 * @return A host pointer to an array containing the retrieved values. The caller is responsible for freeing this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {

	// Allocate device memory and copy keys from host.
	int* dev_keys;
	cudaMalloc((void**)&dev_keys, numKeys * sizeof(int));
	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Allocate device memory for the output values.
	int* dev_values;
	cudaMalloc((void**)&dev_values, numKeys * sizeof(int));
	cudaMemset(dev_values, 0, numKeys * sizeof(int)); // Initialize to 0.

	// Allocate host memory for the final results.
	int* local_values = (int*)malloc(numKeys * sizeof(int));
	memset(local_values, 0, numKeys * sizeof(int));

	// Calculate kernel launch dimensions.
	int blocks_nr = numKeys / 256;
	if (numKeys % 256 != 0)
		blocks_nr++;

	// Launch the retrieval kernel.
	kernel_get_batch > > (dev_hash, dev_keys, dev_values, numKeys);
	// Synchronization: Wait for the kernel to finish.
	cudaDeviceSynchronize();

	// Memory Transfer: Copy the results from device to host.
	cudaMemcpy(local_values, dev_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	// Free device memory.
	cudaFree(dev_keys);
	cudaFree(dev_values);

	return local_values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float (current items / max items).
 */
float GpuHashTable::loadFactor() {
	float loadFactor;
	unsigned long long max_items, curr_nr_items;
	
	// Memory Transfer: Copy metadata from device to host for calculation.
	cudaMemcpy(&max_items, &dev_hash->max_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(&curr_nr_items, &dev_hash->curr_nr_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	// Avoid division by zero if the table has no capacity.
	if (max_items == 0) return 0.0f;

	loadFactor = 1.0f * curr_nr_items /max_items;

	return loadFactor; 
}


// Convenience macros for a simplified external API.
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()
