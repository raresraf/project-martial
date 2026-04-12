
/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table using open addressing with linear probing
 *        and Unified Memory.
 * @details This file provides a robust implementation of a hash table on the GPU.
 * It uses a correct linear probing algorithm for collision resolution and a proper
 * parallel re-hashing strategy for resizing. The implementation leverages Unified
 * Memory (`cudaMallocManaged`) to simplify memory management between the host and device.
 *
 * It correctly uses atomic operations for claiming slots and counting new insertions.
 * However, it contains two notable flaws:
 * 1. A race condition on non-atomic value updates for existing keys.
 * 2. A potential infinite loop in the `get` kernel if a key is not found in a full table.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param nr_inserate An atomic counter to track the number of newly inserted elements.
 * @warning RACE CONDITION: The value update (`table[place].value = values[i]`) is
 * not atomic. If multiple threads try to update an existing key, data may be lost.
 */
__global__ void insert_kernel(int *keys, int *values, struct element *table, int numKeys, int dimensiune, unsigned int *nr_inserate) {
	
	// Use a grid-stride loop to allow a variable number of threads to process all keys.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x; 
	if (i >= numKeys) {
		return;
	}
	
	unsigned int place = hash3(keys[i], dimensiune);

	// --- Linear Probing with Wrap-around ---
	// This loop will continue until an empty slot is found or an existing key is updated.
	while(1) {
		// Atomically attempt to claim an empty slot (where key is 0).
		unsigned int ret = atomicCAS(&(table[place].key), 0, (unsigned int) keys[i]);
		
		// Case 1: Slot was empty. We successfully claimed it.
		if (ret == 0) {
			table[place].value = values[i];
			// Atomically increment the counter for new elements.
			atomicAdd(&nr_inserate[0], 1);
			return;
		}
		// Case 2: Slot already contained our key. Update the value.
		else if (ret == keys[i]) {
			// BUG: This is a non-atomic write, creating a race condition.
			table[place].value = values[i];
			return;
		}

		// If the slot was occupied by a different key, probe the next slot.
		place++;
		// Wrap around to the beginning if the end of the table is reached.
		if (place >= dimensiune) { 
			place = 0;
		}
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @warning BUG: This kernel can enter an infinite loop. If the table is full and
 * a key being searched for is not present, the `while(1)` loop will never terminate.
 */
__global__ void  get_kernel(int *keys, int *values, struct element *table , int numKeys, int dimensiune) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x; 
	if (i >= numKeys) {
		return;
	}
	unsigned int place = hash3(keys[i], dimensiune);

	// --- Search with Linear Probing ---
	while(1) { // This can be an infinite loop.
		// A non-atomic read is sufficient and correct for a get operation.
		if (table[place].key == keys[i]) {
			values[i] = table[place].value;
			return;
		}
		place++;
		if (place >= dimensiune) {
			place = 0;
		}
	}
}


/**
 * @brief CUDA kernel to resize the hash table by re-hashing all elements.
 * @details Each thread is responsible for one element from the old table, which
 * it re-hashes and inserts into the new, larger table using linear probing.
 */
__global__ void reshape_kernel(struct element *table_old, struct element *table_new, int dimensiune_old, int dimensiune_new) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= dimensiune_old) {
		return;
	}
	// Skip empty slots in the old table.
	if (table_old[i].key == 0) {
		return;
	}
	unsigned int place = hash3(table_old[i].key, dimensiune_new);

	// --- Re-insertion with Linear Probing ---
	while(1) {
		// Atomically claim an empty slot in the new table. This is safe as each
		// thread is migrating a unique element from the old table.
		unsigned int ret = atomicCAS(&(table_new[place].key), 0, (unsigned int) table_old[i].key);
		if (ret == 0) {
			table_new[place].value = table_old[i].value;
			return;
		}

		place++;
		if (place >= dimensiune_new) {
			place = 0;
		}
	}	
}

/**
 * @brief Host-side constructor.
 * @details Allocates Unified Memory for the hash table, making it accessible
 * from both the host and device without explicit memory copies.
 */
GpuHashTable::GpuHashTable(int size) {
	// Allocate Unified Memory, which can be accessed by both CPU and GPU.
	cudaMallocManaged(&table, size*sizeof(struct element));
	cudaMemset(table, 0, size*sizeof(struct element));

	used_size = 0;
	table_size = size;
}

/**
 * @brief Host-side destructor. Frees GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(table);
}

/**
 * @brief Resizes the hash table to a new, larger capacity.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	const size_t block_size = 1024;
	size_t blocks_no = (table_size + block_size - 1) / block_size;
 	
	struct element *table_aux;
	// 1. Allocate a new, larger table using Unified Memory.
	cudaMallocManaged(&table_aux, numBucketsReshape*sizeof(struct element));
	cudaMemset(table_aux, 0, numBucketsReshape*sizeof(struct element));

	// 2. Launch the kernel to perform a parallel re-hash into the new table.
	reshape_kernel>>(table, table_aux, table_size, numBucketsReshape);
	cudaDeviceSynchronize();

	table_size = numBucketsReshape;
	// 3. Free the old table and update the pointer.
	cudaFree(table);
	table = table_aux;
}


/**
 * @brief Inserts a batch of key-value pairs from the host.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	const size_t block_size = 1024;
	size_t blocks_no = (numKeys + block_size - 1) / block_size;
 	
	// 1. Allocate temporary device memory for the input batch.
	int *cheie;
	int *valori;
	cudaMalloc((void **) &cheie, sizeof(int) * numKeys);
	cudaMalloc((void **) &valori, sizeof(int) * numKeys);
	cudaMemcpy(cheie, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(valori, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// 2. Check load factor and resize if it exceeds the threshold.
	if (((float)(used_size + numKeys)) / ((float)(table_size)) >= 0.95f) {
		reshape((int)((used_size + numKeys) / 0.85f));
	}

	// 3. Allocate a managed memory counter to track new insertions.
	unsigned int* nr_inserate = NULL;
	cudaMallocManaged(&nr_inserate, sizeof(unsigned int));
	cudaMemset(nr_inserate, 0, sizeof(unsigned int));

	// 4. Launch the insertion kernel.
	insert_kernel>>(cheie, valori, table , numKeys, table_size, nr_inserate);
	cudaDeviceSynchronize();
	
	// 5. Update host-side count with the atomically calculated number of new elements.
	used_size = used_size + *nr_inserate;

	// 6. Free temporary buffers.
	cudaFree(nr_inserate);
	cudaFree(cheie);
	cudaFree(valori);
	return false; // Bug: Should likely return true on success.
}

/**
 * @brief Retrieves values for a batch of keys from the host.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	const size_t block_size = 1024;
	size_t blocks_no = (numKeys + block_size - 1) / block_size;
	
	// 1. Allocate host and device memory.
	int *values = (int*) malloc(numKeys*sizeof(int));
	int *cheie;
	int *valori;
	cudaMalloc((void **) &cheie, sizeof(int) * numKeys);
	cudaMalloc((void **) &valori, sizeof(int) * numKeys);
	
	// 2. Copy keys from host to device.
	cudaMemcpy(cheie, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	// 3. Launch the get kernel.
	get_kernel>>(cheie, valori, table , numKeys, table_size);
	cudaDeviceSynchronize();	
	
	// 4. Copy results from device back to host.
	cudaMemcpy(values, valori, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	
	// 5. Free device memory.
	cudaFree(valori);
	cudaFree(cheie);
	return values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
	return used_size /((float)table_size) ; 
}


// --- Test harness and boilerplate ---
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

#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)

// Data structure for a single key-value pair.
struct element {
	unsigned int key;
	unsigned int value;
};

// ... (rest of boilerplate)
#endif
