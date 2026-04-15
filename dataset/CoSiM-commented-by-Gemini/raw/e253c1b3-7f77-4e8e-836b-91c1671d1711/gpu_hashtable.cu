/**
 * @file gpu_hashtable.cu
 * @brief CUDA-based implementation of a hash table for GPU execution.
 *
 * This file provides a `GpuHashTable` class that manages a hash table entirely in GPU memory.
 * The hash table uses open addressing with linear probing for collision resolution. It is
 * designed to be manipulated in batches from the host, with operations like insertion and
 * retrieval performed in parallel by CUDA kernels. The implementation includes dynamic
 * resizing (rehashing) to maintain performance as the load factor increases.
 *
 * The main components are:
 * - A `GpuHashTable` class to encapsulate the hash table's state and operations.
 * - CUDA kernels for hashing, insertion (`insertKeysandValues`), and retrieval (`getbatch`).
 * - Host-side methods (`insertBatch`, `getBatch`, `reshape`) that orchestrate the GPU operations.
 * - Atomic operations (`atomicCAS`, `atomicAdd`) are used extensively to ensure thread-safe
 *   modifications of the hash table from concurrent threads.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <iostream>

#include "gpu_hashtable.hpp"


/**
 * @brief Constructs a GpuHashTable object.
 * @param size The initial capacity of the hash table.
 *
 * Allocates memory on the GPU for the hash table and initializes it to zero.
 * The actual size allocated is based on the `size` parameter.
 */
GpuHashTable::GpuHashTable(int size)
{
	cudaError_t error;

	// Allocate memory for the hash table on the GPU.
	error = cudaMalloc(&(GpuHashTable::hashtable), size * sizeof(hT));
	DIE(error != cudaSuccess || GpuHashTable::hashtable == NULL, "cudaMalloc hashtable error");
	error = cudaMemset(GpuHashTable::hashtable, 0, size * sizeof(hT));
	DIE(error != cudaSuccess, "cudaMemset hashtable error");

	// Initialize table size properties.
	GpuHashTable::currentTableSize = 0;
	GpuHashTable::tableSize = size;
}



/**
 * @brief Destroys the GpuHashTable object.
 *
 * Frees the GPU memory allocated for the hash table.
 */
GpuHashTable::~GpuHashTable()
{
	cudaError_t error;

	// Free the GPU memory.
	error = cudaFree(GpuHashTable::hashtable);
	DIE(error != cudaSuccess, "cudaFree hashtable error");

	// Reset table size properties.
	GpuHashTable::currentTableSize = 0;
	GpuHashTable::tableSize = 0;
}


/**
 * @brief CUDA kernel to copy valid key-value pairs from the old hash table.
 * @param hashtable      Pointer to the source hash table in GPU memory.
 * @param tableSize      The total size of the source hash table.
 * @param device_keys    Pointer to an array in GPU memory to store keys.
 * @param device_values  Pointer to an array in GPU memory to store values.
 * @param counter        An atomic counter in GPU memory to get a unique index for each thread.
 *
 * This kernel is launched during the `reshape` operation. Each thread checks one slot
 * in the hash table. If the slot contains a valid key (not 0), the thread atomically
 * increments a shared counter to obtain a unique index and copies the key-value pair
 * to the corresponding position in the `device_keys` and `device_values` arrays.
 */
__global__ void copyForReshape(hT *hashtable, int tableSize,
								int *device_keys, int *device_values,
								int *counter)
{
	// Calculate the global thread ID.
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if(idx < tableSize) {
		// Pre-condition: Check if the key at this slot is valid (non-zero).
		if (hashtable[idx].key != 0) {
			// Atomically get a unique index for writing the output.
			int index = atomicAdd(counter, 1);

			// Invariant: Each valid key-value pair is copied to a unique position
			// in the output arrays.
			device_keys[index] = hashtable[idx].key;
			device_values[index] = hashtable[idx].value;
		}
	}
}



/**
 * @brief Resizes the hash table.
 * @param numBucketsReshape The desired number of buckets for the new table.
 *
 * This function handles the process of resizing the hash table, which involves:
 * 1. Allocating a new, larger table on the GPU.
 * 2. If the current table has elements, it launches the `copyForReshape` kernel to
 *    extract all key-value pairs into temporary arrays.
 * 3. These pairs are copied from the GPU back to the host.
 * 4. The old GPU hash table is destroyed.
 * 5. The new table becomes the active hash table.
 * 6. The key-value pairs are re-inserted into the new table from the host using `insertBatch`.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t error;
	hT *newHashTable = NULL;
	// New size is 120% of the required buckets to maintain a reasonable load factor.
	int new_size = 1.2f * numBucketsReshape;

	// Allocate a new hash table on the GPU.
	error = cudaMalloc(&newHashTable, new_size * sizeof(hT));
	DIE(error != cudaSuccess || newHashTable == NULL, "cudaMalloc new hashtable error");
	error = cudaMemset(newHashTable, 0, new_size * sizeof(hT));
	DIE(error != cudaSuccess, "cudaMemset new hashtable error");


	// If the old table has data, we need to rehash it.
	if(GpuHashTable::currentTableSize != 0) {
		const size_t block_size = 1024;
		size_t blocks_no = GpuHashTable::tableSize / block_size;
 
		if (GpuHashTable::tableSize % block_size) 
			++blocks_no;

		int *device_keys = NULL;
		int *counter = NULL;
		int *device_values = NULL;

		// Allocate temporary storage on the GPU for keys and values from the old table.
		error = cudaMalloc(&device_keys, GpuHashTable::currentTableSize * sizeof(int));
		DIE(error != cudaSuccess || device_keys == NULL, "cudaMalloc device_keys error");

		error = cudaMalloc(&device_values, GpuHashTable::currentTableSize * sizeof(int));
		DIE(error != cudaSuccess || device_values == NULL, "cudaMalloc device_values error");

		// Allocate and initialize an atomic counter.
		error = cudaMalloc(&counter, sizeof(int));
		DIE(error != cudaSuccess || counter == NULL, "cudaMalloc counter error");
		error = cudaMemset(counter, 0, sizeof(int));
		DIE(error != cudaSuccess, "cudaMemset counter error");

		// Allocate host memory to bring the data back from the GPU.
		int *host_keys = (int *)malloc(GpuHashTable::currentTableSize * sizeof(int));
		DIE(host_keys == NULL, "malloc host_keys error");

		int *host_values = (int *)malloc(GpuHashTable::currentTableSize * sizeof(int));
		DIE(host_values == NULL, "malloc host_values error");

		
		
		// Launch the kernel to copy data from the old hash table.
		copyForReshape<<<blocks_no, block_size>>>(GpuHashTable::hashtable,
												GpuHashTable::tableSize,
												device_keys, device_values,
												counter);
		error = cudaDeviceSynchronize();
		DIE(error != cudaSuccess, "cudaDeviceSynchronize error");


		// Copy the extracted keys and values back to the host.
		error = cudaMemcpy(host_keys, device_keys,
							GpuHashTable::currentTableSize * sizeof(int),
							cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_keys error");

		error = cudaMemcpy(host_values, device_values,
							GpuHashTable::currentTableSize * sizeof(int),
							cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_values error");


		// Destroy the old hash table.
		GpuHashTable::~GpuHashTable();

		// Set the new hash table as the current one.
		GpuHashTable::tableSize = new_size;
		GpuHashTable::hashtable = newHashTable;
		GpuHashTable::currentTableSize = 0;
		
		// Get the exact number of keys that were copied.
		int numKeys = 0;
		error = cudaMemcpy(&numKeys, counter, sizeof(int), cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy numKeys error");
		
		// Re-insert all the old keys into the new table.
		insertBatch(host_keys, host_values, numKeys);

		// Clean up temporary allocations.
		error = cudaFree(device_keys);
		DIE(error != cudaSuccess, "cudaFree device_keys error");

		error = cudaFree(device_values);
		DIE(error != cudaSuccess, "cudaFree device_values error");

		error = cudaFree(counter);
		DIE(error != cudaSuccess, "cudaFree counter error");

		free(host_keys);
		free(host_values);

		return;

	}

	// If the old table was empty, just replace it with the new one.
	GpuHashTable::~GpuHashTable();

	GpuHashTable::tableSize = new_size;

	GpuHashTable::hashtable = newHashTable;
}



/**
 * @brief CUDA kernel to compute hash codes for a batch of keys.
 * @param keys      An array of keys in GPU memory.
 * @param hashcodes An array in GPU memory to store the computed hash codes.
 * @param numkeys   The number of keys in the batch.
 * @param tablesize The total size of the hash table.
 *
 * Each thread computes the hash for a single key in parallel.
 * The hash function is a simple multiplicative hash.
 */
__global__ void getHashCode(int *keys, int *hashcodes, int numkeys, int tablesize)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numkeys)
		// Multiplicative hashing with large prime numbers to distribute keys.
		hashcodes[idx] = (((long)keys[idx]) * 1402982055436147llu)
								% 452517535812813007llu % tablesize;
}




/**
 * @brief CUDA kernel to insert a batch of keys and values into the hash table.
 * @param hashcodes         Initial hash codes for each key.
 * @param keys              The keys to insert.
 * @param values            The values corresponding to the keys.
 * @param numKeys           The number of keys in the batch.
 * @param hashtable         The hash table.
 * @param currentTableSize  Atomic counter for the number of elements in the table.
 * @param tablesize         The total size of the hash table.
 * @param counter           Atomic counter to track the number of successful insertions in this batch.
 *
 * This kernel implements the core logic for insertion with linear probing. Each thread
 * attempts to insert one key.
 * 1. It uses `atomicCAS` (Compare-And-Swap) to try to claim an empty slot (where key is 0).
 * 2. If the slot is empty, it inserts the key and value and marks the key as processed by setting it to 0.
 * 3. If the slot contains the same key, it just updates the value.
 * 4. If the slot is occupied by a different key (a collision), it increments the hash code
 *    (linear probing) to try the next slot in the subsequent kernel launch. The host-side
 *    `insertBatch` function calls this kernel in a loop until all keys are inserted.
 */
__global__ void insertKeysandValues(int *hashcodes, int *keys, int *values,
									int numKeys, hT *hashtable,
									int *currentTableSize, int tablesize,
									int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numKeys) {
		// Pre-condition: If the key is 0, it has already been inserted in a previous iteration.
		if (keys[idx] == 0)
			return;

		// Try to insert the key using an atomic Compare-And-Swap.
		int key = atomicCAS(&hashtable[hashcodes[idx]].key, 0, keys[idx]);
		
		// Case 1: The slot was empty (key was 0). Insertion successful.
		if (key == 0) {
			hashtable[hashcodes[idx]].value = values[idx];
			
			// Mark this key as processed by setting it to 0.
			keys[idx] = 0;
			atomicAdd(currentTableSize, 1);
			atomicAdd(counter, 1);
		// Case 2: The slot already contained the same key. Update successful.
		} else if (key == keys[idx]) {
			// Update the value for the existing key.
			hashtable[hashcodes[idx]].value = values[idx];
			atomicAdd(counter, 1);
			
			// Mark this key as processed.
			keys[idx] = 0;
		// Case 3: The slot was occupied by a different key (collision).
		} else {
			// Linear probing: increment the hash index to try the next slot.
			// The loop in the host-side `insertBatch` function will re-launch this kernel.
			hashcodes[idx] += 1;
			hashcodes[idx] %= tablesize;
		}
	}
}





/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys      An array of keys on the host.
 * @param values    An array of values on the host.
 * @param numKeys   The number of pairs to insert.
 * @return `true` on success.
 *
 * This host function orchestrates the batch insertion process:
 * 1. Checks if the hash table needs to be resized and calls `reshape` if necessary.
 * 2. Copies the keys and values from host memory to GPU memory.
 * 3. Launches `getHashCode` kernel to compute the initial hash position for each key.
 * 4. Enters a loop that repeatedly calls `insertKeysandValues` kernel. This loop
 *    implements the iterative nature of linear probing. The kernel handles collisions
 *    by updating the hash codes, and the loop continues until all keys in the batch
 *    have been successfully inserted.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys = NULL;
	int *device_values = NULL;
	cudaError_t error;


	// Check if the table needs to be resized to accommodate the new keys.
	if ((GpuHashTable::currentTableSize + numKeys) > GpuHashTable::tableSize)
		reshape((GpuHashTable::currentTableSize + numKeys));


	// Allocate GPU memory for keys and values and copy them from the host.
	error = cudaMalloc(&device_keys, numKeys * sizeof(int));
	DIE(error != cudaSuccess || device_keys == NULL, "cudaMalloc device_keys error");

	error = cudaMalloc(&device_values, numKeys * sizeof(int));


	DIE(error != cudaSuccess || device_values == NULL, "cudaMalloc device_values error");


	error = cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(error != cudaSuccess, "cudaMemcpy keys error");

	error = cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(error != cudaSuccess, "cudaMemcpy values error");


	// Allocate GPU memory for hashcodes and compute them in parallel.
	int *hashcodes = NULL;
	error = cudaMalloc(&hashcodes, numKeys * sizeof(int));
	DIE(error != cudaSuccess || hashcodes == NULL, "cudaMalloc hashcodes error");

	const size_t block_size = 1024;
	size_t blocks_no = numKeys / block_size;
 
	if (numKeys % block_size) 
		++blocks_no;

	getHashCode<<<blocks_no, block_size>>>(device_keys, hashcodes,
											numKeys, GpuHashTable::tableSize);
	error = cudaDeviceSynchronize();
	DIE(error != cudaSuccess, "cudaDeviceSynchronize error");



	// Allocate and initialize atomic counter for the total number of elements.
	int *device_current = NULL;
	error = cudaMalloc(&device_current, sizeof(int));
	DIE(error != cudaSuccess, "cudaMalloc device_current error");
	error = cudaMemset(device_current, 0, sizeof(int));
	DIE(error != cudaSuccess, "cudaMemset device_current error");



	// Allocate and initialize an atomic counter for this batch insertion.
	int *device_counter = NULL;
	error = cudaMalloc(&device_counter, sizeof(int));
	DIE(error != cudaSuccess || device_counter == NULL, "cudaMalloc device_counter error");
	error = cudaMemset(device_counter, 0, sizeof(int));
	DIE(error != cudaSuccess, "cudaMemset device_counter error");


	int host_counter = 0;
	int old_counter = 0;
	// Loop until all keys in the batch are inserted.
	// This loop handles collisions by repeatedly calling the insertion kernel.
	while(1) {
		// Launch the insertion kernel.
		insertKeysandValues<<<blocks_no, block_size>>>(hashcodes, device_keys,
													device_values, numKeys,
													GpuHashTable::hashtable,
													device_current,
													GpuHashTable::tableSize,
													device_counter);
		error = cudaDeviceSynchronize();
		DIE(error != cudaSuccess, "cudaDeviceSynchronize error");

		// Check how many keys were inserted in this iteration.
		old_counter = host_counter;
		error = cudaMemcpy(&host_counter, device_counter, sizeof(int), cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_counter error");

		// Update the total size of the hash table.
		GpuHashTable::currentTableSize += host_counter - old_counter;

		// If the counter equals the number of keys, all have been inserted.
		if(host_counter == numKeys)
			break;
	}
	

	// Free all temporary GPU memory.
	error = cudaFree(device_keys);
	DIE(error != cudaSuccess, "cudaFree device_keys error");

	error = cudaFree(device_values);
	DIE(error != cudaSuccess, "cudaFree device_values error");

	error = cudaFree(hashcodes);


	DIE(error != cudaSuccess, "cudaFree hashcodes error");

	error = cudaFree(device_current);
	DIE(error != cudaSuccess, "cudaFree device_current error");

	error = cudaFree(device_counter);
	DIE(error != cudaSuccess, "cudaFree device_counter error");

	return true;
}




/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param values     Output array to store the retrieved values.
 * @param hashcodes  The hash codes for the keys.
 * @param keys       The keys to look up.
 * @param numKeys    The number of keys in the batch.
 * @param hashtable  The hash table.
 * @param tablesize  The total size of the hash table.
 * @param counter    An atomic counter to track the number of found keys.
 *
 * This kernel implements the lookup logic with linear probing. Each thread searches for
 * one key.
 * 1. It checks the key at the given hash index.
 * 2. If the key matches, it copies the value to the output array and marks the key as found.
 * 3. If it does not match, it increments the hash index (linear probing) to check the
 *    next slot in the subsequent kernel launch. The host-side `getBatch` function calls
 *    this kernel in a loop.
 */
__global__ void getbatch(int *values, int *hashcodes, int *keys, int numKeys,
						hT *hashtable, int tablesize, int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numKeys) {
		// Pre-condition: If key is 0, it has been found in a previous iteration.
		if (keys[idx] == 0)
			return;

		// Check if the key at the current hash position matches.
		if(hashtable[hashcodes[idx]].key == keys[idx]) {
			// Key found, copy the value and mark as processed.
			values[idx] = hashtable[hashcodes[idx]].value;
			atomicAdd(counter, 1);
			keys[idx] = 0;
		} else {
			// Key not found, apply linear probing for the next iteration.
			hashcodes[idx] += 1;
			hashcodes[idx] %= tablesize;
		}
	}
}





/**
 * @brief Retrieves the values for a batch of keys.
 * @param keys      An array of keys on the host to look up.
 * @param numKeys   The number of keys to look up.
 * @return A pointer to an array on the host containing the retrieved values.
 *
 * This host function orchestrates the batch retrieval process:
 * 1. Copies the keys from host to GPU memory.
 * 2. Launches the `getHashCode` kernel to compute initial hash positions.
 * 3. Enters a loop that repeatedly calls the `getbatch` kernel. This iterative
 *    process handles linear probing for keys that were not found at their initial
 *    hash position. The loop continues until all keys are found.
 * 4. Copies the retrieved values from GPU back to host memory and returns them.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *ret_values = (int *)malloc(numKeys * sizeof(int));
	cudaError_t error;

	// Allocate GPU memory for the results.
	int *device_ret_values = NULL;
	error = cudaMalloc(&device_ret_values, numKeys * sizeof(int));
	DIE(error != cudaSuccess || device_ret_values == NULL, "cudaMalloc device_ret_values error");

	// Allocate GPU memory for keys and copy from host.
	int *device_keys = NULL;
	error = cudaMalloc(&device_keys, numKeys * sizeof(int));
	DIE(error != cudaSuccess || device_keys == NULL, "cudaMalloc device_keys error");
	error = cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(error != cudaSuccess, "cudaMemcpy keys error");


	// Allocate GPU memory for hashcodes and compute them.
	int *hashcodes = NULL;

	error = cudaMalloc(&hashcodes, numKeys * sizeof(int));
	DIE(error != cudaSuccess || hashcodes == NULL, "cudaMalloc hashcodes error");

	const size_t block_size = 1024;
	size_t blocks_no = numKeys / block_size;
 
	if (numKeys % block_size) 
		++blocks_no;

	getHashCode<<<blocks_no, block_size>>>(device_keys, hashcodes, numKeys, GpuHashTable::tableSize);
	error = cudaDeviceSynchronize();
	DIE(error != cudaSuccess, "cudaDeviceSynchronize error");


	// Allocate and initialize an atomic counter for the number of found keys.
	int *device_counter = NULL;
	error = cudaMalloc(&device_counter, sizeof(int));
	DIE(error != cudaSuccess || device_counter == NULL, "cudaMalloc device_counter error");
	error = cudaMemset(device_counter, 0, sizeof(int));
	DIE(error != cudaSuccess, "cudaMemset device_counter error");

	// Loop until all keys are found.
	while(1) {
		getbatch<<<blocks_no, block_size>>>(device_ret_values, hashcodes,
											device_keys, numKeys,
											GpuHashTable::hashtable,
											GpuHashTable::tableSize, device_counter);
		error = cudaDeviceSynchronize();
		DIE(error != cudaSuccess, "cudaDeviceSynchronize error");

		int host_counter = 0;
		error = cudaMemcpy(&host_counter, device_counter, sizeof(int), cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_counter error");
		
		// If the counter equals the number of keys, all have been found.
		if(host_counter == numKeys)
			break;
	}


	
	// Copy the results from GPU back to host.
	error = cudaMemcpy(ret_values, device_ret_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	DIE(error != cudaSuccess, "cudaMemcpy ret_values error");


	// Free all temporary GPU memory.
	error = cudaFree(device_keys);
	DIE(error != cudaSuccess, "cudaFree device_keys error");

	error = cudaFree(device_ret_values);
	DIE(error != cudaSuccess, "cudaFree device_ret_values error");

	error = cudaFree(hashcodes);
	DIE(error != cudaSuccess, "cudaFree hashcodes error");

	error = cudaFree(device_counter);
	DIE(error != cudaSuccess, "cudaFree device_counter error");

	return ret_values;
}



/**
 * @brief Calculates the load factor of the hash table.
 * @return The load factor (number of elements / total capacity).
 */
float GpuHashTable::loadFactor() {
	return ((float)GpuHashTable::currentTableSize) / ((float)GpuHashTable::tableSize);
}


// The following macros and included file appear to be a test harness for the GpuHashTable.
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

// Represents an invalid key.
#define	KEY_INVALID		0

// Macro for error checking and exiting on failure.
#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)
	
// A list of large prime numbers used in hashing functions to help distribute keys.
const size_t primeList[] =
{
	2llu, 3llu, 5llu, 7llu, 11llu, 13llu, 17llu, 23llu, 29llu, 37llu, 47llu,
	59llu, 73llu, 97llu, 127llu, 151llu, 197llu, 251llu, 313llu, 397llu,
	499llu, 631llu, 797llu, 1009llu, 1259llu, 1597llu, 2011llu, 2539llu,
	3203llu, 4027llu, 5087llu, 6421llu, 8089llu, 10193llu, 12853llu, 16193llu,
	20399llu, 25717llu, 32401llu, 40823llu, 51437llu, 64811llu, 81649llu,
	102877llu, 129607llu, 163307llu, 205759llu, 259229llu, 326617llu,
	411527llu, 518509llu, 653267llu, 823117llu, 1037059llu, 1306601llu,
	1646237llu, 2074129llu, 2613229llu, 3292489llu, 4148279llu, 5226491llu,
	6584983llu, 8296553llu, 10453007llu, 13169977llu, 16593127llu, 20906033llu,
	26339969llu, 33186281llu, 41812097llu, 52679969llu, 66372617llu,
	83624237llu, 105359939llu, 132745199llu, 167248483llu, 210719881llu,
	265490441llu, 334496971llu, 421439783llu, 530980861llu, 668993977llu,
	842879579llu, 1061961721llu, 1337987929llu, 1685759167llu, 2123923447llu,
	2675975881llu, 3371518343llu, 4247846927llu, 5351951779llu, 6743036717llu,
	8495693897llu, 10703903591llu, 13486073473llu, 16991387857llu,
	21407807219llu, 26972146961llu, 33982775741llu, 42815614441llu,
	53944293929llu, 67965551447llu, 85631228929llu, 107888587883llu,
	135931102921llu, 171262457903llu, 215777175787llu, 271862205833llu,
	342524915839llu, 431554351609llu, 543724411781llu, 685049831731llu,
	863108703229llu, 1087448823553llu, 1370099663459llu, 1726217406467llu,
	2174897647073llu, 2740199326961llu, 3452434812973llu, 4349795294267llu,
	5480398654009llu, 6904869625999llu, 8699590588571llu, 10960797308051llu,
	13809739252051llu, 17399181177241llu, 21921594616111llu, 27619478504183llu,
	34798362354533llu, 43843189232363llu, 55238957008387llu, 69596724709081llu,
	87686378464759llu, 110477914016779llu, 139193449418173llu,
	175372756929481llu, 220955828033581llu, 278386898836457llu,
	350745513859007llu, 441911656067171llu, 556773797672909llu,
	701491027718027llu, 883823312134381llu, 1113547595345903llu,
	1402982055436147llu, 1767646624268779llu, 2227095190691797llu,
	2805964110872297llu, 3535293248537579llu, 4454190381383713llu,
	5611928221744609llu, 7070586497075177llu, 8908380762767489llu,
	11223856443489329llu, 14141172994150357llu, 17816761525534927llu,
	22447712886978529llu, 28282345988300791llu, 35633523051069991llu,
	44895425773957261llu, 56564691976601587llu, 71267046102139967llu,
	89790851547914507llu, 113129383953203213llu, 142534092204280003llu,
	179581703095829107llu, 226258767906406483llu, 285068184408560057llu,
	359163406191658253llu, 452517535812813007llu, 570136368817120201llu,
	718326812383316683llu, 905035071625626043llu, 1140272737634240411llu,
	1436653624766633509llu, 1810070143251252131llu, 2280545475268481167llu,
	2873307249533267101llu, 3620140286502504283llu, 4561090950536962147llu,
	5746614499066534157llu, 7240280573005008577llu, 9122181901073924329llu,
	11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu
};



// Simple hashing functions for use on the CPU side, likely for comparison or testing.
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

// Defines the structure of a single cell in the hash table, holding a key-value pair.
typedef struct hashtableCell{
	int key;
	int value;
} hT;



/**
 * @brief Declaration of the GpuHashTable class.
 *
 * This class encapsulates the data and methods for the GPU hash table.
 * The method implementations are defined at the beginning of this file.
 */
class GpuHashTable
{
	public:
		hT *hashtable;         // Pointer to the hash table data in GPU memory.
		int tableSize;         // The total capacity of the hash table.
		int currentTableSize;  // The number of elements currently in the table.
		

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
