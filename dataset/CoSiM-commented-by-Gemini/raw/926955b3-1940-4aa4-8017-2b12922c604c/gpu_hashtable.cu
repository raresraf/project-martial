/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA.
 *
 * This file contains the implementation of a hash table that leverages the parallel
 * processing power of GPUs for batch insert and get operations. It uses linear
 * probing for collision resolution and dynamically resizes when a certain load
 * factor is exceeded.
 *
 * @note This implementation has several issues:
 * - A memory leak: The destructor is empty and does not free the CUDA memory
 *   allocated in the constructor or during reshaping.
 * - The constructor initializes the hash table with size 0, requiring an
 *   immediate reshape.
 * - The class definition is duplicated within this file, which is unconventional.
 */
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>

#include "gpu_hashtable.hpp"

#define dim sizeof(int)


/**
 * @brief Constructor for the GpuHashTable class.
 *
 * Initializes the hash table with a given size. However, the `size` parameter is
 * not used, and the initial capacity (`mod`) is set to 0. Memory is allocated on
 * both the host and device.
 *
 * @param size The intended initial size of the hash table (currently unused).
 */
GpuHashTable::GpuHashTable(int size) {
	mod = 0;
	totMem = 0;
	usedMem = (int *)malloc(dim);
	memset(usedMem, 0, 1);

	// Allocate memory on the GPU for keys, values, and the used memory counter.
	cudaMalloc((void **)&(hashMapKeys), mod * dim);
	cudaMemset((void **)&(hashMapKeys), 0, mod * dim);
		
	cudaMalloc((void **)&(hashMapValues), mod * dim);
	cudaMemset((void **)&(hashMapValues), 0, mod * dim);
	
	cudaMalloc((void **)&(usedMemDev), dim);
	cudaMemset((void **)&(usedMemDev), 0, dim);
	
}


/**
 * @brief Destructor for the GpuHashTable class.
 *
 * @warning This destructor is empty and does not free the allocated CUDA memory,
 * leading to a memory leak.
 */
GpuHashTable::~GpuHashTable() {
}


/**
 * @brief CUDA kernel for inserting a batch of key-value pairs into the hash table.
 *
 * Each thread handles one key-value pair. It uses linear probing to find an empty
 * slot or an existing key. `atomicCAS` is used to ensure thread-safe insertion
 * and updates.
 *
 * @param hashMapKeys Device pointer to the hash table's keys.
 * @param hashMapValues Device pointer to the hash table's values.
 * @param keysDev Device pointer to the batch of keys to insert.
 * @param valuesDev Device pointer to the batch of values to insert.
 * @param numKeys The number of keys in the batch.
 * @param mod The capacity of the hash table.
 * @param usedMemDev Device pointer to the counter for used memory slots.
 * @param hashes Device pointer to the pre-calculated hashes for the keys.
 */
__global__ void insertHelp(int *hashMapKeys, int *hashMapValues,
	int *keysDev, int *valuesDev, int numKeys, int mod, int *usedMemDev, int *hashes) {

	// Each thread processes one key-value pair from the batch.
	int i = threadIdx.x + blockDim.x * blockIdx.x;	

	if (i < numKeys) {
		int j;
		bool f = false;
		
		// Linear probing from the initial hash position to the end of the table.
		for (j = 0 ; j < mod - hashes[i]; j++) {
			
			// Try to atomically insert the key if the slot is empty (0).
			if (atomicCAS(&hashMapKeys[j + hashes[i]], 0, keysDev[i]) == 0) {
				// If successful, increment the used memory count and set the value.
				atomicAdd(usedMemDev, 1);
				hashMapValues[j + hashes[i]] = valuesDev[i];
                f = true;
                break;
			} else if (atomicCAS(&hashMapKeys[j + hashes[i]], 0, keysDev[i]) == keysDev[i]) {
				// If the key already exists, just update the value.
				hashMapValues[j + hashes[i]] = valuesDev[i];
				f = true;
				break;
			}	
		}
		
		// If a slot was not found, wrap around and probe from the beginning of the table.
		if (!f) {
			for (j = -hashes[i] ; j < 0 ; j++) {
				if (atomicCAS(&hashMapKeys[j + hashes[i]], 0, keysDev[i]) == 0) {
					atomicAdd(usedMemDev, 1);
					hashMapValues[j + hashes[i]] = valuesDev[i];
		            f = true;
		            break;
				} else if (atomicCAS(&hashMapKeys[j + hashes[i]], 0, keysDev[i]) == keysDev[i]) {
					hashMapValues[j + hashes[i]] = valuesDev[i];
					f = true;
					break;
				}	
			}
		}
	}
} 


/**
 * @brief Helper function to extract non-zero key-value pairs from host arrays.
 *
 * @param oldmod The size of the arrays.
 * @param nonZeroKeys Output array for non-zero keys.
 * @param nonZeroValues Output array for non-zero values.
 * @param allKeys Input array of all keys.
 * @param allValues Input array of all values.
 * @return The number of non-zero key-value pairs found.
 */
 int getNonZeroKeysValues(int oldmod, int* nonZeroKeys, int *nonZeroValues, int *allKeys, int *allValues)
 {
	int nr = 0, i;
	// Invariant: Iterate through the arrays and copy pairs where the key is not invalid ( > 0).
	for (i = 0 ; i < oldmod; i++) {
		if (allKeys[i] > 0) {
			nonZeroKeys[nr] = allKeys[i];
			nonZeroValues[nr++] = allValues[i];
		}
	}
	return nr;
 }




/**
 * @brief Resizes the hash table to a new capacity.
 *
 * This function handles the growth of the hash table. It preserves the existing
 * elements by re-hashing and inserting them into the new, larger table.
 *
 * @param numBucketsReshape The new capacity of the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int oldmod = mod, nr;
	mod = numBucketsReshape;
	totMem = mod;
	
	int *nonZeroKeys, *nonZeroKeysCUDA, *allKeys;	
	int *nonZeroValues, *nonZeroValuesCUDA, *allValues;
	
	// Pre-condition: Only proceed with copying old data if the table is not empty.
	if (usedMem[0] > 0) {

		// Allocate host memory to hold the old data.
		nonZeroKeys = (int *)malloc(oldmod * dim);
		allKeys = (int *)malloc(oldmod * dim);
		cudaMalloc((void**)&(nonZeroKeysCUDA), oldmod * dim);
		
		nonZeroValues = (int *)malloc(oldmod * dim);
		allValues = (int *)malloc(oldmod * dim);	
		cudaMalloc((void**)&(nonZeroValuesCUDA), oldmod * dim);
		
		// Copy old key-value pairs from device to host.
		cudaMemcpy(allKeys, hashMapKeys, oldmod * dim, cudaMemcpyDeviceToHost);
		cudaMemcpy(allValues, hashMapValues, oldmod * dim, cudaMemcpyDeviceTo-Host);
		// Filter out the empty slots.
		nr = getNonZeroKeysValues(oldmod, nonZeroKeys, nonZeroValues, allKeys, allValues);
		
		// Copy the filtered non-zero pairs to a new device buffer.
		cudaMemcpy(nonZeroKeysCUDA, nonZeroKeys, oldmod * dim,
			cudaMemcpyHostToDevice);
		cudaMemcpy(nonZeroValuesCUDA, nonZeroValues, oldmod * dim,
			cudaMemcpyHostToDevice);
	} 

	
	// Allocate new, larger memory for the hash table on the device.
	cudaMalloc((void **)&(hashMapKeys), mod * dim);
    cudaMemset((void **)&(hashMapKeys), 0, mod * dim);
    
    cudaMalloc((void **)&(hashMapValues), mod * dim);
    cudaMemset((void **)&(hashMapValues), 0, mod * dim);

	// Pre-condition: If there were elements in the old table, re-insert them.
	if (usedMem[0] > 0) {
		// Reset the used memory counter.
		usedMem[0] = 0;
		cudaMemcpy(usedMemDev, usedMem, dim, cudaMemcpyHostToDevice);
    
    	int *hashes = (int *)calloc(nr, sizeof(int));
	
		// Re-hash the old keys with the new capacity.
		int i;
		for (i = 0 ; i < nr ; i++) {
			hashes[i] = myHash(nonZeroKeys[i], mod);
		}
		
		int *hashesCUDA;
		cudaMalloc((void **)&hashesCUDA, nr * dim);
		cudaMemcpy(hashesCUDA, hashes, nr * dim, cudaMemcpyHostToDevice);
    
    	
		// Launch the kernel to re-insert the old elements into the new table.
		insertHelp<<<1, 1>>>(hashMapKeys, hashMapValues,
			nonZeroKeysCUDA, nonZeroValuesCUDA, nr, mod, usedMemDev, hashesCUDA);
		
		cudaDeviceSynchronize();
		// Update the host-side used memory count.
		cudaMemcpy(usedMem, usedMemDev, dim, cudaMemcpyDeviceToHost);
	}
}



/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 *
 * If the insertion would cause the load factor to exceed a threshold, the table
 * is resized first.
 *
 * @param keys Pointer to the host array of keys.
 * @param values Pointer to the host array of values.
 * @param numKeys The number of pairs to insert.
 * @return True on success.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	// Pre-condition: Check if the table needs to be resized to maintain a low load factor.
	if (usedMem[0] + numKeys + (usedMem[0] + numKeys) * 0.05f > totMem) {
		// Logic: Grow the table by about 15% of the new required size.
		reshape(usedMem[0] + numKeys + (usedMem[0] + numKeys) * 0.15f);
	}

	int *hashes = (int *)calloc(numKeys, sizeof(int));
	
	
	// Calculate hashes on the host.
	int i;
	for (i = 0 ; i < numKeys ; i++) {
		hashes[i] = myHash(keys[i], mod);
	}
	
	// Copy hashes and key-value data to the device.
	int *hashesCUDA;
	cudaMalloc((void **)&hashesCUDA, numKeys * dim);
	cudaMemcpy(hashesCUDA, hashes, numKeys * dim, cudaMemcpyHostToDevice);

	int *keysCUDA = 0;
	cudaMalloc((void **)&keysCUDA, numKeys * dim);
	cudaMemcpy(keysCUDA, keys, numKeys * dim, cudaMemcpyHostToDevice);
	
	int *valuesCUDA = 0;
	cudaMalloc((void **)&valuesCUDA, numKeys * dim);
	cudaMemcpy(valuesCUDA, values, numKeys * dim, cudaMemcpyHostToDevice);

	// Launch the insertion kernel.
	insertHelp<<<1, 1>>>(hashMapKeys, hashMapValues,
		keysCUDA, valuesCUDA, numKeys, mod, usedMemDev, hashesCUDA);
	
	cudaMemcpy(usedMem, usedMemDev, dim, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	return true;
}




/**
 * @brief CUDA kernel to retrieve a batch of values corresponding to a batch of keys.
 *
 * Each thread handles one key. It uses linear probing to find the key in the hash table.
 *
 * @param hashMapKeys Device pointer to the hash table's keys.
 * @param hashMapValues Device pointer to the hash table's values.
 * @param retCUDA Device pointer to the output array for the retrieved values.
 * @param keysDev Device pointer to the batch of keys to look up.
 * @param numKeys The number of keys in the batch.
 * @param mod The capacity of the hash table.
 * @param hashes Device pointer to the pre-calculated hashes for the keys.
 */
__global__ void helperGetBatch(int *hashMapKeys, int *hashMapValues, int *retCUDA,
	int *keysDev, int numKeys, int mod, int *hashes) {

	// Each thread processes one key from the batch.
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < numKeys) {
		int j;
		bool f = false;
		
		// Linear probing from the initial hash position to the end of the table.
		for (j = 0 ; j < mod - hashes[i]; j++) {
			if (keysDev[i] == hashMapKeys[j + hashes[i]]) {
				retCUDA[i] = hashMapValues[j + hashes[i]];
				f = true;
				break;
			}	
		}
		
		// If not found, wrap around and probe from the beginning of the table.
		if (!f) {
			for (j = -hashes[i] ; j < 0 ; j++) {
				if (keysDev[i] == hashMapKeys[j + hashes[i]]) {
					retCUDA[i] = hashMapValues[j + hashes[i]];
					f = true;
					break;
				}	
			}
		}
	}
}


/**
 * @brief Retrieves the values for a batch of keys from the hash table.
 *
 * @param keys Pointer to the host array of keys to look up.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to a host array containing the corresponding values. The caller
 *         is responsible for freeing this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *ret = (int *)malloc(numKeys * dim);
	
	int *hashes = (int *)calloc(numKeys, sizeof(int));
	
	// Calculate hashes on the host.
	int i;
	for (i = 0 ; i < numKeys ; i++) {
		hashes[i] = myHash(keys[i], mod);
	}
	
	// Copy hashes and keys to the device.
	int *hashesCUDA;
	cudaMalloc((void **)&hashesCUDA, numKeys * dim);
	cudaMemcpy(hashesCUDA, hashes, numKeys * dim, cudaMemcpyHostToDevice);
	
	int *keysDev = 0;
	cudaMalloc((void **)&keysDev, numKeys * dim);
	cudaMemcpy(keysDev, keys, numKeys * dim, cudaMemcpyHostToDevice);
	
	int *retCUDA;
	cudaMalloc((void **)&retCUDA, numKeys * dim);


	// Launch the retrieval kernel.
	helperGetBatch<<<1, 1>>>(hashMapKeys, hashMapValues,
		retCUDA, keysDev, numKeys, mod, hashesCUDA);
	// Copy the results back to the host.
	cudaMemcpy(ret, retCUDA, numKeys * dim, cudaMemcpyDeviceToHost);


	cudaDeviceSynchronize();

	return ret;
}


/**
 * @brief Calculates the current load factor of the hash table.
 *
 * @return The load factor as a float (used_slots / total_slots).
 */
float GpuHashTable::loadFactor() {
	return ((float)usedMem[0])/((float)totMem);
}


// A series of macros to simplify the usage of the GpuHashTable API.
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

// Defines the value used to mark an invalid or empty key slot.
#define	KEY_INVALID		0

/**
 * @brief Macro for error checking and exiting on failure.
 */
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
 * @brief A list of large prime numbers used in the hash function.
 */
const size_t primeList[] =
{
	5746614499066534157llu, 7240280573005008577llu, 9122181901073924329llu,
	11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu
};


/**
 * @brief The hash function used for the hash table.
 *
 * @param data The key to hash.
 * @param limit The capacity of the hash table, used for the modulo operation.
 * @return The calculated hash value.
 */
int myHash(int data, int limit) {
	return ((long)abs(data) * primeList[1]) % primeList[3] % limit;
}
	


/**
 * @brief A duplicate or alternative definition of the GpuHashTable class.
 *
 * This re-definition within the same file is unconventional and suggests code
 * organization issues.
 */
class GpuHashTable
{
	public:
		int *hashMapKeys;
		int *hashMapValues;

		int totMem;
		int *usedMem;
		int *usedMemDev;
		int mod;

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