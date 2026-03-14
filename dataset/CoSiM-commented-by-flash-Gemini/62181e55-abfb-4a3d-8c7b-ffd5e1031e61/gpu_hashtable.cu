/**
 * @62181e55-abfb-4a3d-8c7b-ffd5e1031e61/gpu_hashtable.cu
 * @brief This file implements a GPU-accelerated hash table using CUDA.
 * It provides functionality for inserting, retrieving, and reshaping the hash table
 * using device functions and global kernels optimized for parallel execution.
 * The hash table uses open addressing with linear probing and atomic operations
 * for thread-safe access.
 *
 * Algorithm: CUDA-based hash table with open addressing (linear probing).
 *             Collision resolution: Linear probing.
 *             Concurrency control: `atomicCAS` for key insertion.
 * Time Complexity: Insert/Get: Average O(1), Worst O(N) due to collisions.
 *                  Reshape: O(N) where N is the number of elements.
 * Space Complexity: O(Size) for the hash table, where Size is the number of buckets.
 */

#include <iostream> 
#include <queue>    
#include <stack>    
#include <string>   
#include <vector>   
#include <cuda_runtime.h> 
#include <device_launch_parameters.h> 
#include "gpu_hashtable.hpp"


__device__ long long getHash(int x, int N) {
        if (x < 0) x = -x;
        x = ((x >> 16) ^ x) * HASH_NO;
    	x = ((x >> 16) ^ x) * HASH_NO;
    	x = (x >> 16) ^ x;
		x = x % N;
    	return x;
}


/**
 * @brief CUDA kernel to reshape the hash table.
 * This kernel redistributes existing key-value pairs from an old hash table (h1)
 * to a new, larger hash table (h2). It's used during resize operations to
 * maintain efficient hash table performance.
 *
 * @param h1 The old hash table structure containing existing pairs.
 * @param siz1 The size of the old hash table.
 * @param h2 The new hash table structure to populate.
 * @param siz2 The size of the new hash table.
 */
__global__ void reshape_hashT(hashTable h1, long siz1, hashTable h2, long siz2) {
	int vall, idx = blockIdx.x * blockDim.x + threadIdx.x, key1, key2;
	bool ok = false;
	// Each thread processes one potential pair from the old hash table.
	// Threads corresponding to invalid keys or out-of-bounds indices from the old table return.
	if ((h1.pairs[idx].key == KEY_INVALID) || (siz1 <= idx))
		return;
	
	key2 = h1.pairs[idx].key; // Key to be rehashed.
	vall = getHash(key2, h2.size); // Compute hash for the new table.

	// Block Logic: Linear probing starting from the hashed position (vall) to insert the key into the new table.
	// Invariant: The key-value pair will be inserted at the first available slot or an existing slot if the key is already present.
	for (int i = vall; i < siz2; i++) {
		// Attempt to atomically set the key if the slot is KEY_INVALID.
		key1 = atomicCAS(&h2.pairs[i].key, KEY_INVALID, key2);
		if (key1 == KEY_INVALID) { // If insertion was successful (slot was empty).
			h2.pairs[i].value = h1.pairs[idx].value; // Copy the value.
			ok = true; // Mark as successfully inserted.
			break;     // Exit the loop.
		} 
	}
	
	// Block Logic: If insertion failed in the first linear probe (wrapped around the table),
	// continue probing from the beginning of the table.
	if (!ok) {
		for (int i = 0; i <vall; i++) {
			key1 = atomicCAS(&h2.pairs[i].key, KEY_INVALID, key2);
			if (key1 == KEY_INVALID) {
				h2.pairs[i].value = h1.pairs[idx].value;
				break;
			}
		}
	}
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * Each thread attempts to insert one key-value pair from the input arrays
 * into the hash table using linear probing and atomic operations for thread-safety.
 *
 * @param k Total number of keys to insert (redundant, actual numKeys passed via host).
 * @param keys Pointer to array of keys on device.
 * @param values Pointer to array of values on device.
 * @param h The hash table structure on device.
 * @param siz The size of the hash table.
 */
__global__ void insert_hash(int k, int *keys, int *values, hashTable h, long siz) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x, key, vall;
	
	// Each thread processes one key-value pair from the input batch.
	// Threads with an index greater than or equal to the number of keys to insert return.
	if (k <= idx) return;
	
	vall = getHash(keys[idx], siz); // Compute hash for the current key.

	// Block Logic: Linear probing starting from the hashed position (vall) to insert the key.
	// Invariant: The key-value pair will be inserted at the first available slot or an existing slot if the key is already present.
	for (int i = vall; i < siz; i++) {
		// Attempt to atomically set the key if the slot is KEY_INVALID.
		// If the key is already present (`key == keys[idx]`), update its value.
		key = atomicCAS(&h.pairs[i].key, KEY_INVALID, keys[idx]);
		if (key == KEY_INVALID || key == keys[idx]) {
			h.pairs[i].value = values[idx]; // Set or update the value.
			return; // Insertion successful, thread finishes.
		} 
	}
	
	// Block Logic: If insertion failed in the first linear probe (wrapped around the table),
	// continue probing from the beginning of the table.
	for (int i = 0; i < vall; i++) {
		key = atomicCAS(&h.pairs[i].key, KEY_INVALID, keys[idx]);
		if (key == KEY_INVALID || key == keys[idx]) {
			h.pairs[i].value = values[idx];
			return; // Insertion successful, thread finishes.
		}
	}
}

/**
 * @brief CUDA kernel to retrieve a batch of values for given keys from the hash table.
 * Each thread attempts to retrieve the value for one key from the input array.
 * It uses linear probing to find the key in the hash table.
 *
 * @param k Total number of keys to get (redundant, actual numKeys passed via host).
 * @param keys Pointer to array of keys on device.
 * @param values Pointer to array where retrieved values will be stored on device.
 * @param h The hash table structure on device.
 * @param siz The size of the hash table.
 */
__global__ void get_hash(int k, int *keys, int *values, hashTable h, long siz) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x, vall;
	// Each thread processes one key from the input batch.
	// Threads with an index greater than or equal to the number of keys to get return.
	if (k<=idx) return;
	
	vall = getHash(keys[idx], siz); // Compute hash for the current key.

	// Block Logic: Linear probing starting from the hashed position (vall) to find the key.
	// Invariant: If the key is found, its corresponding value is written to the output array.
	for (int i = vall; i < siz; i++) {
		if (h.pairs[i].key == keys[idx]) { // If the key is found.
			values[idx] = h.pairs[i].value; // Store the associated value.
			return; // Key found, thread finishes.
		} 
	}
	
	// Block Logic: If the key was not found in the first linear probe (wrapped around the table),
	// continue probing from the beginning of the table.
	for (int i = 0; i < vall; i++) {
		if (h.pairs[i].key == keys[idx]) { // If the key is found.
			values[idx] = h.pairs[i].value; // Store the associated value.
			return; // Key found, thread finishes.
		}
	}
}


/**
 * @brief Constructs a new GpuHashTable object.
 * Allocates device memory for the hash table pairs and initializes them to 0 (KEY_INVALID).
 *
 * @param size The initial desired size (number of buckets) for the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	hashT.size = size;
	cntPairs = 0; // Initialize count of stored pairs to 0.
	hashT.pairs = nullptr; // Initialize device pointer before allocation.
	cudaMalloc(&hashT.pairs, size * sizeof(pair)); // Allocate memory on the GPU for key-value pairs.
	cudaMemset(hashT.pairs, 0, size * sizeof(pair)); // Initialize all allocated memory to 0 (KEY_INVALID key, 0 value).
}

/**
 * @brief Destroys the GpuHashTable object.
 * Frees the device memory allocated for the hash table pairs.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashT.pairs); // Free memory allocated on the GPU.
}


/**
 * @brief Reshapes the hash table to a new size.
 * This involves creating a new hash table with `numBucketsReshape` buckets,
 * transferring all existing key-value pairs from the old table to the new one
 * via a CUDA kernel, and then replacing the old table with the new one.
 *
 * @param numBucketsReshape The new size for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	// Calculate grid size for the reshape kernel.
	// Each thread processes one slot from the old hash table.
	int k = hashT.size / THREADS_NO;  
	if (!(hashT.size % THREADS_NO == 0)) k = k + 1; // Adjust grid size if not perfectly divisible.
	
	hashTable newH; // Declare a new hashTable structure.
	newH.size = numBucketsReshape; // Set the size for the new table.

	// Allocate and initialize device memory for the new hash table.
	cudaMalloc(&newH.pairs, numBucketsReshape * sizeof(pair));
	cudaMemset(newH.pairs, 0, numBucketsReshape * sizeof(pair));
	
	// Launch the reshape kernel to transfer data from the old table to the new.
	reshape_hashT<<>>(hashT, hashT.size, newH, newH.size);

	cudaDeviceSynchronize(); // Wait for the kernel to complete.

	cudaFree(hashT.pairs); // Free memory of the old hash table.
	hashT = newH; // Update the hashT member to point to the new hash table.
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * Transfers keys and values from host to device, checks for reshape necessity,
 * launches a CUDA kernel for parallel insertion, and updates the pair count.
 *
 * @param keys Pointer to an array of keys on the host.
 * @param values Pointer to an array of values on the host.
 * @param numKeys The number of key-value pairs in the batch.
 * @return True if the batch insertion process was initiated successfully.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *aKeys, *aValues; // Pointers for device-side arrays of keys and values.
	// Calculate grid size for the insert kernel.
	int k = numKeys / THREADS_NO;
	if (numKeys % THREADS_NO != 0) k++;   
	
	// Calculate the number of pairs after insertion and check against max load factor.
	long nr = cntPairs + numKeys; 
	if (nr / hashT.size >= LOADFACTOR_MAX) reshape((int) (nr / LOADFACTOR_MIN)); // Reshape if load factor exceeds threshold.

	// Allocate device memory for input keys and values.
	cudaMalloc(&aKeys, numKeys * sizeof(int));
	cudaMalloc(&aValues, numKeys * sizeof(int));

	// Copy keys and values from host to device.
	cudaMemcpy(aKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(aValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	// Launch the insert kernel to perform parallel insertion.
	insert_hash<<>>(numKeys, aKeys, aValues, hashT, hashT.size);

	cudaDeviceSynchronize(); // Wait for the kernel to complete.
	cudaFree(aKeys); // Free device memory for input keys.
	cudaFree(aValues); // Free device memory for input values.
	cntPairs += numKeys; // Update the count of pairs in the hash table.
	return true;
}


/**
 * @brief Retrieves a batch of values for given keys from the hash table.
 * Transfers keys from host to device, launches a CUDA kernel to retrieve values
 * in parallel, and then returns a pointer to the device array containing the results.
 *
 * @param keys Pointer to an array of keys on the host for which to retrieve values.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to an array of integers on the device containing the retrieved values.
 *         The caller is responsible for freeing this device memory.
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
	int *aKeys, *valls; // Pointers for device-side arrays of input keys and output values.
	// Calculate grid size for the get_hash kernel.
	int k = numKeys / THREADS_NO;
	if (!(numKeys % THREADS_NO == 0)) k++;
	
	// Allocate device memory for input keys and managed memory for output values.
	cudaMalloc(&aKeys, numKeys * sizeof(int));
	cudaMallocManaged(&valls, numKeys * sizeof(int));

	// Copy keys from host to device.
	cudaMemcpy(aKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	// Launch the get_hash kernel to perform parallel retrieval.
	get_hash<<>>(numKeys, aKeys, valls, hashT, hashT.size);
	cudaDeviceSynchronize(); // Wait for the kernel to complete.
	
	cudaFree(aKeys); // Free device memory for input keys.
	return valls; // Return pointer to device array of retrieved values.
}

/**
 * @brief Calculates the current load factor of the hash table.
 * The load factor is the ratio of the number of stored key-value pairs
 * to the total number of buckets in the hash table.
 *
 * @return A float representing the current load factor. Returns 0 if hashT.size is 0.
 */
float GpuHashTable::loadFactor() {
	if (hashT.size == 0) return 0; // Avoid division by zero.
	return (float(cntPairs) / hashT.size); // Calculate and return load factor.
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp" // Assumed to be a test file, not part of the core hash table implementation.
#ifndef _HASHCPU_
#define _HASHCPU_

#include <iostream> // Included here for fprintf in DIE macro, and potential CPU-side debugging.
#include <errno.h>  // For 'errno' global variable.
#include <string.h> // For 'perror' in DIE macro.

/**
 * @brief Defines the number of threads per block used in CUDA kernels.
 * This is a common configuration parameter for CUDA kernel launches.
 */
#define THREADS_NO 1000

/**
 * @brief Represents an invalid key value.
 * Used to indicate empty or available slots in the hash table.
 */
#define KEY_INVALID 0

/**
 * @brief A constant used in the hash function for bit mixing.
 */
#define HASH_NO 0x45d9f3b

/**
 * @brief Minimum load factor threshold.
 * Used during reshape operations to determine the new table size.
 */
#define LOADFACTOR_MIN 0.8

/**
 * @brief Maximum load factor threshold.
 * If the current load factor exceeds this, the hash table will be reshaped (grown).
 */
#define LOADFACTOR_MAX 0.94

/**
 * @brief Macro for error handling and program termination.
 * If 'assertion' is true, it prints an error message to stderr (including file and line number),
 * followed by a system error description for 'call_description', and then exits the program.
 * @param assertion A boolean expression. If true, the program terminates.
 * @param call_description A string describing the context of the error.
 */
#define DIE(assertion, call_description) \
    do {    \
        if (assertion) {    \
        fprintf(stderr, "(%s, %d): ",    \
        __FILE__, __LINE__);    \
        perror(call_description);    \
        exit(errno);    \
    }    \
} while (0)

/**
 * @brief Structure representing a key-value pair in the hash table.
 */
struct pair {
	int key, value;
};

/**
 * @brief Structure representing the hash table itself on the device.
 * Contains a pointer to an array of key-value pairs and the size of the table.
 */
struct hashTable {
	pair *pairs;
	long size;
};


class GpuHashTable {
	public:
		long cntPairs;
		hashTable hashT;
		GpuHashTable(int size);
		void reshape(int sizeReshape);

		bool insertBatch(int *keys, int *values, int numKeys);
		int *getBatch(int *key, int numItems);
		float loadFactor();

		void occupancy(); // Not implemented in this file.
		void print(std::string info); // Not implemented in this file.
		~GpuHashTable();
};

#endif