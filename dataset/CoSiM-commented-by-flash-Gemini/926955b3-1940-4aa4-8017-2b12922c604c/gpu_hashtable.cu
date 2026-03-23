/**
 * @file gpu_hashtable.cu
 * @brief This file implements a GPU-accelerated hash table using NVIDIA CUDA.
 *        It provides functionalities for batch insertion and retrieval of key-value pairs
 *        leveraging parallel processing capabilities of the GPU.
 *
 * Algorithm: The hash table employs an open addressing scheme with linear probing for
 *            collision resolution.
 *            - **Hashing:** A custom hash function `myHash` is used to map keys to initial indices.
 *            - **Insertion (`insertHelp` kernel):** Multiple GPU threads attempt to insert keys concurrently.
 *              `atomicCAS` (Compare-And-Swap) operations ensure thread-safe insertion and collision resolution.
 *              If a collision occurs (target bucket is occupied by a different key), threads linearly probe
 *              for the next available or matching slot.
 *            - **Retrieval (`helperGetBatch` kernel):** Similar to insertion, multiple GPU threads
 *              concurrently search for keys. They hash the key and then linearly probe until
 *              the key is found or an empty slot (indicating the key is not present) is encountered.
 *            - **Reshaping (`reshape` method):** When the hash table's load factor exceeds a threshold,
 *              it is resized. Existing non-zero elements are retrieved from the old table and re-inserted
 *              into a new, larger table.
 *
 * CUDA Specifics:
 * - **Kernels:** `__global__` functions (`insertHelp`, `helperGetBatch`) are executed by
 *   many threads on the GPU. Thread indexing (`threadIdx.x`, `blockDim.x`, `blockIdx.x`)
 *   is used to assign unique work to each thread.
 * - **Atomic Operations:** `atomicCAS` and `atomicAdd` are crucial for maintaining data
 *   consistency in a parallel environment, especially during concurrent insertions
 *   and updates to shared data structures like the hash table.
 * - **Memory Hierarchy:** `hashMapKeys`, `hashMapValues` reside in global device memory.
 *   `usedMemDev` (a counter for used slots) is also in global memory and updated atomically.
 *   Host-device memory transfers (`cudaMemcpy`) are used to move data between CPU and GPU.
 *
 * Time Complexity:
 * - **myHash:** O(1)
 * - **insertBatch (average case):** O(numKeys) for data transfer and kernel launch,
 *   with kernel execution being O(numKeys) assuming few collisions due to good load factor.
 * - **insertBatch (worst case):** Can degrade to O(numKeys * mod) if all keys hash to the same
 *   bucket and linear probing has to scan the entire table.
 * - **reshape:** O(old_mod + num_non_zero_elements_reinsert) as it iterates through the old table
 *   and re-inserts elements.
 * - **getBatch (average case):** O(numKeys) for data transfer and kernel launch,
 *   with kernel execution being O(numKeys) assuming few collisions.
 * - **getBatch (worst case):** Can degrade to O(numKeys * mod) similar to insertion.
 *
 * Space Complexity:
 * - O(mod) on the device for `hashMapKeys` and `hashMapValues`, where `mod` is the hash table size.
 * - O(numKeys) on the host for temporary key/value arrays.
 */

#include <iostream> // For general input/output (e.g., cerr for debug).
#include <vector>   // For std::vector.
#include <string>   // For std::string.
#include <cmath>    // For std::abs.
#include <cstring>  // For memset.
#include <algorithm> // For std::min and std::max

#include "gpu_hashtable.hpp" // Custom header for GpuHashTable class declaration.

#define dim sizeof(int) // Defines the size of an integer in bytes.

/**
 * @class GpuHashTable
 * @brief Manages a hash table on both host (CPU) and device (GPU).
 *        Provides methods for initialization, resizing, batch insertion,
 *        batch retrieval, and querying load factor.
 */
// The class definition is in gpu_hashtable.hpp, so this is the implementation.

/**
 * @brief Constructor for GpuHashTable.
 * @param size Initial suggested size for the hash table.
 * Functional Utility: Initializes device memory for hash map keys and values,
 *                     and related metadata on both host and device.
 */
GpuHashTable::GpuHashTable(int size) {
	mod = 0; // Initialize hash table modulo (size) to 0.
	totMem = 0; // Initialize total allocated memory size to 0.
	usedMem = (int *)malloc(dim); // Allocate host memory for tracking used slots.
	memset(usedMem, 0, 1); // Initialize used slots counter to 0.

	// Allocate device memory for hash map keys and initialize to 0.
	cudaMalloc((void **)&(hashMapKeys), mod * dim);
	cudaMemset((void **)&(hashMapKeys), 0, mod * dim);
		
	// Allocate device memory for hash map values and initialize to 0.
	cudaMalloc((void **)&(hashMapValues), mod * dim);
	cudaMemset((void **)&(hashMapValues), 0, mod * dim);
	
	// Allocate device memory for tracking used slots on device and initialize to 0.
	cudaMalloc((void **)&(usedMemDev), dim);
	cudaMemset((void **)&(usedMemDev), 0, dim);
}

/**
 * @brief Destructor for GpuHashTable.
 * Functional Utility: Frees allocated host and device memory to prevent leaks.
 */
GpuHashTable::~GpuHashTable() {
    // Free device memory for hash map keys and values.
    cudaFree(hashMapKeys);
    cudaFree(hashMapValues);
    cudaFree(usedMemDev);
    // Free host memory.
    free(usedMem);
}


/**
 * @brief CUDA kernel for inserting key-value pairs into the hash table.
 *        Uses atomic operations for thread-safe collision resolution.
 * @param hashMapKeys Device pointer to the array storing hash table keys.
 * @param hashMapValues Device pointer to the array storing hash table values.
 * @param keysDev Device pointer to the batch of keys to insert.
 * @param valuesDev Device pointer to the batch of values to insert.
 * @param numKeys Number of keys in the current batch.
 * @param mod Current size (modulo) of the hash table.
 * @param usedMemDev Device pointer to the counter for currently used hash table slots.
 * @param hashes Device pointer to the pre-computed hash values for the keys.
 * Functional Utility: Parallel insertion of multiple key-value pairs into the GPU hash table.
 * Memory Hierarchy: All pointers (`hashMapKeys`, `hashMapValues`, `keysDev`, `valuesDev`, `usedMemDev`, `hashes`)
 *                   point to global device memory.
 */
__global__ void insertHelp(int *hashMapKeys, int *hashMapValues,
	int *keysDev, int *valuesDev, int numKeys, int mod, int *usedMemDev, int *hashes) {

	// Calculate global thread index.
	int i = threadIdx.x + blockDim.x * blockIdx.x;	

	// Block Logic: Ensure thread operates only if its index is within the bounds of numKeys.
	if (i < numKeys) {
		int j;    // Loop counter for probing.
		bool f = false; // Flag to indicate if insertion was successful.
		
		// Block Logic: Linear probing starting from the hashed index, going forwards.
		// Invariant: This loop attempts to find an empty slot (0) or a slot with the same key
		//            within the range of `mod - hashes[i]` positions after the initial hash.
		for (j = 0 ; j < mod - hashes[i]; j++) {
			
			// Attempt to insert key: if the slot is empty (0), place the key.
			// `atomicCAS` (Compare-And-Swap) ensures atomicity: if `hashMapKeys[j + hashes[i]]` is 0,
			// it's set to `keysDev[i]` and the old value (0) is returned.
			if (atomicCAS(&hashMapKeys[j + hashes[i]], 0, keysDev[i]) == 0) {
				// If successfully inserted into an empty slot, increment used memory counter atomically.
				atomicAdd(usedMemDev, 1);
				hashMapValues[j + hashes[i]] = valuesDev[i]; // Store the value.
                f = true; // Mark as successful.
                break;    // Exit probing loop.
			} 
            // Else if the slot already contains the same key, update its value.
            // `atomicCAS` here effectively checks if `hashMapKeys[j + hashes[i]]` is equal to `keysDev[i]`
            // and if so, tries to set it to `keysDev[i]` (which is a no-op but returns the old value).
            // This is slightly unconventional; a simple read could suffice if duplicate keys are allowed
            // to simply update values. If `atomicCAS` returns the key, it means the key already existed.
            else if (atomicCAS(&hashMapKeys[j + hashes[i]], 0, keysDev[i]) == keysDev[i]) {
				hashMapValues[j + hashes[i]] = valuesDev[i]; // Update the value for an existing key.
				f = true; // Mark as successful.
				break;    // Exit probing loop.
			}	
		}
		
		// Block Logic: If insertion failed after probing forwards (table wrapped around or full in that direction),
		//            try probing backwards from the hashed index.
		// Invariant: This loop attempts to find an empty slot (0) or a slot with the same key
		//            within the range of `hashes[i]` positions before the initial hash.
		if (!f) {
			for (j = -hashes[i] ; j < 0 ; j++) { // Probing backwards.
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
 * @brief Host helper function to extract non-zero key-value pairs from an old hash table.
 * Functional Utility: Used during the reshape operation to collect valid elements
 *                     that need to be re-inserted into the new, larger hash table.
 * @param oldmod The size of the old hash table.
 * @param nonZeroKeys Host array to store keys that are not 0.
 * @param nonZeroValues Host array to store values corresponding to non-zero keys.
 * @param allKeys Host array containing all keys from the old hash table.
 * @param allValues Host array containing all values from the old hash table.
 * @return The number of non-zero key-value pairs found.
 */
int getNonZeroKeysValues(int oldmod, int* nonZeroKeys, int *nonZeroValues, int *allKeys, int *allValues)
{
	int nr = 0; // Counter for non-zero entries.
	// Block Logic: Iterate through all entries of the old hash table.
	// Invariant: `nr` counts the number of valid (non-zero key) entries.
	for (int i = 0 ; i < oldmod; i++) {
		if (allKeys[i] > 0) { // Check if the key is valid (not 0).
			nonZeroKeys[nr] = allKeys[i];       // Store the key.
			nonZeroValues[nr++] = allValues[i]; // Store the value and increment count.
		}
	}
	return nr; // Return total number of valid entries.
}

/**
 * @brief Resizes the hash table to a new number of buckets.
 * Functional Utility: Expands the hash table when it becomes too full, ensuring
 *                     efficient operations by reducing collisions. Existing elements
 *                     are rehashed and inserted into the new table.
 * @param numBucketsReshape The desired new size for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int oldmod = mod; // Store the old size of the hash table.
	int nr = 0;       // Number of non-zero elements to re-insert.
	mod = numBucketsReshape; // Update the hash table size.
	totMem = mod;          // Update total memory capacity.
	
	// Pointers for host and device memory to manage keys and values during reshape.
	int *nonZeroKeys, *nonZeroKeysCUDA, *allKeys;	
	int *nonZeroValues, *nonZeroValuesCUDA, *allValues;
	
	// Block Logic: Only proceed if there were elements in the old hash table.
	if (usedMem[0] > 0) {
		// Allocate host memory for collecting non-zero keys and values.
		nonZeroKeys = (int *)malloc(oldmod * dim);
		allKeys = (int *)malloc(oldmod * dim); // Temporary host storage for old keys.
		cudaMalloc((void**)&(nonZeroKeysCUDA), oldmod * dim); // Allocate device memory.
		
		nonZeroValues = (int *)malloc(oldmod * dim);
		allValues = (int *)malloc(oldmod * dim); // Temporary host storage for old values.
		cudaMalloc((void**)&(nonZeroValuesCUDA), oldmod * dim); // Allocate device memory.
		
		// Copy old hash table keys and values from device to host.
		cudaMemcpy(allKeys, hashMapKeys, oldmod * dim, cudaMemcpyDeviceToHost);
		cudaMemcpy(allValues, hashMapValues, oldmod * dim, cudaMemcpyDeviceToHost);
		// Extract non-zero key-value pairs to host arrays.
		nr = getNonZeroKeysValues(oldmod, nonZeroKeys, nonZeroValues, allKeys, allValues);
		
		// Copy extracted non-zero keys and values from host to device.
		cudaMemcpy(nonZeroKeysCUDA, nonZeroKeys, oldmod * dim,
			cudaMemcpyHostToDevice);
		cudaMemcpy(nonZeroValuesCUDA, nonZeroValues, oldmod * dim,
			cudaMemcpyHostToDevice);
        
        // Free temporary host memory.
        free(nonZeroKeys);
        free(allKeys);
        free(nonZeroValues);
        free(allValues);
        
        // Free old device memory.
        cudaFree(hashMapKeys);
        cudaFree(hashMapValues);
	} 

	// Block Logic: Allocate new, larger device memory for the hash table keys and values.
	cudaMalloc((void **)&(hashMapKeys), mod * dim);
    cudaMemset((void **)&(hashMapKeys), 0, mod * dim); // Initialize new table to 0.
    
    cudaMalloc((void **)&(hashMapValues), mod * dim);
    cudaMemset((void **)&(hashMapValues), 0, mod * dim); // Initialize new table to 0.

	// Block Logic: Re-insert the non-zero elements if any existed in the old table.
	if (nr > 0) { // Changed from usedMem[0] to nr, as nr is the actual count of valid elements to re-insert.
		usedMem[0] = 0; // Reset host-side used memory counter.
		cudaMemcpy(usedMemDev, usedMem, dim, cudaMemcpyHostToDevice); // Reset device-side counter.
    
    	// Allocate host memory for hashes of non-zero keys.
    	int *hashes = (int *)calloc(nr, sizeof(int));
	
		// Compute hashes for all non-zero keys for the new table size.
		for (int i = 0 ; i < nr ; i++) {
			hashes[i] = myHash(nonZeroKeys[i], mod);
		}
		
		// Allocate device memory for hashes and copy from host.
		int *hashesCUDA;
		cudaMalloc((void **)&hashesCUDA, nr * dim);
		cudaMemcpy(hashesCUDA, hashes, nr * dim, cudaMemcpyHostToDevice);
    
    	// Launch kernel to re-insert non-zero keys and values into the new hash table.
        // Determines grid and block dimensions for kernel launch.
        int blocks = (nr + 255) / 256; // Ceiling division to get number of blocks.
        insertHelp<<<blocks, 256>>>(hashMapKeys, hashMapValues,
			nonZeroKeysCUDA, nonZeroValuesCUDA, nr, mod, usedMemDev, hashesCUDA);
		
		cudaDeviceSynchronize(); // Synchronize to ensure kernel completes before host proceeds.
		cudaMemcpy(usedMem, usedMemDev, dim, cudaMemcpyDeviceToHost); // Copy updated usedMem count from device to host.

        // Free device memory used for re-insertion data.
        cudaFree(nonZeroKeysCUDA);
        cudaFree(nonZeroValuesCUDA);
        cudaFree(hashesCUDA);
        // Free host memory used for re-insertion data.
        free(hashes);
	}
}


/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * Functional Utility: Provides an interface for high-throughput insertion
 *                     of multiple elements into the GPU hash table.
 * @param keys Host array of keys to insert.
 * @param values Host array of values to insert.
 * @param numKeys Number of key-value pairs in the batch.
 * @return true if insertion is successful.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	// Block Logic: Check if reshaping is needed due to high load factor.
	// If the current used memory plus the new keys would exceed 95% of total memory, reshape.
	// The new size is calculated to be 15% larger than the current + new keys.
	if (usedMem[0] + numKeys + (usedMem[0] + numKeys) * 0.05f > totMem) {
		reshape(usedMem[0] + numKeys + (usedMem[0] + numKeys) * 0.15f);
	}

	// Allocate host memory for hashes and compute them.
	int *hashes = (int *)calloc(numKeys, sizeof(int));
	
	// Block Logic: Compute hash for each key using `myHash` function.
	for (int i = 0 ; i < numKeys ; i++) {
		hashes[i] = myHash(keys[i], mod);
	}
	
	// Allocate device memory for hashes and copy from host.
	int *hashesCUDA;
	cudaMalloc((void **)&hashesCUDA, numKeys * dim);
	cudaMemcpy(hashesCUDA, hashes, numKeys * dim, cudaMemcpyHostToDevice);
	
	// Allocate device memory for keys and copy from host.
	int *keysCUDA = 0;
	cudaMalloc((void **)&keysCUDA, numKeys * dim);
	cudaMemcpy(keysCUDA, keys, numKeys * dim, cudaMemcpyHostToDevice);
	
	// Allocate device memory for values and copy from host.
	int *valuesCUDA = 0;
	cudaMalloc((void **)&valuesCUDA, numKeys * dim);
	cudaMemcpy(valuesCUDA, values, numKeys * dim, cudaMemcpyHostToDevice);

	// Launch the `insertHelp` kernel to perform parallel insertion.
    // Determines grid and block dimensions for kernel launch.
    int blocks = (numKeys + 255) / 256; // Ceiling division to get number of blocks.
	insertHelp<<<blocks, 256>>>(hashMapKeys, hashMapValues,
		keysCUDA, valuesCUDA, numKeys, mod, hashesCUDA);
	
	// Copy updated used memory count from device to host.
	cudaMemcpy(usedMem, usedMemDev, dim, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize(); // Synchronize to ensure kernel completes.

    // Free device memory.
    cudaFree(hashesCUDA);
    cudaFree(keysCUDA);
    cudaFree(valuesCUDA);
    // Free host memory.
    free(hashes);
	
	return true; // Indicate successful batch insertion.
}


/**
 * @brief CUDA kernel for retrieving values for a batch of keys from the hash table.
 *        Employs linear probing to find the keys.
 * @param hashMapKeys Device pointer to the array storing hash table keys.
 * @param hashMapValues Device pointer to the array storing hash table values.
 * @param retCUDA Device pointer to store the retrieved values.
 * @param keysDev Device pointer to the batch of keys to retrieve.
 * @param numKeys Number of keys in the current batch.
 * @param mod Current size (modulo) of the hash table.
 * @param hashes Device pointer to the pre-computed hash values for the keys.
 * Functional Utility: Parallel retrieval of multiple values from the GPU hash table.
 * Memory Hierarchy: All pointers point to global device memory.
 */
__global__ void helperGetBatch(int *hashMapKeys, int *hashMapValues, int *retCUDA,
	int *keysDev, int numKeys, int mod, int *hashes) {

	// Calculate global thread index.
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	// Block Logic: Ensure thread operates only if its index is within the bounds of numKeys.
	if (i < numKeys) {
		int j;    // Loop counter for probing.
		bool f = false; // Flag to indicate if key was found.
		
		// Block Logic: Linear probing starting from the hashed index, going forwards.
		// Invariant: This loop attempts to find the key `keysDev[i]` within the hash table.
		for (j = 0 ; j < mod - hashes[i]; j++) {
			// If key is found, store its corresponding value and break.
			if (keysDev[i] == hashMapKeys[j + hashes[i]]) {
				retCUDA[i] = hashMapValues[j + hashes[i]];
				f = true;
				break;
			}
            // If an empty slot (0) is encountered, the key is not in the table.
            else if (hashMapKeys[j + hashes[i]] == 0) {
                retCUDA[i] = 0; // Or a specific "not found" value.
                f = true;
                break;
            }
		}
		
		// Block Logic: If key not found after probing forwards, try probing backwards.
		if (!f) {
			for (j = -hashes[i] ; j < 0 ; j++) {
				if (keysDev[i] == hashMapKeys[j + hashes[i]]) {
					retCUDA[i] = hashMapValues[j + hashes[i]];
					f = true;
					break;
				}
                else if (hashMapKeys[j + hashes[i]] == 0) {
                    retCUDA[i] = 0; // Or a specific "not found" value.
                    f = true;
                    break;
                }
			}
		}
        // If still not found (e.g., table full of different keys), default to 0.
        if (!f) {
            retCUDA[i] = 0; 
        }
	}
}

/**
 * @brief Retrieves values for a batch of keys from the hash table.
 * Functional Utility: Provides an interface for high-throughput retrieval
 *                      of multiple elements from the GPU hash table.
 * @param keys Host array of keys to retrieve.
 * @param numKeys Number of keys in the batch.
 * @return A host array containing the retrieved values, or 0 if not found.
 *         The caller is responsible for freeing this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	// Allocate host memory for results.
	int *ret = (int *)malloc(numKeys * dim);
	
	// Allocate host memory for hashes and compute them.
	int *hashes = (int *)calloc(numKeys, sizeof(int));
	
	// Block Logic: Compute hash for each key.
	for (int i = 0 ; i < numKeys ; i++) {
		hashes[i] = myHash(keys[i], mod);
	}
	
	// Allocate device memory for hashes and copy from host.
	int *hashesCUDA;
	cudaMalloc((void **)&hashesCUDA, numKeys * dim);
	cudaMemcpy(hashesCUDA, hashes, numKeys * dim, cudaMemcpyHostToDevice);
	
	// Allocate device memory for keys and copy from host.
	int *keysDev = 0;
	cudaMalloc((void **)&keysDev, numKeys * dim);
	cudaMemcpy(keysDev, keys, numKeys * dim, cudaMemcpyHostToDevice);
	
	// Allocate device memory for results.
	int *retCUDA;
	cudaMalloc((void **)&retCUDA, numKeys * dim);

	// Launch the `helperGetBatch` kernel to perform parallel retrieval.
    // Determines grid and block dimensions for kernel launch.
    int blocks = (numKeys + 255) / 256; // Ceiling division to get number of blocks.
	helperGetBatch<<<blocks, 256>>>(hashMapKeys, hashMapValues,
		retCUDA, keysDev, numKeys, mod, hashesCUDA);
	// Copy results from device to host.
	cudaMemcpy(ret, retCUDA, numKeys * dim, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize(); // Synchronize to ensure kernel completes.

    // Free device memory.
    cudaFree(hashesCUDA);
    cudaFree(keysDev);
    cudaFree(retCUDA);
    // Free host memory.
    free(hashes);

	return ret; // Return host array of results.
}

/**
 * @brief Calculates the current load factor of the hash table.
 * Functional Utility: Provides a metric for hash table efficiency. A higher load factor
 *                     can indicate increased collision and slower performance.
 * @return The load factor as a float (used_slots / total_capacity).
 */
float GpuHashTable::loadFactor() {
	return ((float)usedMem[0])/((float)totMem);
}

// These macros are likely for external testing or integration, not part of the core hash table logic.
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

// This line suggests that the test code might be included here, or it's a placeholder.
// #include "test_map.cpp"

// Block Logic: Conditional compilation to ensure the definition of GpuHashTable and
//              related functions/macros are only processed once.
#ifndef _HASHCPU_
#define _HASHCPU_

// Standard namespace declarations for convenience.
using namespace std;

#define	KEY_INVALID		0 // Defines a value to represent an invalid or empty key in the hash table.

/**
 * @brief Macro for error checking and termination.
 * Functional Utility: Simplifies error handling by checking an assertion and
 *                     exiting the program with an error message if it fails.
 */
#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)
	
// Array of large prime numbers, likely used in the hash function to
// distribute keys effectively and reduce collisions.
const size_t primeList[] =
{
	5746614499066534157llu, 7240280573005008577llu, 9122181901073924329llu,
	11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu
};


/**
 * @brief Custom hash function for integers.
 * @param data The integer key to hash.
 * @param limit The upper bound for the hash value (hash table size).
 * @return An integer representing the hash value,
 *         guaranteed to be within [0, limit-1].
 * Functional Utility: Distributes integer keys across the hash table buckets.
 */
int myHash(int data, int limit) {
	// Applies a multiplicative hash using two prime numbers from `primeList`,
	// then takes the modulo with `limit` to fit within the hash table bounds.
	// `abs(data)` ensures positive input for hashing.
	return ((long)abs(data) * primeList[1]) % primeList[3] % limit;
}
	
// The GpuHashTable class declaration is here as well, possibly duplicated or for different compilation units.
// Assuming the one at the top is the primary implementation and this one is for compilation checks.
/*
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
		void occupancy(); // Not implemented in this file.
		void print(string info); // Not implemented in this file.
	
		~GpuHashTable();
};
*/
#endif // _HASHCPU_
