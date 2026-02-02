/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA.
 *
 * This file provides a Cuda C++ implementation of a hash table designed to operate
 * efficiently on a GPU. It utilizes a two-bucket (or Cuckoo-like) hashing strategy
 * with atomic operations to handle collisions and ensure thread-safe concurrent
 * access during insertion and rehashing. The hash table supports batch insertions
 * and retrievals, making it suitable for high-throughput data processing in parallel environments.
 * It also includes dynamic resizing (rehashing) to maintain performance under varying load.
 *
 * Functional Utility: Provides a high-performance key-value store optimized for
 * parallel processing on NVIDIA GPUs, particularly useful in applications requiring
 * rapid lookups and insertions in large datasets.
 *
 * Algorithm:
 * - Two-bucket hashing: Each key is mapped to two potential locations using a hash function.
 * - Linear probing/collision resolution: If the primary hashed location is occupied,
 *   a linear scan is performed to find an available slot within a designated range.
 * - Atomic operations: `atomicCAS` (Compare-And-Swap) is used to ensure thread-safe
 *   insertion and rehashing, preventing race conditions when multiple threads
 *   attempt to modify the same hash table entry.
 * - Dynamic Resizing (Reshape/Rehash): When the load factor exceeds a threshold,
 *   the hash table is resized to a larger prime number capacity, and all existing
 *   elements are rehashed into the new, larger table. This uses CUDA kernels for
 *   parallel rehashing.
 * - Batch Operations: Operations like insert and get are designed to process
 *   multiple key-value pairs in parallel using CUDA kernels.
 *
 * CUDA Architecture Details:
 * - `__global__` kernels: `kernel_insert`, `kernel_get`, `kernel_rehash` are executed
 *   on the GPU by multiple threads in parallel.
 * - `__device__` functions: `myHash` is a device-side helper function.
 * - Grid and Block organization: Threads are organized into blocks and grids, with
 *   `idx = blockIdx.x * blockDim.x + threadIdx.x` calculating a unique global thread ID.
 * - Device memory allocation: `cudaMalloc` and `cudaFree` are used for managing memory
 *   on the GPU.
 * - Synchronization: `cudaDeviceSynchronize` ensures that all GPU operations complete
 *   before proceeding with host-side code.
 *
 * Time Complexity:
 * - `myHash`: O(1)
 * - Host-side `GpuHashTable` methods: Dominated by CUDA kernel launch overhead and `cudaMemcpy` (O(N) for N items in batch),
 *   plus kernel execution time.
 * - `kernel_insert`, `kernel_get`, `kernel_rehash`: In the best case, O(1) for each thread (direct hit).
 *   In the worst case, performance can degrade to O(bucketSize) or O(linear_probe_length) per thread if many collisions occur.
 *   With good hash functions and sufficient load factor, average case is closer to O(1) per thread.
 * - `reshape`: O(oldBucketSize) due to parallel rehash kernel.
 * Space Complexity:
 * - `GpuHashTable`: O(bucketSize) for the two hash buckets on device memory.
 * - Kernel arguments/temporary device arrays: O(numKeys) for batch operations.
 */
#include <iostream> ///< For standard C++ input/output operations, e.g., `std::cerr`.
#include <vector> ///< For dynamic arrays, though not explicitly used in the core hash table logic here.
#include <string> ///< For string manipulation.
#include <cmath> ///< For mathematical functions, e.g., `abs`.
#include <algorithm> ///< For algorithms like `min`, `max`, though not explicitly used here.

#include "gpu_hashtable.hpp" ///< Header file defining `GpuHashTable` class and `hash_entry` struct.

/**
 * @brief CUDA device function to compute a hash value for a given integer data.
 * Functional Utility: Provides a fast hash function executed on the GPU, used by kernels
 * to map keys to bucket indices. The hash function uses multiplication and modulo
 * operations with large prime numbers to distribute keys evenly.
 *
 * @param data The integer key to hash.
 * @param limit The upper bound (size of the hash table) for the hash value.
 * @return An integer hash value within the range [0, limit-1].
 */
__device__ int myHash(int data, int limit) {
	// Algorithm: Multiplicative hashing with large primes, followed by modulo to fit within limits.
	// The specific prime numbers (20906033, 5351951779) are chosen for good distribution.
	return ((long) abs(data) * 20906033) % 5351951779 % limit;
}

/**
 * @brief Constructor for `GpuHashTable`.
 * Functional Utility: Initializes the GPU hash table by allocating device memory
 * for the two hash buckets and setting their initial state to empty (KEY_INVALID).
 *
 * @param size The initial size (number of entries) for each hash bucket.
 */
GpuHashTable::GpuHashTable(int size) {
	pairsInserted = 0; ///< Initialize count of inserted pairs.
	bucketSize = size; ///< Store the size of each bucket.
	bucket1 = nullptr; ///< Initialize bucket pointers to null.
	bucket2 = nullptr;
	// Block Logic: Allocate device memory for the first bucket.
	if (cudaMalloc(&bucket1, size * sizeof(hash_entry)) != cudaSuccess) {
		return; // Error handling: if allocation fails, return.
	}
	cudaMemset(bucket1, 0, size * sizeof(hash_entry)); ///< Initialize bucket 1 memory to zeros.
	// Block Logic: Allocate device memory for the second bucket.
	if (cudaMalloc(&bucket2, size * sizeof(hash_entry)) != cudaSuccess) {
		return; // Error handling: if allocation fails, return.
	}
	cudaMemset(bucket2, 0, size * sizeof(hash_entry)); ///< Initialize bucket 2 memory to zeros.
}

/**
 * @brief Destructor for `GpuHashTable`.
 * Functional Utility: Frees the device memory allocated for the hash buckets,
 * preventing memory leaks on the GPU.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(bucket1); ///< Free device memory for bucket 1.
	cudaFree(bucket2); ///< Free device memory for bucket 2.
}

/**
 * @brief CUDA kernel for parallel insertion of key-value pairs into the hash table.
 * Functional Utility: Each thread in this kernel attempts to insert a single key-value
 * pair into the hash table. It uses `myHash` to find potential locations and `atomicCAS`
 * to handle collisions and ensure thread-safe insertion into the two buckets.
 *
 * @param keys Pointer to device memory containing an array of keys to insert.
 * @param values Pointer to device memory containing an array of values corresponding to keys.
 * @param numKeys The number of key-value pairs to insert.
 * @param bucket1 Pointer to device memory for the first hash bucket.
 * @param bucket2 Pointer to device memory for the second hash bucket.
 * @param bucketSize The size of each hash bucket.
 */
__global__ void kernel_insert(int *keys, int *values, int numKeys, hash_entry* bucket1, hash_entry* bucket2, int bucketSize) {

	// Functional Utility: Calculate unique global thread ID.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numKeys) return; // Guard against out-of-bounds access if more threads than keys.

	int keyOld, keyNew;
	keyNew = keys[idx]; // Key to insert by current thread.
	int hash = myHash(keyNew, bucketSize); // Compute hash for the key.

	// Block Logic: Attempt to insert in the first bucket (linear probing from hash to end).
	for (int i = hash; i < bucketSize; i++) {
		// Functional Utility: Atomically try to set the key if the slot is invalid or already contains the key.
		// Atomically sets bucket1[i].key to keyNew if its current value is KEY_INVALID.
		// Returns the old value of bucket1[i].key.
		keyOld = atomicCAS(&bucket1[i].key, KEY_INVALID, keyNew);
		if (keyOld == KEY_INVALID || keyOld == keyNew) { // If slot was empty or contained the same key.
			bucket1[i].value = values[idx]; // Insert the value.
			return; // Insertion successful, thread exits.
		}
		// Block Logic: If bucket1[i] is occupied, try to insert into bucket2[i].
		keyOld = atomicCAS(&bucket2[i].key, KEY_INVALID, keyNew);
		if (keyOld == KEY_INVALID || keyOld == keyNew) {
			bucket2[i].value = values[idx];
			return;
		}
	}
	// Block Logic: If not inserted yet, continue linear probing from start to hash-1.
	for (int i = 0; i < hash; i++) {
		keyOld = atomicCAS(&bucket1[i].key, KEY_INVALID, keyNew);
		if (keyOld == KEY_INVALID || keyOld == keyNew) {
			bucket1[i].value = values[idx];
			return;
		}
		keyOld = atomicCAS(&bucket2[i].key, KEY_INVALID, keyNew);
		if (keyOld == KEY_INVALID || keyOld == keyNew) {
			bucket2[i].value = values[idx];
			return;
		}
	}
	// Precondition: This part of the code should ideally not be reached if the hash table is well-sized and has proper collision handling.
	// Functional Utility: If this point is reached, it indicates a failure to insert due to full table or persistent collisions.
	// In a robust implementation, this would trigger a rehash or an error.
}

/**
 * @brief CUDA kernel for parallel retrieval of values for a batch of keys.
 * Functional Utility: Each thread in this kernel attempts to find the value
 * associated with a single key in the hash table. It searches both buckets
 * using linear probing from the hashed location.
 *
 * @param keys Pointer to device memory containing an array of keys to search for.
 * @param values Pointer to device memory where retrieved values will be stored.
 * @param numItems The number of keys to retrieve.
 * @param bucket1 Pointer to device memory for the first hash bucket.
 * @param bucket2 Pointer to device memory for the second hash bucket.
 * @param bucketSize The size of each hash bucket.
 */
__global__ void kernel_get(int *keys, int *values, int numItems, hash_entry* bucket1, hash_entry* bucket2, int bucketSize) {
	// Functional Utility: Calculate unique global thread ID.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numItems) return; // Guard against out-of-bounds access.

	int crtKey = keys[idx]; // Key to search for by current thread.
	int hash = myHash(crtKey, bucketSize); // Compute hash for the key.

	// Block Logic: Search for the key in the first bucket (linear probing from hash to end).
	for (int i = hash; i < bucketSize; i++) {
		if (bucket1[i].key == crtKey) { // If key found in bucket 1.
			values[idx] = bucket1[i].value; // Store the value.
			return; // Retrieval successful.
		}
		if (bucket2[i].key == crtKey) { // If key found in bucket 2.
			values[idx] = bucket2[i].value; // Store the value.
			return;
		}
	}
	// Block Logic: If not found yet, continue linear probing from start to hash-1.
	for (int i = 0; i < hash; i++) {
		if (bucket1[i].key == crtKey) {
			values[idx] = bucket1[i].value;
			return;
		}
		if (bucket2[i].key == crtKey) {
			values[idx] = bucket2[i].value;
			return;
		}
	}
	// Functional Utility: If the key is not found, the corresponding value in `values` array
	// will remain its initialized state (typically 0 or some other default).
}

/**
 * @brief CUDA kernel for parallel rehashing of existing entries into a new, larger hash table.
 * Functional Utility: Each thread processes an entry from the old hash table and attempts
 * to insert it into the new hash table. This is similar to `kernel_insert` but specifically
 * for moving elements during a resize operation. Atomic operations (`atomicCAS`) are used
 * to ensure thread-safe insertion into the new buckets.
 *
 * @param oldBucket1 Pointer to device memory for the first old hash bucket.
 * @param oldBucket2 Pointer to device memory for the second old hash bucket.
 * @param oldBucketSize The size of the old hash buckets.
 * @param newBucket1 Pointer to device memory for the first new hash bucket.
 * @param newBucket2 Pointer to device memory for the second new hash bucket.
 * @param newBucketSize The size of the new hash buckets.
 */
__global__ void kernel_rehash(hash_entry* oldBucket1, hash_entry* oldBucket2, int oldBucketSize,
hash_entry* newBucket1, hash_entry* newBucket2, int newBucketSize) {
	// Functional Utility: Calculate unique global thread ID.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= oldBucketSize) return; // Guard against out-of-bounds access.

	// Block Logic: Process entry from oldBucket1 if it contains a valid key.
	if (oldBucket1[idx].key != KEY_INVALID) {
		int keyNew, keyOld, hash;
		keyNew = oldBucket1[idx].key; // Key to rehash.
		hash = myHash(keyNew, newBucketSize); // Compute hash for the new table size.
		bool completed = false;
		// Attempt to insert in the newBucket1 (linear probing from hash to end).
		for (int i = hash; i < newBucketSize; i++) {
			keyOld = atomicCAS(&newBucket1[i].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) { // If slot was empty.
				newBucket1[i].value = oldBucket1[idx].value; // Insert value.
				completed = true;
				break;
			}
			keyOld = atomicCAS(&newBucket2[i].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) { // If slot was empty.
				newBucket2[i].value = oldBucket2[idx].value; // Insert value.
				completed = true;
				break;
			}
		}
		// If not inserted yet, continue linear probing from start to hash-1.
		if (completed == false) {
			for (int i = 0; i < hash; i++) {
				keyOld = atomicCAS(&newBucket1[i].key, KEY_INVALID, keyNew);
				if (keyOld == KEY_INVALID) {
					newBucket1[i].value = oldBucket1[idx].value;
					break;
				}
				keyOld = atomicCAS(&newBucket2[i].key, KEY_INVALID, keyNew);
				if (keyOld == KEY_INVALID) {
					newBucket2[i].value = oldBucket2[idx].value;
					break;
				}
			}
		}
	}
	// Block Logic: Process entry from oldBucket2 if it contains a valid key.
	// This ensures that all entries from both old buckets are rehashed.
	if (oldBucket2[idx].key != KEY_INVALID) {
		int keyNew, keyOld, hash;
		keyNew = oldBucket2[idx].key;
		hash = myHash(keyNew, newBucketSize);
		bool completed = false;
		for (int i = hash; i < newBucketSize; i++) {
			keyOld = atomicCAS(&newBucket1[i].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket1[i].value = oldBucket1[idx].value;
				completed = true;
				break;
			}
			keyOld = atomicCAS(&newBucket2[i].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket2[i].value = oldBucket2[idx].value;
				completed = true;
				break;
			}
		}
		if (completed == false) {
			for (int i = 0; i < hash; i++) {
			keyOld = atomicCAS(&newBucket1[i].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket1[i].value = oldBucket1[idx].value;
				break;
			}
			keyOld = atomicCAS(&newBucket2[i].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket2[i].value = oldBucket2[idx].value;
				break;
			}
		}
		}
	}
}

/**
 * @brief Resizes the hash table to a new capacity and rehashes all existing elements.
 * Functional Utility: Dynamically adjusts the capacity of the hash table to maintain
 * performance (e.g., lower collision rates) as the number of inserted elements changes.
 * This involves allocating new device memory, launching a kernel to rehash elements,
 * and then freeing the old memory.
 *
 * @param sizeReshape The new size for the hash table buckets.
 */
void GpuHashTable::reshape(int sizeReshape) {
	hash_entry* newBucket1; ///< Pointer for the first new hash bucket on device.
	hash_entry* newBucket2; ///< Pointer for the second new hash bucket on device.
	// Block Logic: Allocate device memory for the first new bucket and initialize to zeros.
	if (cudaMalloc(&newBucket1, sizeReshape * sizeof(hash_entry)) != cudaSuccess) {
		return;
	}
	cudaMemset(newBucket1, 0, sizeReshape * sizeof(hash_entry));
	// Block Logic: Allocate device memory for the second new bucket and initialize to zeros.
	if (cudaMalloc(&newBucket2, sizeReshape * sizeof(hash_entry)) != cudaSuccess) {
		return;
	}
	cudaMemset(newBucket2, 0, sizeReshape * sizeof(hash_entry));
	// Functional Utility: Determine the number of blocks needed to launch the kernel,
	// ensuring all elements of the old table are processed.
	unsigned int numBlocks = bucketSize / THREADS_PER_BLOCK;
	if (bucketSize % THREADS_PER_BLOCK != 0) numBlocks++;
	// Functional Utility: Launch the kernel to rehash all elements from old buckets into new ones.
	kernel_rehash<<<numBlocks, THREADS_PER_BLOCK>>>(bucket1, bucket2, bucketSize, newBucket1, newBucket2, sizeReshape);
	cudaDeviceSynchronize(); ///< Ensure kernel completes before proceeding.
	cudaFree(bucket1); ///< Free old bucket 1 device memory.
	cudaFree(bucket2); ///< Free old bucket 2 device memory.
	bucket1 = newBucket1; ///< Update bucket 1 pointer to the new bucket.
	bucket2 = newBucket2; ///< Update bucket 2 pointer to the new bucket.
	bucketSize = sizeReshape; ///< Update the hash table's bucket size.
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * Functional Utility: Orchestrates the parallel insertion of multiple key-value pairs.
 * It manages host-to-device memory transfers, triggers resizing (rehashing) if the
 * load factor becomes too high, launches the insertion kernel, and updates internal counters.
 *
 * @param keys Pointer to an array of keys on the host.
 * @param values Pointer to an array of values on the host.
 * @param numKeys The number of key-value pairs in the batch.
 * @return `true` if insertion was successful, `false` otherwise (e.g., memory allocation failure).
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *keysFromDevice, *valuesFromDevice; ///< Pointers for device memory.
	unsigned int numBlocks;
	// Block Logic: Allocate device memory for keys.
	if (cudaMalloc(&keysFromDevice, numKeys * sizeof(int)) != cudaSuccess) {
		return false;
	}
	// Block Logic: Allocate device memory for values.
	if (cudaMalloc(&valuesFromDevice, numKeys * sizeof(int)) != cudaSuccess) {
		return false;
	}

	// Functional Utility: Check load factor and trigger reshape if necessary.
	// Precondition: `pairsInserted` and `bucketSize` are up-to-date.
	if ((pairsInserted + numKeys) * 1.0 / bucketSize >= MAX_LOAD)
		reshape(int((pairsInserted + numKeys) / MIN_LOAD)); // Resize to accommodate new keys while maintaining MIN_LOAD.

	cudaMemcpy(keysFromDevice, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice); ///< Copy keys from host to device.
	cudaMemcpy(valuesFromDevice, values, numKeys * sizeof(int), cudaMemcpyHostToDevice); ///< Copy values from host to device.
	// Functional Utility: Determine the number of blocks needed for the kernel launch.
	numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK != 0) numBlocks++;

	// Functional Utility: Launch the insertion kernel.
	kernel_insert<<<numBlocks, THREADS_PER_BLOCK>>>(keysFromDevice, valuesFromDevice, numKeys, bucket1, bucket2, bucketSize);
	cudaDeviceSynchronize(); ///< Ensure kernel completes.
	pairsInserted += numKeys; ///< Update count of inserted pairs.
	cudaFree(keysFromDevice); ///< Free temporary device memory for keys.
	cudaFree(valuesFromDevice); ///< Free temporary device memory for values.
	return true;
}

/**
 * @brief Retrieves a batch of values for given keys from the hash table.
 * Functional Utility: Orchestrates the parallel retrieval of multiple values.
 * It manages host-to-device memory transfer for keys, launches the retrieval kernel,
 * and returns a pointer to device memory containing the results.
 *
 * @param keys Pointer to an array of keys on the host to search for.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to device memory containing the retrieved values. The caller
 *         is responsible for `cudaFree`ing this memory. Returns `nullptr` on allocation failure.
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
	int *keysFromDevice, *valuesFromDevice; ///< Pointers for device memory.
	unsigned int numBlocks;
	// Block Logic: Allocate device memory for keys.
	if (cudaMalloc(&keysFromDevice, numKeys * sizeof(int)) != cudaSuccess) {
		std::cerr<< "Memory for keys failed to allocate\n";
		return nullptr;
	}
	// Block Logic: Allocate device memory for values.
	if (cudaMalloc(&valuesFromDevice, numKeys * sizeof(int)) != cudaSuccess) {
		std::cerr<< "Memory for values failed to allocate\n";
		return  nullptr;
	}
	cudaMemcpy(keysFromDevice, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice); ///< Copy keys from host to device.
	// Functional Utility: Determine the number of blocks needed for the kernel launch.
	numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK != 0) numBlocks++;
	// Functional Utility: Launch the retrieval kernel.
	kernel_get<<<numBlocks, THREADS_PER_BLOCK>>>(keysFromDevice, valuesFromDevice, numKeys, bucket1, bucket2, bucketSize);
	cudaDeviceSynchronize(); ///< Ensure kernel completes.
	cudaFree(keysFromDevice); ///< Free temporary device memory for keys.
	return valuesFromDevice; ///< Return pointer to device memory containing results.
}

/**
 * @brief Calculates the current load factor of the hash table.
 * Functional Utility: Provides a metric for the density of the hash table,
 * indicating how full it is. This is typically used to trigger resizing operations
 * to maintain efficient performance.
 *
 * @return The load factor (pairsInserted / bucketSize). Returns 0.0 if `bucketSize` is 0.
 */
float GpuHashTable::loadFactor() {
	if (bucketSize == 0) return 0.0; // Avoid division by zero.
	return pairsInserted * 1.0 / bucketSize; // Calculate load factor.
}

// Unimplemented methods in this file.
// void GpuHashTable::occupancy() {
// // Placeholder for future implementation to check bucket occupancy.
// }
//
// void GpuHashTable::print(std::string info) {
// // Placeholder for future implementation to print hash table contents for debugging.
// }


#define HASH_INIT GpuHashTable GpuHashTable(1); ///< Macro for initializing a GpuHashTable object.
#define HASH_RESERVE(size) GpuHashTable.reshape(size); ///< Macro for reshaping the GpuHashTable.

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys) ///< Macro for batch insertion.
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys) ///< Macro for batch retrieval.

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor() ///< Macro for getting the load factor.

// #include "test_map.cpp" ///< Include for testing purposes. (Usually commented out in production code) 
// The content below this point is typically from a .hpp file or a common utility header,
// and it defines constants, helper functions, and the class structure.
#ifndef _HASHCPU_
#define _HASHCPU_

#include <stddef.h> ///< For `std::size_t`.
#include <cstdio> ///< For `fprintf`, `perror`.
#include <cstdlib> ///< For `exit`.
#include <cerrno> ///< For `errno`.

#define THREADS_PER_BLOCK 1024 ///< Constant: Number of CUDA threads per block.
#define MIN_LOAD 0.5 ///< Constant: Minimum load factor before considering shrinking/rehashing for optimization.
#define MAX_LOAD 0.75 ///< Constant: Maximum load factor before triggering a resize/rehash.
#define KEY_INVALID 0 ///< Constant: Represents an invalid or empty key in the hash table.

/**
 * @brief Macro for error handling and exiting the program.
 * Functional Utility: Provides a convenient way to check assertions and
 * terminate the program with an error message if an assertion fails, printing
 * file and line number information.
 *
 * @param assertion The boolean condition to check.
 * @param call_description A string describing the function call or context.
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
 * @brief Array of large prime numbers.
 * Functional Utility: Used as part of the hash function calculations (`hash1`, `hash2`, `hash3`)
 * to generate distinct and well-distributed hash values, reducing collisions.
 * The use of large primes helps ensure a more uniform distribution of hashed keys.
 */
const std::size_t primeList[] =
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


/**
 * @brief Hash function 1.
 * Functional Utility: Computes a hash value for a given integer data using a specific
 * prime from `primeList` and applies modulo operations to keep the result within `limit`.
 * This is used as one of the candidate hash functions for the two-bucket hashing.
 *
 * @param data The integer key to hash.
 * @param limit The upper bound (size of the hash table) for the hash value.
 * @return An integer hash value within the range [0, limit-1].
 */
int hash1(int data, int limit) {
	return ((long) abs(data) * primeList[64]) % primeList[90] % limit;
}

/**
 * @brief Hash function 2.
 * Functional Utility: Computes a hash value for a given integer data using a specific
 * prime from `primeList` and applies modulo operations to keep the result within `limit`.
 * This is used as one of the candidate hash functions for the two-bucket hashing.
 *
 * @param data The integer key to hash.
 * @param limit The upper bound (size of the hash table) for the hash value.
 * @return An integer hash value within the range [0, limit-1].
 */
int hash2(int data, int limit) {
	return ((long) abs(data) * primeList[67]) % primeList[91] % limit;
}

/**
 * @brief Hash function 3.
 * Functional Utility: Computes a hash value for a given integer data using a specific
 * prime from `primeList` and applies modulo operations to keep the result within `limit`.
 * This is used as one of the candidate hash functions for the two-bucket hashing.
 * (Note: The kernel code seems to only use `myHash`, suggesting `hash1`, `hash2`, `hash3`
 * might be alternative or experimental hash functions for the host side or future extensions).
 *
 * @param data The integer key to hash.
 * @param limit The upper bound (size of the hash table) for the hash value.
 * @return An integer hash value within the range [0, limit-1].
 */
int hash3(int data, int limit) {
	return ((long) abs(data) * primeList[70]) % primeList[93] % limit;
}

/**
 * @struct hash_entry
 * @brief Represents a single entry in the hash table buckets.
 * Functional Utility: Stores a key-value pair. The `key` field is used to identify
 * the entry, with `KEY_INVALID` signifying an empty slot.
 */
struct hash_entry {
	int key; ///< The key of the hash entry.
	int value; ///< The value associated with the key.
};

/**
 * @class GpuHashTable
 * @brief (Forward Declaration/Redeclaration from gpu_hashtable.hpp)
 * Functional Utility: This class definition provides the public interface for the
 * GPU hash table, allowing host-side code to interact with and manage the device-side
 * hash table implementation. It includes methods for initialization, resizing,
 * batch insertion, batch retrieval, and querying the load factor.
 */
class GpuHashTable {
	int pairsInserted; ///< Counter for the number of key-value pairs currently in the hash table.
	hash_entry* bucket1; ///< Pointer to the device memory for the first hash bucket.
	hash_entry* bucket2; ///< Pointer to the device memory for the second hash bucket.
	int bucketSize; ///< The current allocated size of each hash bucket.

public:
	GpuHashTable(int size); ///< Constructor: Initializes the hash table with a given size.

	void reshape(int sizeReshape); ///< Resizes the hash table and rehashes elements.

	bool insertBatch(int *keys, int *values, int numKeys); ///< Inserts a batch of key-value pairs.

	int *getBatch(int *key, int numItems); ///< Retrieves a batch of values for given keys.

	float loadFactor(); ///< Calculates the current load factor of the hash table.

	void occupancy(); ///< Placeholder for a method to calculate bucket occupancy (unimplemented here).

	void print(std::string info); ///< Placeholder for a method to print hash table contents (unimplemented here).

	~GpuHashTable(); ///< Destructor: Frees allocated device memory.
};

#endif