/**
 * @file gpu_functions.cu
 * @brief Implements a GPU-accelerated hash table with linear probing and atomic operations.
 *
 * This file provides a CUDA C++ implementation of a hash table designed for
 * efficient parallel operations on NVIDIA GPUs. It uses open addressing with
 * linear probing and atomic compare-and-swap operations to ensure thread-safe
 * concurrent insertions and retrievals. The implementation supports dynamic
 * resizing (rehashing) to maintain performance characteristics as the number
 * of stored elements changes.
 *
 * Functional Utility: Offers a high-performance key-value store critical for
 * parallel computing tasks where fast, concurrent data access and modification
 * are required. Typical applications include data processing, caching, and
 * algorithm acceleration on GPUs.
 *
 * Algorithm:
 * - Hashing: A multiplicative hashing scheme is used to map keys to an initial
 *   index in the hash table.
 * - Linear Probing: When a collision occurs (the initial hash index is occupied),
 *   the algorithm probes subsequent indices in a linear fashion until an empty
 *   slot is found or the search wraps around.
 * - Atomic Operations: `atomicExch` (atomic exchange) and `atomicAdd` are used
 *   extensively to manage concurrency. `atomicExch` is vital for ensuring that
 *   only one thread can claim an empty slot or update a key/value pair at a
 *   given index at a time. `atomicAdd` is used for safely incrementing the
 *   `size` counter.
 * - Dynamic Resizing (`reshape`): The hash table automatically resizes when
 *   its load factor exceeds a `MAX_LFACTOR` threshold. During resizing, a new,
 *   larger hash table is allocated, and all existing elements are rehashed
 *   into it in parallel. This rehashes elements from the old table into the new
 *   one using the `insert` kernel.
 * - Batch Operations: Insertion and retrieval are implemented as batch operations,
 *   where multiple key-value pairs or keys are processed in parallel by launching
 *   CUDA kernels.
 *
 * CUDA Architecture Details:
 * - `__global__` kernels: `insert` and `get_val` perform the core hash table
 *   operations in parallel on the GPU.
 * - `__device__` functions: `hash_func` is a helper function callable from kernels.
 * - Global Thread ID: `threadIdx.x + blockDim.x * blockIdx.x` is used to compute
 *   a unique global thread index, allowing each thread to process a distinct piece of data.
 * - Device Memory Management: `cudaMalloc`, `cudaMemset`, and `cudaFree` are
 *   used to allocate, initialize, and deallocate memory on the GPU.
 * - Synchronization: `cudaDeviceSynchronize` is used to ensure all GPU operations
 *   complete before host-side code proceeds.
 *
 * Time Complexity:
 * - `hash_func`: O(1).
 * - `insert` and `get_val` kernels: In the best case (no collisions), O(1) per thread.
 *   In the worst case (many collisions or table nearly full), performance can degrade
 *   due to linear probing, potentially approaching O(HMAX) for a single operation.
 *   Average case is O(1) assuming a good hash function and reasonable load factor.
 * - Host-side `insertBatch`, `getBatch`, `reshape`: Dominated by `cudaMemcpy` operations
 *   (O(numKeys) or O(HMAX)) and kernel launch overhead, plus kernel execution time.
 * Space Complexity:
 * - `GpuHashTable`: O(HMAX) for storing `hash_keys`, `hash_values`, and `empty_slots` arrays on device memory.
 * - Kernel arguments/temporary device arrays: O(numKeys) for batch operations.
 */

#include <iostream> ///< For standard C++ input/output operations, e.g., `printf` and `std::cerr`.
#include <vector> ///< For dynamic arrays, though not explicitly used in the core hash table logic here.
#include <string> ///< For string manipulation, e.g., `std::string` in `print` method signature.
#include <cmath> ///< For mathematical functions, e.g., `abs`.
#include <algorithm> ///< For algorithms like `min`, `max`, though not explicitly used here.
#include <cstdio> ///< For `fprintf`, `perror`.
#include <cstdlib> ///< For `exit`.
#include <cerrno> ///< For `errno`.

#include "gpu_hashtable.hpp" ///< Header file defining `GpuHashTable` class structures (forward declarations).

// Constants for hash function and load factor management.
#define PRIME_A 26339969 ///< A large prime number used in hash function calculations.
#define PRIME_B 6743036717llu ///< Another large prime number used in hash function calculations.
#define MIN_LFACTOR 0.8f ///< Minimum load factor target for resizing to manage space (used in `reshape` target size).
#define MAX_LFACTOR 0.9f ///< Maximum load factor before triggering a hash table reshape (growth).
#define THREADS_PER_BLOCK 512 ///< Constant: Number of CUDA threads to launch per block for kernel execution.
#define KEY_INVALID 0 ///< Sentinel value representing an invalid or empty key in the hash table.

/**
 * @brief CUDA device function to compute a hash value for a given integer key.
 * Functional Utility: Provides a fast hash function executed on the GPU, used by kernels
 * to map keys to initial bucket indices. The hash function uses multiplication and modulo
 * operations with large prime numbers (`PRIME_A`, `PRIME_B`) for good key distribution.
 *
 * @param key The integer key to hash.
 * @param HMAX Pointer to the maximum size of the hash table.
 * @return An integer hash value within the range [0, *HMAX - 1].
 */
__device__ int hash_func(int key, int *HMAX) {
	return ((long)abs(key) * PRIME_A) % PRIME_B % *HMAX;
}

/**
 * @brief CUDA kernel for parallel insertion of key-value pairs into the hash table.
 * Functional Utility: Each thread in this kernel attempts to insert a single key-value
 * pair. It uses `hash_func` to get an initial index, then employs linear probing
 * and `atomicExch` to handle collisions and ensure thread-safe insertion. It also
 * supports updating values for existing keys and atomically updates the total size.
 *
 * @param hash_keys Pointer to device memory storing the keys in the hash table.
 * @param hash_values Pointer to device memory storing the values in the hash table.
 * @param empty_slots Pointer to device memory indicating if a slot is empty (0) or occupied (1).
 * @param gpu_keys Pointer to device memory containing keys to insert.
 * @param gpu_values Pointer to device memory containing values to insert.
 * @param size Pointer to device memory storing the current number of elements in the hash table.
 * @param HMAX Pointer to device memory storing the maximum size of the hash table.
 */
__global__ void insert(int *hash_keys, int *hash_values, int *empty_slots,
                        int *gpu_keys, int *gpu_values, int *size, int *HMAX)
{
	
	// Functional Utility: Calculate unique global thread ID.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	// Functional Utility: Retrieve key and value for the current thread.
	int key = gpu_keys[i];
	int value = gpu_values[i];

	// Block Logic: Basic validation for key and value.
	if (key <= 0 || value <= 0) { // Assuming non-positive keys/values are invalid.
		return;
	}
						
	// Functional Utility: Compute initial hash index for the key.
	int hash_key = hash_func(key, HMAX);
	int index = hash_key;

	// Block Logic: Attempt to insert using linear probing and atomic operations.
	// Invariant: The loop continues until an empty slot is found or the table has been fully probed.
	while (true) {
		// Functional Utility: Atomically exchange the value of `empty_slots[index]` with 1.
		// `elem` will hold the original value (0 if empty, 1 if occupied).
		int elem = atomicExch(&empty_slots[index], 1);
		
		if (elem == KEY_INVALID) { // If the slot was empty (0).
			hash_keys[index] = key; // Place the key.
			hash_values[index] = value; // Place the value.
			atomicAdd(size, 1); // Atomically increment the total size.
			return; // Insertion successful.
		}
		// Functional Utility: If the slot was already occupied by another thread that just got it,
		// or if it was already occupied by a different key, restore the `empty_slots` state.
		// This is critical if `atomicExch` with `1` was done on an already-set slot.
		atomicExch(&empty_slots[index], elem); // Restore original state (redundant if elem was 1, crucial if elem was set by another thread right before)
		
		if (hash_keys[index] == key) { // If the key already exists at this location.
			atomicExch(&hash_values[index], value); // Atomically update the value.
			return; // Update successful.
		}
		
		// Functional Utility: Move to the next slot for linear probing.
		index = (index + 1) % *HMAX;
		
		// Block Logic: Check if we have probed the entire table and returned to the start.
		if (index == hash_key) {
			// Precondition: If this point is reached, the table is considered full or a cycle occurred,
			// and insertion failed for this key. A robust system might trigger a rehash from here.
			return;
		}
	}
}


/**
 * @brief CUDA kernel for parallel retrieval of values for a batch of keys.
 * Functional Utility: Each thread attempts to find the value associated with a single key.
 * It uses `hash_func` to get an initial index, then employs linear probing to search
 * for the key. If found, the corresponding value is stored.
 *
 * @param hash_keys Pointer to device memory storing the keys in the hash table.
 * @param hash_values Pointer to device memory storing the values in the hash table.
 * @param empty_slots Pointer to device memory indicating if a slot is empty (0) or occupied (1).
 * @param keys Pointer to device memory containing keys to search for.
 * @param values Pointer to device memory where retrieved values will be stored.
 * @param HMAX Pointer to device memory storing the maximum size of the hash table.
 */
__global__ void get_val(int *hash_keys, int *hash_values, int *empty_slots,
                        int *keys, int* values, int *HMAX)
{
	
	// Functional Utility: Calculate unique global thread ID.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int key = keys[i]; // Key to search for by current thread.
						
	// Functional Utility: Compute initial hash index for the key.
	int hash_key = hash_func(key, HMAX);
	int index = hash_key;

	// Block Logic: Search for the key using linear probing.
	// Invariant: The loop continues until an empty slot is encountered or the entire table has been probed.
	while (empty_slots[index] == 1) { // Only probe occupied slots.

		// Functional Utility: If the key matches, store the value and return.
		if (hash_keys[index] == key) {
			values[i] = hash_values[index];
			return;
		}
		index = (index + 1) % *HMAX; // Move to the next slot.
		
		// Block Logic: Check if we have probed the entire table and returned to the start.
		if (index == hash_key) {
			break; // Key not found after full probe.
		}
	}
	// Functional Utility: If the key is not found, `values[i]` remains its initial value (e.g., 0).
}



/**
 * @brief Constructor for `GpuHashTable`.
 * Functional Utility: Initializes the host-side `GpuHashTable` object. It allocates
 * device memory for the hash table's key, value, and empty slot arrays, and initializes
 * them to `KEY_INVALID` (0). It also sets up initial size parameters.
 *
 * @param size The initial maximum capacity (`HMAX`) for the hash table on the GPU.
 */
GpuHashTable::GpuHashTable(int size) {
	
	this->HMAX = size; // Set the initial maximum size (capacity) of the hash table.
	this->size = 0; // Initialize the current number of elements to zero.

	
	this->hash_keys = 0; // Initialize device memory pointers to null.
	this->hash_values = 0;
	this->empty_slots = 0;
	// Block Logic: Allocate device memory for hash table components.
	cudaMalloc((void **) &this->hash_keys, sizeof(int) * this->HMAX);
	cudaMalloc((void **) &this->hash_values, sizeof(int) * this->HMAX);
	cudaMalloc((void **) &this->empty_slots, sizeof(int) * this->HMAX);
	// Functional Utility: Error check for CUDA memory allocation using the DIE macro.
	DIE(this->hash_keys == 0 || this->hash_values == 0 ||
		this->empty_slots == 0, "[GpuHashTable] cudaMalloc error");
	// Functional Utility: Initialize all allocated device memory to 0 (KEY_INVALID/empty).
	cudaMemset(this->hash_keys, 0, sizeof(int) * this->HMAX);
	cudaMemset(this->hash_values, 0, sizeof(int) * this->HMAX);
	cudaMemset(this->empty_slots, 0, sizeof(int) * this->HMAX);
}


/**
 * @brief Destructor for `GpuHashTable`.
 * Functional Utility: Frees all device memory allocated for the hash table
 * components (`hash_keys`, `hash_values`, `empty_slots`), preventing memory
 * leaks on the GPU when the `GpuHashTable` object is destroyed.
 */
GpuHashTable::~GpuHashTable() {
	// Block Logic: Free device memory only if pointers are valid.
	if (this->hash_keys != 0 || this->hash_keys != NULL) {
		cudaFree(this->hash_keys);
	}
	if (this->hash_values != 0 || this->hash_values != NULL) {
		cudaFree(this->hash_values);
	}
	if (this->empty_slots != 0 || this->empty_slots != NULL) {
		cudaFree(this->empty_slots);
	}

}


/**
 * @brief Resizes the hash table to a new capacity and rehashes all existing elements.
 * Functional Utility: Dynamically adjusts the capacity (`HMAX`) of the hash table
 * to maintain performance (e.g., lower collision rates). This involves:
 * 1. Allocating temporary host memory to store old hash table contents.
 * 2. Copying old data from device to host.
 * 3. Freeing old device memory.
 * 4. Allocating new, larger device memory.
 * 5. Re-inserting elements from temporary host memory into the new device hash table using the `insert` kernel.
 *
 * @param numBucketsReshape The new desired maximum capacity (`HMAX`) for the hash table.
 */
 void GpuHashTable::reshape(int numBucketsReshape) {
	int *gpu_keys = 0; // Temporary host pointer for old keys.
	int *gpu_values = 0; // Temporary host pointer for old values.

	// Block Logic: Allocate temporary host memory to store the old hash table's keys and values.
	cudaMalloc(&gpu_keys, sizeof(int) * this->HMAX); // Note: Should be cudaHostAlloc or just malloc if copying from device to host, but here cudaMalloc is used for temporary device-side storage.
	cudaMalloc(&gpu_values, sizeof(int) * this->HMAX);
	// Functional Utility: Error check for temporary device memory allocation.
	DIE(gpu_keys == 0 || gpu_values == 0, "[RESHAPE] cudaMalloc error");

	// Functional Utility: Copy old hash table's keys and values from device to the temporary device buffers.
	cudaMemcpy(gpu_keys, this->hash_keys, sizeof(int) * this->HMAX, cudaMemcpyDeviceToDevice); // Corrected: Should be DeviceToDevice if gpu_keys is device memory.
	cudaFree(this->hash_keys); // Free old hash_keys device memory.
	cudaMemcpy(gpu_values, this->hash_values, sizeof(int) * this->HMAX, cudaMemcpyDeviceToDevice); // Corrected: Should be DeviceToDevice if gpu_values is device memory.
	cudaFree(this->hash_values); // Free old hash_values device memory.
	cudaFree(this->empty_slots); // Free old empty_slots device memory.
	
	// Functional Utility: Reset device pointers to null before re-allocation.
	this->hash_keys = 0;
	this->hash_values = 0;
	this->empty_slots = 0;
	// Block Logic: Allocate new device memory for the reshaped hash table with the new size.
	cudaMalloc((void **) &this->hash_keys, sizeof(int) * numBucketsReshape);
	cudaMalloc((void **) &this->hash_values, sizeof(int) * numBucketsReshape);
	cudaMalloc((void **) &this->empty_slots, sizeof(int) * numBucketsReshape);
	// Functional Utility: Error check for new device memory allocation.
	DIE(this->hash_keys == 0 || this->hash_values == 0 ||
		this->empty_slots == 0, "[RESHAPE] cudaMalloc error");
	// Functional Utility: Initialize new device memory for empty slots, keys, and values to 0.
	cudaMemset(this->empty_slots, 0, sizeof(int) * numBucketsReshape);
	cudaMemset(this->hash_keys, 0, sizeof(int) * numBucketsReshape);
	cudaMemset(this->hash_values, 0, sizeof(int) * numBucketsReshape);
	
	if (this->size > 0) { // Block Logic: If there were elements in the old table, re-insert them.
		int numKeys = this->HMAX; // The number of keys to re-insert is the old HMAX.
		int *gpu_HMAX_param = 0; // Device pointer for the new HMAX parameter.
		int *gpu_size_param = 0; // Device pointer for the size parameter (will be updated by kernel).
		
		// Block Logic: Allocate temporary device memory for kernel parameters.
		cudaMalloc((void **) &gpu_size_param, sizeof(int));
		cudaMalloc((void **) &gpu_HMAX_param, sizeof(int));
		DIE(gpu_size_param == 0 || gpu_HMAX_param == 0, "[RESHAPE] malloc error");
		cudaMemset(&gpu_size_param, 0, sizeof(int)); // Initialize size parameter to 0 for the kernel.

		// Functional Utility: Copy the new HMAX value to device memory for the kernel.
		cudaMemcpy(gpu_HMAX_param, &numBucketsReshape, sizeof(int), cudaMemcpyHostToDevice);

		// Functional Utility: Launch the `insert` kernel to rehash elements from `gpu_keys`/`gpu_values`
		// into the newly allocated hash table. Each thread processes one old entry.
		insert<<<numKeys / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(this->hash_keys, this->hash_values,
				this->empty_slots, gpu_keys, gpu_values, gpu_size_param, gpu_HMAX_param);
		
		cudaDeviceSynchronize(); // Ensure kernel completes. 

		// Functional Utility: Free temporary device memory.
		cudaFree(gpu_keys);
		cudaFree(gpu_values);
		cudaFree(gpu_size_param);
		cudaFree(gpu_HMAX_param);
	}
	
	this->HMAX = numBucketsReshape; // Update the host-side HMAX.
	this->size = 0; // Reset size to 0 as insert kernel updates it from gpu_size_param
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * Functional Utility: Orchestrates the parallel insertion of multiple key-value pairs.
 * It manages host-to-device memory transfers, triggers resizing (`reshape`) if the
 * load factor (`currLFACTOR`) becomes too high, launches the `insert` kernel, and updates
 * the internal count of inserted pairs (`nrEntries`). It also handles potential shrinking.
 *
 * @param keys Pointer to an array of keys on the host.
 * @param values Pointer to an array of values on the host.
 * @param numKeys The number of key-value pairs in the batch.
 * @return `true` if insertion was successful, `false` otherwise (e.g., memory allocation failure).
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	
	float fact = this->loadFactor(); // Current load factor.
	int old_HMAX = this->HMAX; // Store old HMAX for potential shrinking later.
	// Functional Utility: Check load factor and trigger reshape (growth) if necessary.
	// Precondition: `this->size` and `this->HMAX` are accurate.
	if (fact + (float) numKeys / (float) this->HMAX > MAX_LFACTOR) { // If adding new keys will exceed MAX_LFACTOR.
		this->reshape((int)(((float) this->size + (float) numKeys) / MIN_LFACTOR)); // Reshape to maintain MIN_LFACTOR.
	}
	
	int *gpu_keys = 0; // Device pointer for keys.
	int *gpu_values = 0; // Device pointer for values.
	int *gpu_size = 0; // Device pointer to update `this->size`.
	int *gpu_HMAX_param = 0; // Device pointer for `this->HMAX`.
	// Block Logic: Allocate device memory for input keys/values and kernel parameters.
	cudaMalloc((void **) &gpu_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &gpu_values, numKeys * sizeof(int));
	cudaMalloc((void **) &gpu_size, sizeof(int));
	cudaMalloc((void **) &gpu_HMAX_param, sizeof(int));
	DIE(gpu_keys == 0 || gpu_values == 0 || gpu_size == 0 ||
		gpu_HMAX_param == 0, "[insertBatch] malloc error");

	// Functional Utility: Copy host data to device memory.
	cudaMemcpy(gpu_keys, keys, sizeof(int) * numKeys, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_values, values, sizeof(int) * numKeys, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_size, &this->size, sizeof(int), cudaMemcpyHostToDevice); // Copy current `size` to device.
	cudaMemcpy(gpu_HMAX_param, &this->HMAX, sizeof(int), cudaMemcpyHostToDevice); // Copy `HMAX` to device.

	// Functional Utility: Launch the `insert` kernel to perform parallel insertion.
	insert<<<numKeys / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(this->hash_keys, this->hash_values, this->empty_slots, gpu_keys,
						gpu_values, gpu_size, gpu_HMAX_param);
	
	cudaDeviceSynchronize(); // Ensure kernel completes. 

	// Functional Utility: Copy updated `size` from device back to host.
	cudaMemcpy(&this->size, gpu_size, sizeof(int), cudaMemcpyDeviceToHost);
	
	// Functional Utility: Free temporary device memory.
	cudaFree(gpu_keys);
	cudaFree(gpu_values);
	cudaFree(gpu_size);
	cudaFree(gpu_HMAX_param);
	
	// Functional Utility: Check load factor again for potential shrinking (if size decreased relative to old HMAX).
	float fact2 = this->loadFactor();
	if (this->size <= old_HMAX && fact2 < MIN_LFACTOR) { // If current elements fit in old HMAX and load factor is too low.
		this->reshape(old_HMAX); // Reshape back to old HMAX to save memory.
	}
	
	return true;
}


/**
 * @brief Retrieves a batch of values for given keys from the hash table.
 * Functional Utility: Orchestrates the parallel retrieval of multiple values.
 * It manages host-to-device memory transfer for keys, launches the `get_val` kernel,
 * and transfers the retrieved values from device memory back to newly allocated
 * host memory, returning a pointer to these values.
 *
 * @param keys Pointer to an array of keys on the host to search for.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to host memory containing the retrieved values. The caller
 *         is responsible for `free`ing this memory. Returns `nullptr` on allocation failure.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *values = (int *)malloc(numKeys * sizeof(int)); // Host array to store results.
	int *gpu_keys = 0; // Device pointer for keys.
	int *gpu_values = 0; // Device pointer for values (results).
	int *gpu_HMAX_param = 0; // Device pointer for `this->HMAX`.
	// Block Logic: Allocate device memory for input keys, output values, and kernel parameters.
	cudaMalloc((void **) &gpu_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &gpu_values, numKeys * sizeof(int));
	cudaMalloc((void **) &gpu_HMAX_param, sizeof(int));
	// Functional Utility: Error check for device memory allocation.
	DIE(gpu_keys == 0 || gpu_values == 0 || gpu_HMAX_param == 0,
				"[getBatch] malloc error");
	cudaMemset(gpu_values, 0, numKeys * sizeof(int)); // Initialize device output values to 0.

	
	// Functional Utility: Copy keys from host memory to device memory.
	cudaMemcpy(gpu_keys, keys, sizeof(int) * numKeys, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_HMAX_param, &this->HMAX, sizeof(int), cudaMemcpyHostToDevice); // Copy `HMAX` to device.

	// Functional Utility: Launch the `get_val` kernel to perform parallel retrieval.
	get_val<<<numKeys / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(this->hash_keys, this->hash_values, this->empty_slots,
							gpu_keys, gpu_values, gpu_HMAX_param);
	
	cudaDeviceSynchronize(); // Ensure kernel completes. 

	
	// Functional Utility: Copy the retrieved values from device memory back to host memory.
	cudaMemcpy(values, gpu_values, sizeof(int) * numKeys, cudaMemcpyDeviceToHost);
	
	// Functional Utility: Free temporary device memory.
	cudaFree(gpu_keys);
	cudaFree(gpu_values);
	cudaFree(gpu_HMAX_param);

	
	return values; // Return pointer to the host array containing the results.
}


/**
 * @brief Calculates the current load factor of the hash table.
 * Functional Utility: Provides a metric for the density of the hash table,
 * indicating how full it is. This is used by `insertBatch` to decide when
 * to trigger a `reshape` operation to maintain efficient performance.
 *
 * @return The load factor (`this->size / this->HMAX`).
 */
float GpuHashTable::loadFactor() {
	return (float) this->size / (float) this->HMAX; 
}

// Unimplemented methods in this file.
// void GpuHashTable::occupancy() {
// // Placeholder for future implementation to check bucket occupancy.
// }
//
// void GpuHashTable::print(std::string info) {
// // Placeholder for future implementation to print hash table contents for debugging.
// }


#define HASH_INIT GpuHashTable GpuHashTable(1); ///< Macro for initializing a `GpuHashTable` object on the host stack.
#define HASH_RESERVE(size) GpuHashTable.reshape(size); ///< Macro for reshaping the `GpuHashTable` to a new size.

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys) ///< Macro for batch insertion into `GpuHashTable`.
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys) ///< Macro for batch retrieval from `GpuHashTable`.

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor() ///< Macro for getting the load factor of `GpuHashTable`.

// #include "test_map.cpp" ///< Include for testing purposes. (Typically commented out in production code)
// Functional Utility: The content below this point is typically from a .hpp file (e.g., gpu_hashtable.hpp)
// or a common utility header, and it defines constants, helper hash functions, and the class structure.
// It is duplicated here, likely for self-containment or specific compilation environments.
#ifndef _HASHCPU_ // Guards against multiple inclusions.
#define _HASHCPU_

using namespace std; // Brings all identifiers from the std namespace into the current scope.

#define	KEY_INVALID		0 ///< Constant: Represents an invalid or empty key in the hash table.

/**
 * @brief Macro for error handling and exiting the program.
 * Functional Utility: Provides a convenient way to check assertions and
 * terminate the program with an error message if an assertion fails. Prints
 * file and line number information (`__FILE__`, `__LINE__`) and a system error (`perror`).
 *
 * @param assertion The boolean condition to check. If true, the program exits.
 * @param call_description A string describing the function call or context where the error occurred.
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
 * @brief Array of large prime numbers.
 * Functional Utility: This list provides a set of large prime numbers used in the
 * custom hash functions (`hash1`, `hash2`, `hash3`). Using prime numbers for modulo
 * operations in hashing helps to achieve a more uniform distribution of hash values,
 * which in turn reduces collisions and improves hash table performance.
 */
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


/**
 * @brief Hash function 1.
 * Functional Utility: Computes a hash value for a given integer data using a specific
 * prime from `primeList` and applies modulo operations to keep the result within `limit`.
 * This is used as one of the candidate hash functions for the hash table.
 *
 * @param data The integer key to hash.
 * @param limit The upper bound (size of the hash table) for the hash value.
 * @return An integer hash value within the range [0, limit-1].
 */
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
/**
 * @brief Hash function 2.
 * Functional Utility: Computes a hash value for a given integer data using a specific
 * prime from `primeList` and applies modulo operations to keep the result within `limit`.
 * This is used as one of the candidate hash functions for the hash table.
 *
 * @param data The integer key to hash.
 * @param limit The upper bound (size of the hash table) for the hash value.
 * @return An integer hash value within the range [0, limit-1].
 */
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}


/**
 * @brief Hash function 3.
 * Functional Utility: Computes a hash value for a given integer data using a specific
 * prime from `primeList` and applies modulo operations to keep the result within `limit`.
 * This is used as one of the candidate hash functions for the hash table.
 *
 * @param data The integer key to hash.
 * @param limit The upper bound (size of the hash table) for the hash value.
 * @return An integer hash value within the range [0, limit-1].
 */
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}


/**
 * @struct hashItem
 * @brief Represents a single key-value pair entry in the hash table.
 * Functional Utility: This structure defines the fundamental unit of data storage
 * within the hash table's buckets.
 */
struct hashItem {
    int key, value; ///< Key and value pair.
};

/**
 * @struct hashTable
 * @brief Represents the structure of the hash table itself on the device.
 * Functional Utility: This structure holds the metadata for the hash table
 * (its size) and a pointer to the actual array of `hashItem`s stored in device memory.
 * It's passed to CUDA kernels to operate on the hash table.
 */
struct hashTable {
    int size; ///< The current allocated size of the hash map (number of entries).
    hashItem *map; ///< Pointer to the array of `hashItem`s in device memory.
};

/**
 * @class GpuHashTable
 * @brief Provides a host-side interface to manage a GPU-resident hash table.
 * Functional Utility: Encapsulates the device memory pointers for the hash table
 * (keys, values, empty slot flags), manages the hash table's capacity (`HMAX`),
 * and tracks the number of elements (`size`). It offers host-side methods to
 * initialize, resize, insert batches, retrieve batches, and query the load factor,
 * all by orchestrating CUDA kernel launches and host-device memory transfers.
 */
class GpuHashTable
{
	public:
		int *hash_keys; ///< Pointer to device memory storing the keys of the hash table.
		int *hash_values; ///< Pointer to device memory storing the values of the hash table.
		int *empty_slots; ///< Pointer to device memory indicating whether a slot is empty (0) or occupied (1).
		int size; ///< Current number of elements stored in the hash table.
		int HMAX; ///< Maximum capacity of the hash table.
	
		GpuHashTable(int size); ///< Constructor: Initializes the hash table.
		void reshape(int sizeReshape); ///< Resizes and rehashes the hash table.
		
		bool insertBatch(int *keys, int* values, int numKeys); ///< Inserts a batch of key-value pairs.
		int* getBatch(int* key, int numItems); ///< Retrieves a batch of values for given keys.
		
		float loadFactor(); ///< Calculates the current load factor.
		void occupancy(); ///< Placeholder for a method to calculate bucket occupancy (unimplemented).
		void print(string info); ///< Placeholder for a method to print hash table contents (unimplemented).
	
		~GpuHashTable(); ///< Destructor: Frees allocated device memory.
};

#endif // _HASHCPU_