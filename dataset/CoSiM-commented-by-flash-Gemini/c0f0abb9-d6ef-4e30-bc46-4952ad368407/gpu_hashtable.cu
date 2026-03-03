/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA.
 *
 * This module provides a `GpuHashTable` class that supports insertion, retrieval,
 * and dynamic resizing (reshaping) of a hash table on a CUDA-enabled GPU.
 * It utilizes CUDA kernels for parallel operations and atomic functions
 * for thread-safe access and collision resolution in a parallel environment.
 * The hash table employs open addressing with linear probing for collision handling.
 *
 * Key features:
 * - Parallel `get`, `insert`, and `reshape` operations implemented as CUDA kernels.
 * - `atomicCAS` for thread-safe updates to hash table slots.
 * - Dynamic resizing of the hash table based on load factor thresholds.
 * - Host-side interface (`GpuHashTable` class) for managing GPU memory and launching kernels.
 *
 * @attention This implementation assumes integer keys and values.
 *
 * HPC & Parallelism Considerations:
 * - Memory Hierarchy: The main hash table (`hash_list`) resides in global device memory.
 * - Thread Indexing: Standard CUDA linear indexing (`blockIdx.x * blockDim.x + threadIdx.x`) is used.
 * - Synchronization: `atomicCAS` is critical for ensuring correctness during concurrent
 *   insertions and rehashing by multiple threads accessing the same global memory locations.
 *   `cudaDeviceSynchronize()` is used to ensure host-side code waits for GPU operations.
 * - Collision Resolution: Linear probing is used, which can suffer from primary clustering
 *   in parallel environments if not managed carefully (e.g., through load factor control).
 */

// Standard C++ headers
#include <iostream>   // For standard input/output operations (e.g., std::cerr)
#include <vector>     // For dynamic arrays (though not directly used in the .cu file)
#include <string>     // For string manipulation (e.g., std::string)
#include <algorithm>  // For algorithms like std::max (though not explicitly used here)
#include <numeric>    // For numerical operations (though not explicitly used here)
#include <cstdio>     // For C-style input/output (e.g., fprintf, perror)
#include <cstdlib>    // For general utilities (e.g., exit, abs)
#include <cstring>    // For memory manipulation (e.g., memset)
#include <cerrno>     // For error number definitions (e.g., errno)

// CUDA specific headers
#include <cuda_runtime.h> // For CUDA runtime API functions (e.g., cudaMalloc, cudaMemcpy, cudaFree)
#include <device_launch_parameters.h> // For __global__ and __device__ keywords

// Project-specific header
#include "gpu_hashtable.hpp" // Contains declarations for GpuHashTable class, HashItem, GPU_Table structs.


/**
 * @brief CUDA kernel to perform parallel lookups (get operations) in the hash table.
 *
 * Each thread in the grid is responsible for looking up a specific key.
 * It uses open addressing with linear probing to find the key. The search
 * starts from the hash-computed position and wraps around if necessary.
 *
 * @param keys Pointer to an array of keys to lookup on the device (global memory).
 * @param values Pointer to an array where found values will be stored on the device (global memory).
 * @param numKeys The total number of keys to lookup.
 * @param hashh The GPU_Table structure representing the hash table on the device.
 *
 * HPC & Parallelism:
 * - Thread Indexing: `idx = blockIdx.x * blockDim.x + threadIdx.x` computes a unique linear thread ID.
 * - Memory Access: Accesses `keys`, `values`, and `hashh.hash_list` in global device memory.
 * - Collision Resolution: Linear probing is implemented in a loop, potentially leading to
 *   divergence if threads encounter different collision chains.
 */
__global__ void kernel_get(int* keys, int* values, int numKeys, GPU_Table hashh) {
    // Calculate a unique linear index for the current thread within the grid.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Compute the initial hash position for the key.
	int pos_new = hash11(keys[idx], hashh.hash_size);
	
    // Guard clause: ensure thread index is within bounds of valid keys.
	if(idx >= numKeys)
		return;

    // Block Logic: Linear probing search - search from the initial hash position to the end of the table.
    // Invariant: The loop iterates through potential slots in the hash table.
	for (int i = pos_new; i < hashh.hash_size; i++) {
        // If the key at the current slot matches the target key, store its value and return.
		if(hashh.hash_list[i].key == keys[idx]) {
			values[idx] = hashh.hash_list[i].value;
			return;
		}
	}

    // Block Logic: Linear probing search (wrap-around) - continue search from beginning of the table to pos_new.
    // This handles cases where the key might have wrapped around due to linear probing.
	for (int i = 0; i < pos_new; i++) {
        // If the key at the current slot matches the target key, store its value and return.
		if(hashh.hash_list[i].key == keys[idx]) {
			values[idx] = hashh.hash_list[i].value;
			return;
		}
	}
}


/**
 * @brief CUDA kernel to perform parallel insertions or updates in the hash table.
 *
 * Each thread in the grid is responsible for inserting or updating a specific key-value pair.
 * It uses open addressing with linear probing and `atomicCAS` for thread-safe operations.
 * If the key already exists, its value is updated. If a slot is empty, the key is inserted.
 *
 * @param keys Pointer to an array of keys to insert/update on the device (global memory).
 * @param values Pointer to an array of values corresponding to the keys (global memory).
 * @param numKeys The total number of key-value pairs to process.
 * @param hashh The GPU_Table structure representing the hash table on the device.
 *
 * HPC & Parallelism:
 * - Thread Indexing: `idx = blockIdx.x * blockDim.x + threadIdx.x` computes a unique linear thread ID.
 * - Memory Access: Accesses `keys`, `values`, and `hashh.hash_list` in global device memory.
 * - Synchronization: `atomicCAS` is crucial here. It atomically compares a memory location's
 *   value with an expected value and, if they match, swaps it with a new value. This
 *   ensures that only one thread successfully writes to an empty slot (`KEY_INVALID`)
 *   or updates a key's value without race conditions.
 * - Collision Resolution: Linear probing means threads might contend for slots or follow
 *   long chains. `atomicCAS` helps manage concurrent access to these slots.
 */
__global__ void kernel_insert(int* keys, int* values, int numKeys, GPU_Table hashh) {
    // Calculate a unique linear index for the current thread within the grid.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int key = keys[idx];     // Key to insert/update.
	int value = values[idx]; // Value to associate with the key.
    // Compute the initial hash position for the key.
	int pos_new = hash11(key, hashh.hash_size);
	int ok;                  // Variable to store the result of atomicCAS.

    // Guard clause: ensure thread index is within bounds of valid keys.
	if(idx >= numKeys)
		return;
	
    // Block Logic: Linear probing search and insertion - from hash position to end of table.
    // Invariant: The loop tries to find an existing key or an empty slot for insertion.
	for (int i = pos_new; i < hashh.hash_size; i++) {
		
        // If the key at the current slot matches the target key, update its value and return.
		if(key == hashh.hash_list[i].key) {
			hashh.hash_list[i].value = value;
			return;
		}

        // Attempt to atomically insert the key if the slot is empty (KEY_INVALID).
        // `atomicCAS` returns the old value of `hashh.hash_list[i].key`.
		ok = atomicCAS(&hashh.hash_list[i].key, KEY_INVALID, key);
        // If the old value was `KEY_INVALID`, it means this thread successfully acquired the slot.
		if(ok == KEY_INVALID) {
			hashh.hash_list[i].value = value;
			return;
		}
	}

    // Block Logic: Linear probing search and insertion (wrap-around) - from beginning to pos_new.
    // This handles cases where the key might hash to a position that required wrapping around.
	for (int i = 0; i < pos_new; i++) {
		
        // If the key at the current slot matches the target key, update its value and return.
		if(key == hashh.hash_list[i].key) {
			hashh.hash_list[i].value = value;
			return;
		}

        // Attempt to atomically insert the key if the slot is empty (KEY_INVALID).
		ok = atomicCAS(&hashh.hash_list[i].key, KEY_INVALID, key);
        // If the old value was `KEY_INVALID`, it means this thread successfully acquired the slot.
		if(ok == KEY_INVALID) {
			hashh.hash_list[i].value = value;
			return;
		}
	}
}


/**
 * @brief CUDA kernel to reshape (rehash) elements from an old hash table to a new one.
 *
 * Each thread in the grid is responsible for taking one item from the old hash table
 * and inserting it into the new hash table. This is used during resizing operations.
 * It uses open addressing with linear probing and `atomicCAS` to handle collisions
 * and ensure thread-safe insertion into the new table.
 *
 * @param hashh The GPU_Table structure representing the old hash table on the device.
 * @param new_hash The GPU_Table structure representing the new hash table on the device.
 *
 * HPC & Parallelism:
 * - Thread Indexing: `idx = blockIdx.x * blockDim.x + threadIdx.x` computes a unique linear thread ID.
 * - Memory Access: Accesses `hashh.hash_list` (old table) and `new_hash.hash_list` (new table)
 *   in global device memory.
 * - Synchronization: `atomicCAS` is essential for collision resolution when multiple
 *   threads attempt to insert their old items into potentially the same new hash table slots.
 *   This ensures only one thread succeeds in writing to an empty slot.
 * - Collision Resolution: Similar to `kernel_insert`, linear probing is used.
 */
__global__ void kernel_reshape(GPU_Table hashh, GPU_Table new_hash) {
    // Calculate a unique linear index for the current thread within the grid.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Retrieve the hash item from the old hash table.
	HashItem old = hashh.hash_list[idx];
	int ok; // Variable to store the result of atomicCAS.

    // Compute the initial hash position for the old item's key in the new hash table.
	int pos_new = hash11(old.key, new_hash.hash_size);

    // Guard clause: ensure thread index is within bounds of the old hash table size
    // and that the key is valid (not an empty slot).
	if(idx >= hashh.hash_size || old.key == KEY_INVALID)
		return;

    // Block Logic: Linear probing search and insertion into the new hash table - from hash position to end.
    // Invariant: The loop tries to find an empty slot in the new hash table for the old item.
	for (int i = pos_new; i < new_hash.hash_size; i++) {
        // Attempt to atomically insert the old item's key into an empty slot (`KEY_INVALID`).
		ok = atomicCAS(&new_hash.hash_list[i].key, KEY_INVALID, old.key);
        // If successful, set the corresponding value and return.
		if(ok == KEY_INVALID) {
			new_hash.hash_list[i].value = old.value;
			return;
		}
	}

    // Block Logic: Linear probing search and insertion (wrap-around) into the new hash table - from beginning to pos_new.
    // This handles cases where the rehashed item needs to wrap around in the new table.
	for (int i = 0; i < pos_new; i++) {
        // Attempt to atomically insert the old item's key into an empty slot (`KEY_INVALID`).
		ok = atomicCAS(&new_hash.hash_list[i].key, KEY_INVALID, old.key);
        // If successful, set the corresponding value and return.
		if(ok == KEY_INVALID) {
			new_hash.hash_list[i].value = old.value;
			return;
		}
	}
}


/**
 * @brief Host-side class providing an interface to a GPU-accelerated hash table.
 *
 * This class manages the GPU memory for the hash table and orchestrates the
 * launching of CUDA kernels for batch operations (insert, get) and reshaping.
 */
class GpuHashTable
{
	GPU_Table hashh; // The GPU_Table structure representing the hash table on the device.

	public:
		/**
		 * @brief Constructor for GpuHashTable.
		 * Allocates and initializes GPU memory for the hash table.
		 *
		 * @param size The desired initial size (number of buckets) for the hash table.
		 */
		GpuHashTable(int size) {
			cudaError_t err; // Variable to store CUDA error codes.
			hashh.hash_size = size;      // Set the size of the hash table.
			hashh.nr_elems = 0;          // Initialize element count to zero.
			hashh.hash_list = NULL;      // Initialize hash_list pointer.

			// Allocate global device memory for the hash table (array of HashItem structs).
			err = cudaMalloc((void**)&hashh.hash_list, size * sizeof(struct HashItem));
			if(err != cudaSuccess)
				std::cerr << "cudaMalloc failed in GpuHashTable constructor: " << cudaGetErrorString(err) << "\n";

			// Initialize all hash table slots by setting their keys to KEY_INVALID (0).
			err = cudaMemset(hashh.hash_list, 0, size * sizeof(struct HashItem));
			if(err != cudaSuccess)
				std::cerr << "cudaMemset failed in GpuHashTable constructor: " << cudaGetErrorString(err) << "\n";
		}

		/**
		 * @brief Destructor for GpuHashTable.
		 * Frees the GPU memory allocated for the hash table.
		 */
		~GpuHashTable() {
			cudaError_t err; // Variable to store CUDA error codes.
			// Free the global device memory previously allocated for the hash table.
			err = cudaFree(hashh.hash_list);
			if(err != cudaSuccess)
				std::cerr << "cudaFree failed in GpuHashTable destructor: " << cudaGetErrorString(err) << "\n";
		}

		/**
		 * @brief Reshapes (resizes) the hash table to a new number of buckets.
		 *
		 * This involves allocating a new larger table, transferring existing elements
		 * from the old table to the new one using a CUDA kernel, and then freeing
		 * the old table.
		 *
		 * @param numBucketsReshape The desired new size (number of buckets) for the hash table.
		 */
		void reshape(int numBucketsReshape) {
			cudaError_t err; // Variable to store CUDA error codes.
			GPU_Table new_hash; // Structure to represent the new hash table.
			new_hash.hash_size = numBucketsReshape; // Set the size of the new hash table.
			new_hash.nr_elems = hashh.nr_elems;     // Copy the number of elements from the old table.
			int nr_blocks;                           // Number of CUDA blocks to launch.

			// Calculate the number of blocks needed based on numBucketsReshape and BLOCK_DIM.
			// Ensures enough blocks to cover all elements of the old hash table during reshape.
			if(numBucketsReshape % BLOCK_DIM != 0)
				nr_blocks = numBucketsReshape / BLOCK_DIM + 1;
			else
				nr_blocks = numBucketsReshape / BLOCK_DIM;

			// Allocate global device memory for the new hash table.
			err = cudaMalloc((void**)&new_hash.hash_list, numBucketsReshape *
														sizeof(struct HashItem));
			if(err != cudaSuccess)
				std::cerr << "cudaMalloc failed in reshape for new hash: " << cudaGetErrorString(err) << "\n";

			// Initialize all slots in the new hash table by setting keys to KEY_INVALID.
			err = cudaMemset(new_hash.hash_list, 0, numBucketsReshape *
														sizeof(struct HashItem));
			if(err != cudaSuccess)
				std::cerr << "cudaMemset failed in reshape for new hash: " << cudaGetErrorString(err) << "\n";
			
			// Launch the kernel_reshape to transfer elements from the old table to the new one.
			// Each thread processes one item from the old hash table.
			kernel_reshape<<>>(hashh, new_hash);

			// Synchronize the device to ensure all kernel operations are complete before proceeding on the host.
			err = cudaDeviceSynchronize();
			if(err != cudaSuccess)
				std::cerr << "cudaDeviceSynchronize failed after kernel_reshape: " << cudaGetErrorString(err) << "\n";

			// Free the global device memory of the old hash table.
			err = cudaFree(hashh.hash_list);
			if(err != cudaSuccess)
				std::cerr << "cudaFree failed for old hash_list in reshape: " << cudaGetErrorString(err) << "\n";

			hashh = new_hash; // Update the host-side GPU_Table struct to point to the new hash table.
		}

		/**
		 * @brief Inserts a batch of key-value pairs into the hash table on the GPU.
		 *
		 * This function handles memory allocation for keys/values on the device,
		 * performs data transfer from host to device, checks the load factor
		 * and triggers a reshape if necessary, launches the `kernel_insert`,
		 * and finally frees the temporary device memory.
		 *
		 * @param keys Pointer to an array of keys on the host to insert.
		 * @param values Pointer to an array of values on the host to insert.
		 * @param numKeys The number of key-value pairs in the batch.
		 * @return true on success.
		 */
		bool insertBatch(int *keys, int* values, int numKeys) {
			int* device_values;   // Pointer for values array on the device.
			cudaError_t err1, err2; // Variables to store CUDA error codes.
			int* device_keys;     // Pointer for keys array on the device.
			int nr_blocks;        // Number of CUDA blocks to launch.

			// Calculate the number of blocks needed based on numKeys and BLOCK_DIM.
			if(numKeys % BLOCK_DIM != 0)
				nr_blocks = numKeys / BLOCK_DIM + 1;
			else
				nr_blocks = numKeys / BLOCK_DIM;

			// Allocate global device memory for the keys and values arrays.
			err1 = cudaMalloc((void**)&device_values, numKeys * sizeof(int));
			err2 = cudaMalloc((void**)&device_keys, numKeys * sizeof(int));
			if(err1 != cudaSuccess || err2 != cudaSuccess)
				std::cerr << "cudaMalloc failed in insertBatch: " << cudaGetErrorString(err1) << " " << cudaGetErrorString(err2) << "\n";

			// Copy keys and values from host memory to device global memory.
			err1 = cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
			err2 = cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
			if(err1 != cudaSuccess || err2 != cudaSuccess)
				std::cerr << "cudaMemcpy failed in insertBatch: " << cudaGetErrorString(err1) << " " << cudaGetErrorString(err2) << "\n";

			// Check load factor and reshape if it exceeds the threshold (0.85).
			if((float)(hashh.nr_elems + numKeys) / hashh.hash_size > 0.85f)
				reshape((int)((hashh.nr_elems + numKeys) / 0.8f)); // Reshape to a new size based on desired load factor.

			// Launch the kernel_insert to perform parallel insertions/updates.
			kernel_insert<<>>(device_keys, device_values, numKeys, hashh);

			// Synchronize the device to ensure all kernel operations are complete.
			err1 = cudaDeviceSynchronize();
			if(err1 != cudaSuccess )
				std::cerr << "cudaDeviceSynchronize failed after kernel_insert: " << cudaGetErrorString(err1) << "\n";
			
			// Update the host-side count of elements in the hash table.
			hashh.nr_elems += numKeys;

			// Free the temporary global device memory allocated for keys and values.
			err1 = cudaFree(device_keys);
			err2 = cudaFree(device_values);
			if(err1 != cudaSuccess || err2 != cudaSuccess)
				std::cerr << "cudaFree failed for device_keys/device_values in insertBatch: " << cudaGetErrorString(err1) << " " << cudaGetErrorString(err2) << "\n";

			return true;
		}

		/**
		 * @brief Retrieves a batch of values for given keys from the hash table on the GPU.
		 *
		 * This function allocates device memory for keys and values, transfers keys
		 * from host to device, initializes device values, launches the `kernel_get`,
		 * transfers results back to the host, and finally frees temporary device memory.
		 *
		 * @param keys Pointer to an array of keys on the host to lookup.
		 * @param numKeys The number of keys in the batch.
		 * @return int* A pointer to an array of retrieved values on the host.
		 *         The caller is responsible for freeing this host memory using `free()`.
		 */
		int* getBatch(int* keys, int numKeys) {
			cudaError_t err1, err2; // Variables to store CUDA error codes.
			int* host_values;       // Pointer for values array on the host.
			int* device_values;     // Pointer for values array on the device.
			int* device_keys;       // Pointer for keys array on the device.
			int nr_blocks;          // Number of CUDA blocks to launch.

			// Calculate the number of blocks needed based on numKeys and BLOCK_DIM.
			if(numKeys % BLOCK_DIM != 0)
				nr_blocks = numKeys / BLOCK_DIM + 1;
			else
				nr_blocks = numKeys / BLOCK_DIM;

			// Allocate global device memory for the keys and values arrays.
			err1 = cudaMalloc((void**)&device_values, numKeys * sizeof(int));
			err2 = cudaMalloc((void**)&device_keys, numKeys * sizeof(int));
			if(err1 != cudaSuccess || err2 != cudaSuccess)
				std::cerr << "cudaMalloc failed in getBatch: " << cudaGetErrorString(err1) << " " << cudaGetErrorString(err2) << "\n";
			
			// Copy keys from host memory to device global memory.
			err1 = cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
			if(err1 != cudaSuccess )
				std::cerr << "cudaMemcpy failed for device_keys in getBatch: " << cudaGetErrorString(err1) << "\n";
			
			// Initialize device_values to 0. (A default for not-found keys might be 0, or KEY_INVALID from elsewhere).
			err1  = cudaMemset(device_values, 0, numKeys * sizeof(int));
			if(err1 != cudaSuccess )
				std::cerr << "cudaMemset failed for device_values in getBatch: " << cudaGetErrorString(err1) << "\n";
			
			// Allocate host memory for the retrieved values.
			host_values = (int*)malloc(numKeys * sizeof(int));

			// Launch the kernel_get to perform parallel lookups.
			kernel_get<<>>(device_keys, device_values, numKeys, hashh);
			
			// Synchronize the device to ensure all kernel operations are complete.
			err1 = cudaDeviceSynchronize();
			if(err1 != cudaSuccess )
				std::cerr << "cudaDeviceSynchronize failed after kernel_get: " << cudaGetErrorString(err1) << "\n";
			
			// Copy the retrieved values from device global memory back to host memory.
			err1 = cudaMemcpy(host_values, device_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
			if(err1 != cudaSuccess)
				std::cerr <<"cudaMemcpy failed for host_values in getBatch: " << cudaGetErrorString(err1) << "\n";

			// Free the temporary global device memory allocated for keys and values.
			err1 = cudaFree(device_keys);
			err2 = cudaFree(device_values);
			if(err1 != cudaSuccess || err2 != cudaSuccess)
				std::cerr << "cudaFree failed for device_keys/device_values in getBatch: " << cudaGetErrorString(err1) << " " << cudaGetErrorString(err2) << "\n";

			return host_values;
		}

		/**
		 * @brief Calculates the current load factor of the hash table.
		 *
		 * The load factor is the ratio of the number of elements to the total
		 * number of available hash table slots (size).
		 *
		 * @return float The current load factor. Returns 0 if hash table size is 0.
		 */
		float loadFactor() {
			if(hashh.hash_size == 0)
				return 0;
			else
				return (float)hashh.nr_elems / hashh.hash_size;
		}

		// Placeholder for future implementation.
		void occupancy();
		// Placeholder for future implementation.
		void print(string info);
};

// --- Convenience Macros for GpuHashTable Operations ---
// These macros provide a simplified interface to the GpuHashTable class methods.
#define HASH_INIT GpuHashTable GpuHashTable(1); // Initializes a GpuHashTable object named `GpuHashTable` with size 1.
#define HASH_RESERVE(size) GpuHashTable.reshape(size); // Reshapes the `GpuHashTable` to a new size.

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys) // Inserts a batch of items.
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys) // Retrieves a batch of values.

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor() // Gets the current load factor.

// This include likely brings in test definitions for the hash table.
// In competitive programming or library development, tests might be
// included directly in the source file or a separate compilation unit.
#include "test_map.cpp"

// --- Global Constants and Helper Functions ---
// These definitions are often in a header file (e.g., gpu_hashtable.hpp) but are
// sometimes duplicated or fully defined in the .cu file.
#ifndef _HASHCPU_ // Include guard to prevent multiple definitions.
#define _HASHCPU_


using namespace std; // Using standard namespace.

#define	KEY_INVALID		0 // Macro representing an invalid or empty key in a hash table slot.
#define BLOCK_DIM 1024 // Define the number of threads per CUDA block.
#define PRIME_ONE 11llu // First prime number for hash function (unsigned long long).
#define PRIME_TWO 171262457903llu // Second prime number for hash function (unsigned long long).

/**
 * @brief Macro for error checking and program termination.
 * @param assertion A boolean expression. If true, an error is reported and the program exits.
 * @param call_description A string describing the failed call.
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
 * @brief Structure representing a single item (key-value pair) in the hash table.
 */
struct HashItem {
   int value;   // The value associated with the key.
   int key;     // The key of the item.
};

/**
 * @brief Structure representing the hash table on the GPU.
 * This structure holds metadata about the hash table, including its size,
 * number of elements, and a pointer to the array of HashItem structs in global memory.
 */
struct GPU_Table {
	int hash_size;    // Total number of buckets/slots in the hash table.
	int nr_elems;     // Current number of elements stored in the hash table.
	HashItem* hash_list; // Pointer to the array of HashItem structs in GPU global memory.
};

// A list of prime numbers, potentially used for hash functions or to find
// suitable sizes for hash tables during resizing.
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
