/**
 * @file gpu_hashtable.cu
 * @brief CUDA C++ implementation of a hash table optimized for GPU execution.
 *
 * This file provides a `GpuHashTable` class that leverages CUDA to perform
 * hash table operations (insertion, retrieval, reshaping) in parallel on the GPU.
 * It uses open addressing with linear probing for collision resolution and `atomicCAS`
 * for thread-safe concurrent modifications to the hash table.
 *
 * Domain-Specific Awareness (HPC & Parallelism, Performance Optimization):
 * - Memory Hierarchy: Utilizes `cudaMalloc`, `cudaMemset`, `cudaMemcpy` for efficient
 *   device memory management and data transfer between host and device.
 * - Thread Indexing: Kernels (`kernel_reshape`, `kernel_insert`, `kernel_get`) use
 *   `blockIdx.x` and `threadIdx.x` to parallelize operations across the hash table.
 * - Synchronization: `atomicCAS` ensures atomic updates to hash table entries,
 *   critical for concurrent operations in a lock-free manner on the GPU.
 * - Algorithm: Open addressing with linear probing. The hash table is reshaped (resized)
 *   when the load factor reaches `MAXIMUM_LOAD_FACTOR` to maintain performance.
 *
 * Time Complexity:
 * - `GpuHashTable` constructor/destructor: O(size) for memory allocation/deallocation.
 * - `device_hash`: O(1) per call.
 * - `kernel_reshape`: O(old_size) in parallel for `old_size` threads. Collision resolution
 *   adds a factor proportional to probe length.
 * - `kernel_insert`: O(numKeys) in parallel for `numKeys` threads. Collision resolution
 *   adds a factor proportional to probe length.
 * - `kernel_get`: O(numKeys) in parallel for `numKeys` threads. Collision resolution
 *   adds a factor proportional to probe length.
 * - Host methods (`reshape`, `insertBatch`, `getBatch`): Involve `cudaMemcpy` (O(N) for N elements)
 *    and kernel launches (parallel time complexity of kernels).
 *
 * Space Complexity: O(size) for the hash table on the device.
 */

// Standard C++ headers
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <map>

// CUDA specific headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Custom header for GpuHashTable definitions (expected to contain entry struct and class declaration)
#include "gpu_hashtable.hpp"

// Prime numbers used in hash function to ensure good distribution.
#define FIRST_PRIME 13169977
#define SECOND_PRIME 5351951779

/**
 * @brief Device-side hash function.
 * Computes a hash value for a given data key, mapping it to a bucket within a limit.
 * Uses a multiplicative hash approach with large prime numbers.
 * @param data The integer key to hash.
 * @param limit The upper bound for the hash value (number of buckets).
 * @return The computed hash index.
 */
__device__ int device_hash(int data, int limit)
{
	return ((long long)abs(data) * FIRST_PRIME) % SECOND_PRIME % limit;
}

/**
 * @brief CUDA kernel for inserting a batch of key-value pairs into the hash table.
 * Each thread attempts to insert one key-value pair. Uses `atomicCAS` for
 * thread-safe insertion and linear probing for collision resolution.
 * @param keys Pointer to an array of keys to insert on the device.
 * @param values Pointer to an array of values to insert on the device.
 * @param numKeys The number of key-value pairs to insert.
 * @param hash_table Pointer to the hash table on the device.
 * @param size The current size of the hash table.
 */
__global__ void kernel_insert(int *keys, int *values, int numKeys,
			      entry *hash_table, int size)
{
	// Calculate global thread ID. Each thread processes one key-value pair.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, hash_value, old_value;

	// Pre-condition: Ensure thread ID is within the bounds of `numKeys`.
	if (idx >= numKeys)
		return;

	// Invariant: Compute initial hash for the current key.
	hash_value = device_hash(keys[idx], size);

	// Block Logic: Linear probing starting from the calculated hash value, wrapping around if necessary.
	// First pass: Probe from `hash_value` to the end of the table.
	for (i = hash_value; i < size; i++) {
		// Atomically compare and swap: attempt to write the key if the slot is KEY_INVALID.
		// `old_value` will contain the value of `hash_table[i].key` *before* the atomic operation.
		old_value = atomicCAS(&hash_table[i].key, KEY_INVALID, keys[idx]);
		// If `old_value` was `KEY_INVALID`, our key was successfully inserted.
		// If `old_value` was equal to `keys[idx]`, it means another thread already inserted this key (duplicate).
		if (old_value == keys[idx] || old_value == KEY_INVALID) {
			hash_table[i].value = values[idx]; // Write the corresponding value.
			return; // Successfully inserted or duplicate handled.
		}
	}
	// Second pass: Probe from the beginning of the table to `hash_value` (wraparound).
	for (i = 0; i < hash_value; i++) {
		old_value = atomicCAS(&hash_table[i].key, KEY_INVALID, keys[idx]);
		
		if (old_value == keys[idx] || old_value == KEY_INVALID) {
			hash_table[i].value = values[idx]; // Write the corresponding value.
			return; // Successfully inserted or duplicate handled.
		}
	}
}

/**
 * @brief CUDA kernel for retrieving values for a batch of keys from the hash table.
 * Each thread attempts to retrieve one value. Uses linear probing to find keys.
 * @param keys Pointer to an array of keys to search for on the device.
 * @param values Pointer to an array on the device where retrieved values will be stored.
 * @param numKeys The number of keys to retrieve.
 * @param hash_table Pointer to the hash table on the device.
 * @param size The current size of the hash table.
 */
__global__ void kernel_get(int *keys, int *values, int numKeys,
			   entry *hash_table, int size)
{
	// Calculate global thread ID. Each thread processes one key lookup.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, hash_value;

	// Pre-condition: Ensure thread ID is within the bounds of `numKeys`.
	if (idx >= numKeys)
		return;

	// Invariant: Compute initial hash for the current key.
	hash_value = device_hash(keys[idx], size);

	// Block Logic: Linear probing starting from the calculated hash value, wrapping around if necessary.
	// First pass: Probe from `hash_value` to the end of the table.
	for (i = hash_value; i < size; i++) {
		// If key matches, store the value and break.
		if (hash_table[i].key == keys[idx]) {
			values[idx] = hash_table[i].value;
			return;
		}
		// If an empty slot (KEY_INVALID) is found, the key is not in the table.
		if (hash_table[i].key == KEY_INVALID) {
			values[idx] = KEY_INVALID; // Indicate key not found.
			return;
		}
	}

	// Second pass: Probe from the beginning of the table to `hash_value` (wraparound).
	for (i = 0; i < hash_value; i++) {
		if (hash_table[i].key == keys[idx]) {
			values[idx] = hash_table[i].value;
			return;
		}
		if (hash_table[i].key == KEY_INVALID) {
			values[idx] = KEY_INVALID; // Indicate key not found.
			return;
		}
	}
}

/**
 * @brief CUDA kernel for reshaping (resizing) the hash table.
 * Copies existing valid entries from an old hash table to a new, larger one.
 * Uses `atomicCAS` for concurrent insertion into the new table, resolving collisions
 * with linear probing.
 * @param hash_table Pointer to the old hash table on the device.
 * @param new_hash_table Pointer to the new, larger hash table on the device.
 * @param old_size The size of the old hash table.
 * @param new_size The size of the new hash table.
 */
__global__ void kernel_reshape(entry *hash_table, entry *new_hash_table,
			       int old_size, int new_size)
{
	// Calculate global thread ID. Each thread processes one entry from the old hash table.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, new_hash, old_value;

	// Pre-condition: Ensure thread ID is within the bounds of the old hash table size
	// and that the entry in the old hash table is valid.
	if (idx >= old_size || hash_table[idx].key == KEY_INVALID)
		return;

	// Invariant: Compute initial hash for the key in the new hash table.
	new_hash = device_hash(hash_table[idx].key, new_size);

	// Block Logic: Linear probing for collision resolution in the new hash table.
	// First pass: Probe from `new_hash` to the end of the new table.
	for (i = new_hash; i < new_size; i++) {
		// Atomically compare and swap: attempt to write the key if the slot is KEY_INVALID.
		old_value = atomicCAS(&new_hash_table[i].key, KEY_INVALID,
				      hash_table[idx].key);
		
		// If `old_value` was `KEY_INVALID`, our key was successfully inserted.
		if (old_value == KEY_INVALID) {
			new_hash_table[i].value = hash_table[idx].value; // Write the corresponding value.
			return; // Successfully inserted.
		}
		// If `old_value` was not `KEY_INVALID`, it means another key is already in this slot.
		// Collision: Continue probing.
	}
	// Second pass: Probe from the beginning of the new table to `new_hash` (wraparound).
	for (i = 0; i < new_hash; i++) {
		old_value = atomicCAS(&new_hash_table[i].key, KEY_INVALID,
				      hash_table[idx].key);
		
		if (old_value == KEY_INVALID) {
			new_hash_table[i].value = hash_table[idx].value; // Write the corresponding value.
			return; // Successfully inserted.
		}
	}
}


/**
 * @brief Constructor for `GpuHashTable`.
 * Allocates device memory for the hash table and initializes it with `KEY_INVALID`.
 * @param size The initial size (number of buckets) for the hash table.
 */
GpuHashTable::GpuHashTable(int size)
{
	cudaError_t err;

	// Initialize number of inserted pairs and hash table size.
	this->numberInsertedPairs = 0;
	this->size = size;

	// Allocate device memory for the hash table entries.
	err = cudaMalloc(&this->hash_table, size * sizeof(entry));
	DIE(err != cudaSuccess, "cudaMalloc init"); // Error checking macro.

	// Initialize all hash table entries with 0 (which is KEY_INVALID) to indicate empty slots.
	err = cudaMemset(this->hash_table, 0, size * sizeof(entry));
	DIE(err != cudaSuccess, "cudaMemset init"); // Error checking macro.
}

/**
 * @brief Destructor for `GpuHashTable`.
 * Frees the device memory allocated for the hash table.
 */
GpuHashTable::~GpuHashTable()
{
	cudaFree(this->hash_table); // Free device memory.
}

/**
 * @brief Reshapes (resizes) the hash table on the GPU.
 * This host-side method allocates a new, larger hash table on the device,
 * launches the `kernel_reshape` to transfer entries, and updates the hash table pointer and size.
 * @param numBucketsReshape The desired new size for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape)
{
	entry *new_hash_table; // Pointer for the new hash table on the device.
	cudaError_t err;
	unsigned int numBlocks;
	int old_size = this->size; // Store old size for kernel.

	this->size = numBucketsReshape; // Update the hash table size.

	// Allocate device memory for the new hash table.
	err = cudaMalloc(&new_hash_table, numBucketsReshape * sizeof(entry));
	DIE(err != cudaSuccess, "cudaMalloc reshape"); // Error checking macro.

	// Initialize the new hash table with 0 (KEY_INVALID).
	err = cudaMemset(new_hash_table, 0, numBucketsReshape * sizeof(entry));
	DIE(err != cudaSuccess, "cudaMemset reshape"); // Error checking macro.

	// Calculate number of blocks for kernel launch.
	numBlocks = old_size / THREADS_PER_BLOCK;
	if (old_size % THREADS_PER_BLOCK)
		numBlocks++;

	// Launch the kernel_reshape to copy entries from old to new hash table.
	kernel_reshape<<<numBlocks, THREADS_PER_BLOCK>>>(
		this->hash_table, new_hash_table, old_size, numBucketsReshape);

	cudaDeviceSynchronize(); // Wait for the kernel to complete.

	// Free the old hash table memory.
	err = cudaFree(this->hash_table);
	DIE(err != cudaSuccess, "cudaFree reshape"); // Error checking macro.

	this->hash_table = new_hash_table; // Update the hash table pointer to the new one.
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * This host-side method checks for load factor, reshapes if necessary,
 * transfers data to the device, launches the `kernel_insert` kernel, and frees host memory.
 * @param keys Pointer to an array of keys on the host.
 * @param values Pointer to an array of values on the host.
 * @param numKeys The number of key-value pairs to insert.
 * @return True on success.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys)
{
	int *device_keys; // Pointer for keys on device.
	int *device_values; // Pointer for values on device.
	size_t total_size = numKeys * sizeof(int); // Total size of data to transfer.
	unsigned int numBlocks;
	cudaError_t err;

	// Allocate device memory for keys.
	err = cudaMalloc(&device_keys, total_size);
	// Pre-condition: Check for CUDA memory allocation failure.
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc device_keys");
		return false;
	}
	// Allocate device memory for values.
	err = cudaMalloc(&device_values, total_size);
	// Pre-condition: Check for CUDA memory allocation failure.
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc device_values");
		return false;
	}

	// Pre-condition: If load factor exceeds MAXIMUM_LOAD_FACTOR, reshape the hash table.
	if (float(numberInsertedPairs + numKeys) / this->size >= 
	    MAXIMUM_LOAD_FACTOR)
		// Reshape to a new size based on current entries and minimum load factor.
		reshape(int((numberInsertedPairs + numKeys) /
			    MINIMUM_LOAD_FACTOR));

	// Copy keys from host to device.
	err = cudaMemcpy(device_keys, keys, total_size, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy device_keys"); // Error checking macro.
	// Copy values from host to device.
	err = cudaMemcpy(device_values, values, total_size,
			 cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy device_values"); // Error checking macro.

	// Calculate number of blocks for kernel launch.
	numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK)
		numBlocks++;

	// Launch the kernel_insert.
	kernel_insert<<<numBlocks, THREADS_PER_BLOCK>>>(
		device_keys, device_values, numKeys, this->hash_table,
		this->size);

	cudaDeviceSynchronize(); // Wait for the kernel to complete.

	numberInsertedPairs += numKeys; // Update the count of inserted pairs.

	// Free device memory.
	err = cudaFree(device_keys);
	DIE(err != cudaSuccess, "cudaFree device_keys"); // Error checking macro.

	err = cudaFree(device_values);
	DIE(err != cudaSuccess, "cudaFree device_values"); // Error checking macro.

	return true;
}


/**
 * @brief Retrieves a batch of values for a given set of keys.
 * This host-side method allocates host memory for results, transfers keys to device,
 * launches the `kernel_get` kernel, transfers results back to host, and frees device memory.
 * @param keys Pointer to an array of keys on the host.
 * @param numKeys The number of keys to retrieve.
 * @return A pointer to an array of retrieved values on the host. Caller is responsible for `free`ing it.
 *         Returns `nullptr` on CUDA memory allocation failure.
 */
int *GpuHashTable::getBatch(int *keys, int numKeys)
{
	int *device_keys; // Pointer for keys on device.
	int *values, *device_values; // Pointers for values on host and device.
	size_t total_size = numKeys * sizeof(int); // Total size of data to transfer.
	unsigned int numBlocks;
	cudaError_t err;

	// Allocate device memory for keys.
	err = cudaMalloc(&device_keys, total_size);
	// Pre-condition: Check for CUDA memory allocation failure.
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc device_keys");
		return nullptr;
	}
	// Allocate device memory for values.
	err = cudaMalloc(&device_values, total_size);
	// Pre-condition: Check for CUDA memory allocation failure.
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc values");
		return nullptr;
	}

	// Initialize device values array with 0 (KEY_INVALID).
	err = cudaMemset(device_values, 0, total_size);
	DIE(err != cudaSuccess, "cudaMemset getbatch"); // Error checking macro.

	// Copy keys from host to device.
	err = cudaMemcpy(device_keys, keys, total_size, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy device_keys"); // Error checking macro.

	// Allocate host memory for results.
	values = (int *)malloc(total_size);
	DIE(values == NULL, "malloc values"); // Error checking macro.

	// Calculate number of blocks for kernel launch.
	numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK)
		numBlocks++;

	// Launch the kernel_get.
	kernel_get<<<numBlocks, THREADS_PER_BLOCK>>>(device_keys,
					       device_values, numKeys,
					       this->hash_table,
					       this->size);

	cudaDeviceSynchronize(); // Wait for kernel to complete.

	// Copy results from device to host.
	err = cudaMemcpy(values, device_values, total_size,
			 cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, "cudaMemcpy values"); // Error checking macro.

	// Free device memory.
	err = cudaFree(device_keys);
	DIE(err != cudaSuccess, "cudaFree device_keys"); // Error checking macro.

	err = cudaFree(device_values);
	DIE(err != cudaSuccess, "cudaFree device_values"); // Error checking macro.

	return values; // Return pointer to host results.
}

/**
 * @brief Calculates the current load factor of the hash table.
 * The load factor is the ratio of the number of inserted pairs to the total hash table size.
 * @return The load factor as a float, or 0 if the hash table size is 0.
 */
float GpuHashTable::loadFactor()
{
	// Pre-condition: Avoid division by zero if hash table size is 0.
	if (this->size == 0)
		return 0;
	else
		return (float(numberInsertedPairs) / this->size); // numberInsertedPairs / size.
}

// Macros for convenient usage of GpuHashTable functions.
#define HASH_INIT GpuHashTable GpuHashTable(1); // Initialize a GpuHashTable instance.
#define HASH_RESERVE(size) GpuHashTable.reshape(size); // Reshape the hash table.

#define HASH_BATCH_INSERT(keys, values, numKeys)                               
	GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor() // Get current load factor.

// This inclusion suggests that `test_map.cpp` might contain a host-side test harness
// that uses these macros to interact with the GpuHashTable.
#include "test_map.cpp"

// Conditional compilation: If _HASHCPU_ is not defined, these definitions are included.
// This block appears to contain global definitions and a class declaration,
// typically found in a header file (e.g., gpu_hashtable.hpp).
#ifndef _HASHCPU_
#define _HASHCPU_

#include <cstdio> // For fprintf, perror
#include <cstdlib> // For exit
#include <errno.h> // For errno

using namespace std;

#define	KEY_INVALID			0 // Sentinel value for an invalid/empty key.
#define THREADS_PER_BLOCK 		1024 // Standard number of threads per block for CUDA kernels.
#define MINIMUM_LOAD_FACTOR		0.8 // Target minimum load factor after reshaping.
#define MAXIMUM_LOAD_FACTOR		0.9 // Maximum load factor before triggering a reshape.

// Macro for error checking and exiting.
#define DIE(assertion, call_description) \
	do { \
		if (assertion) { \
		fprintf(stderr, "(%s, %d): ", \
		__FILE__, __LINE__); \
		perror(call_description); \
		exit(errno); \
	} \
} while (0)
	
// Predefined list of large prime numbers, likely used for hash functions or table sizing.
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
 * @brief First hash function (likely part of a family of hash functions for probing).
 * @param data The integer key.
 * @param limit The upper bound for the hash value.
 * @return The computed hash index.
 */
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
/**
 * @brief Second hash function.
 * @param data The integer key.
 * @param limit The upper bound for the hash value.
 * @return The computed hash index.
 */
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
/**
 * @brief Third hash function.
 * @param data The integer key.
 * @param limit The upper bound for the hash value.
 * @return The computed hash index.
 */
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

/**
 * @brief Structure representing a single entry in the hash table.
 * @param key The key of the entry. `KEY_INVALID` indicates an empty slot.
 * @param value The value associated with the key.
 */
typedef struct entry {
	int key;
	int value;
} entry;

/**
 * @brief Definition of the `GpuHashTable` class and its members.
 * This class provides the host-side interface for managing a hash table
 * that resides and operates primarily on the GPU.
 */
class GpuHashTable
{
	public:
		// Public members representing the state of the hash table on the GPU.
		int numberInsertedPairs; // Number of currently inserted key-value pairs.
		int size; // Total number of buckets (capacity) of the hash table.
		entry *hash_table; // Pointer to the hash table array on the device.

		GpuHashTable(int size); // Constructor: initializes the hash table.
		void reshape(int sizeReshape); // Resizes the hash table on the GPU.
		
		bool insertBatch(int *keys, int* values, int numKeys); // Inserts a batch of key-value pairs into the hash table.
		int* getBatch(int* key, int numItems); // Retrieves a batch of values for keys from the hash table.
		
		float loadFactor(); // Calculates the current load factor of the hash table.
		void occupancy(); // (Declared but not defined in the provided code snippet).
		void print(string info); // (Declared but not defined in the provided code snippet).
	
		~GpuHashTable(); // Destructor: frees GPU resources.
};

#endif