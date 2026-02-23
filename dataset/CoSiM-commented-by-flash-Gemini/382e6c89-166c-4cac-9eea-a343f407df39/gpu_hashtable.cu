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
 * - Thread Indexing: Kernels (`resize`, `insert`, `get`) use `blockIdx.x` and `threadIdx.x`
 *   to parallelize operations across the hash table.
 * - Synchronization: `atomicCAS` ensures atomic updates to hash table entries,
 *   critical for concurrent operations in a lock-free manner on the GPU.
 * - Algorithm: Open addressing with linear probing. When the load factor exceeds a threshold,
 *   the hash table is reshaped (resized) to maintain performance.
 *
 * Time Complexity:
 * - `GpuHashTable` constructor/destructor: O(size) for memory allocation/deallocation.
 * - `hash_func`: O(1) per call.
 * - `resize` kernel: O(hash_size) in parallel for `hash_size` threads. Collision resolution
 *   adds a factor proportional to probe length.
 * - `insert` kernel: O(numKeys) in parallel for `numKeys` threads. Collision resolution
 *   adds a factor proportional to probe length.
 * - `get` kernel: O(numKeys) in parallel for `numKeys` threads. Collision resolution
 *   adds a factor proportional to probe length.
 * - Host methods (`reshape`, `insertBatch`, `getBatch`): Involve `cudaMemcpy` (O(N) for N elements)
 *   and kernel launches (parallel time complexity of kernels).
 *
 * Space Complexity: O(hash_size) for the hash table on the device.
 */

// Standard C++ headers
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <map> // For comparison/testing in CPU version

// CUDA specific headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Custom header for GpuHashTable definitions (expected to contain entry struct and class declaration)
#include "gpu_hashtable.hpp"


/**
 * @brief Constructor for `GpuHashTable`.
 * Allocates device memory for the hash table and initializes it with `KEY_INVALID`.
 * @param size The initial size (number of buckets) for the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	hash_size = size; // Initialize the total number of buckets.
	num_entries = 0;  // Initialize the count of active entries.
	// Allocate device memory for the hash table entries.
	cudaMalloc((void **) &hashtable, size * sizeof(entry));
	// Initialize all hash table entries with KEY_INVALID to indicate empty slots.
	cudaMemset(hashtable, KEY_INVALID, size * sizeof(entry));
}

/**
 * @brief Destructor for `GpuHashTable`.
 * Frees the device memory allocated for the hash table.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashtable); // Free device memory.
}

/**
 * @brief Device-side hash function.
 * Computes a hash value for a given data key, mapping it to a bucket within a limit.
 * Uses a multiplicative hash approach with large prime numbers.
 * @param data The integer key to hash.
 * @param limit The upper bound for the hash value (number of buckets).
 * @return The computed hash index.
 */
__device__ uint32_t hash_func(int data, int limit) {
	// Formula: ((abs(data) * P1) % P2) % limit
	// P1 and P2 are large prime numbers to ensure good distribution.
	return ((long)abs(data) * 105359939) % 1685759167 % limit;
}

/**
 * @brief CUDA kernel for resizing the hash table.
 * Copies existing entries from an old hash table to a new, larger one.
 * Uses `atomicCAS` for concurrent insertion into the new table, resolving collisions
 * with linear probing.
 * @param hashtable Pointer to the old hash table on the device.
 * @param new_hash Pointer to the new, larger hash table on the device.
 * @param hash_size The size of the old hash table.
 * @param numBucketsReshape The size of the new hash table.
 */
__global__ void resize(GpuHashTable::entry *hashtable, GpuHashTable::entry *new_hash,
						int hash_size, int numBucketsReshape) {
	// Calculate global thread ID. Each thread processes one entry from the old hash table.
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// Pre-condition: Ensure thread ID is within the bounds of the old hash table size.
	if (tid < hash_size) {
		// If the current entry in the old hash table is invalid, skip it.
		if (hashtable[tid].key == KEY_INVALID)
			return;
		
		// Invariant: Compute initial hash for the key in the new hash table.
		uint32_t key = hash_func(hashtable[tid].key, numBucketsReshape);
		// Block Logic: Linear probing for collision resolution.
		while (true) {
			// Atomically compare and swap: attempt to write the key if the slot is KEY_INVALID.
			// `prev` will contain the value of `new_hash[key].key` *before* the atomic operation.
			uint32_t prev = atomicCAS(&new_hash[key].key, KEY_INVALID, hashtable[tid].key);
			// If `prev` was KEY_INVALID, our key was successfully inserted.
			// If `prev` was equal to `hashtable[tid].key`, it means another thread already inserted this key (duplicate).
			if (prev == hashtable[tid].key || prev == KEY_INVALID) {
				new_hash[key].value = hashtable[tid].value; // Write the corresponding value.
				break; // Successfully inserted or duplicate handled.
			}
			// Collision: Move to the next bucket (linear probing).
			key++;
			key %= numBucketsReshape; // Wrap around if end of table reached.
		}
	}
}

/**
 * @brief Reshapes (resizes) the hash table on the GPU.
 * This host-side method allocates a new, larger hash table on the device,
 * launches the `resize` kernel to transfer entries, and updates the hash table pointer and size.
 * @param numBucketsReshape The desired new size for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	uint32_t block_size = 100; // CUDA block size for kernel launch.
	uint32_t blocks_no = hash_size / block_size; // Number of blocks needed.
	if (hash_size % block_size)
		++blocks_no; // Add an extra block if there's a remainder.
	struct entry *new_hash; // Pointer for the new hash table on the device.
	
	// Allocate device memory for the new hash table.
	cudaMalloc((void **) &new_hash, numBucketsReshape * sizeof(entry));
	// Initialize the new hash table with KEY_INVALID.
	cudaMemset(new_hash, KEY_INVALID, numBucketsReshape * sizeof(entry));
	// Launch the resize kernel to copy entries from old to new hash table.
	resize<<<blocks_no, block_size>>>(hashtable, new_hash, hash_size, numBucketsReshape);
	cudaDeviceSynchronize(); // Wait for the kernel to complete.
	cudaFree(hashtable); // Free the old hash table memory.
	hashtable = new_hash; // Update the hash table pointer to the new one.
	hash_size = numBucketsReshape; // Update the hash table size.
}

/**
 * @brief CUDA kernel for inserting a batch of key-value pairs into the hash table.
 * Each thread attempts to insert one key-value pair. Uses `atomicCAS` for
 * thread-safe insertion and linear probing for collision resolution.
 * @param hashtable Pointer to the hash table on the device.
 * @param hash_size The current size of the hash table.
 * @param keys Pointer to an array of keys to insert on the device.
 * @param values Pointer to an array of values to insert on the device.
 * @param numKeys The number of key-value pairs to insert.
 */
__global__ void insert(GpuHashTable::entry *hashtable, int hash_size,
						int *keys, int* values, int numKeys) {
	// Calculate global thread ID. Each thread processes one key-value pair.
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// Pre-condition: Ensure thread ID is within the bounds of `numKeys`.
	if (tid < numKeys) {
		// Invariant: Compute initial hash for the current key.
		uint32_t key = hash_func(keys[tid], hash_size);
		// Block Logic: Linear probing for collision resolution.
		while (true) {
			// Atomically compare and swap: attempt to write the key if the slot is KEY_INVALID.
			uint32_t prev = atomicCAS(&hashtable[key].key, KEY_INVALID, keys[tid]);
			// If `prev` was KEY_INVALID, our key was successfully inserted.
			// If `prev` was equal to `keys[tid]`, it means another thread already inserted this key (duplicate).
			if (prev == keys[tid] || prev == KEY_INVALID) {
				hashtable[key].value = values[tid]; // Write the corresponding value.
				return; // Successfully inserted or duplicate handled.
			}
			// Collision: Move to the next bucket (linear probing).
			key++;
			key %= hash_size; // Wrap around if end of table reached.
		}
	}
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * This host-side method checks for load factor, reshapes if necessary,
 * transfers data to the device, launches the `insert` kernel, and frees host memory.
 * @param keys Pointer to an array of keys on the host.
 * @param values Pointer to an array of values on the host.
 * @param numKeys The number of key-value pairs to insert.
 * @return True on success.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *new_values; // Temporary storage for values.
	
	// Retrieve existing values for the given keys to check for new entries.
	new_values = getBatch(keys, numKeys);
	// Invariant: Count new entries to update `num_entries`.
	for (int i = 0; i < numKeys; i++)
		if (new_values[i] == KEY_INVALID)
			num_entries++;
	// Pre-condition: If load factor exceeds 0.9, reshape the hash table.
	if ((float)(num_entries) / hash_size >= 0.9)
		reshape(num_entries + (int)(0.1 * num_entries)); // New size = current entries + 10% buffer.

	uint32_t block_size = 100; // CUDA block size.
	uint32_t blocks_no = numKeys / block_size; // Number of blocks.
	if (numKeys % block_size)
		++blocks_no; // Add extra block for remainder.
	int *dev_keys = 0; // Pointer for keys on device.
	int *dev_values = 0; // Pointer for values on device.
	
	// Allocate device memory for keys and values.
	cudaMalloc((void **) &dev_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &dev_values, numKeys * sizeof(int));
	// Copy keys and values from host to device.
	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	// Launch the insert kernel.
	insert<<<blocks_no, block_size>>>(hashtable, hash_size, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize(); // Wait for the kernel to complete.
	// Free device memory.
	cudaFree(dev_keys);
	cudaFree(dev_values);
	free(new_values); // Free temporary host memory.
	return true;
}

/**
 * @brief CUDA kernel for retrieving values for a batch of keys from the hash table.
 * Each thread attempts to retrieve one value. Uses linear probing to find keys.
 * @param hashtable Pointer to the hash table on the device.
 * @param hash_size The current size of the hash table.
 * @param keys Pointer to an array of keys to search for on the device.
 * @param values Pointer to an array on the device where retrieved values will be stored.
 * @param numKeys The number of keys to retrieve.
 */
__global__ void get(GpuHashTable::entry *hashtable, int hash_size,
						int *keys, int *values, int numKeys) {
	// Calculate global thread ID. Each thread processes one key lookup.
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// Pre-condition: Ensure thread ID is within the bounds of `numKeys`.
	if (tid < numKeys) {
		// Invariant: Compute initial hash for the current key.
		uint32_t key = hash_func(keys[tid], hash_size);
		// Block Logic: Linear probing to find the key.
		while (true) {
			// If key matches, store the value and break.
			if (hashtable[key].key == keys[tid]) {
				values[tid] = hashtable[key].value;
				break;
			}
			// If an empty slot (KEY_INVALID) is found, the key is not in the table.
			if (hashtable[key].key == KEY_INVALID) {
				values[tid] = KEY_INVALID; // Indicate key not found.
				break;
			}
			// Move to the next bucket (linear probing).
			key++;
			key %= hash_size; // Wrap around.
		}
	}
}

/**
 * @brief Retrieves a batch of values for a given set of keys.
 * This host-side method allocates host memory for results, transfers keys to device,
 * launches the `get` kernel, transfers results back to host, and frees device memory.
 * @param keys Pointer to an array of keys on the host.
 * @param numKeys The number of keys to retrieve.
 * @return A pointer to an array of retrieved values on the host. Caller is responsible for `free`ing it.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *results = (int *)malloc(numKeys * sizeof(int)); // Allocate host memory for results.
	uint32_t block_size = 100; // CUDA block size.
	uint32_t blocks_no = numKeys / block_size; // Number of blocks.
	if (numKeys % block_size)
		++blocks_no; // Add extra block for remainder.
	int *dev_keys = 0; // Pointer for keys on device.
	int *dev_values = 0; // Pointer for values on device.
	
	// Allocate device memory for keys and values.
	cudaMalloc((void **) &dev_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &dev_values, numKeys * sizeof(int));
	// Copy keys from host to device.
	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	// Initialize device values array with KEY_INVALID.
	cudaMemset(dev_values, KEY_INVALID, numKeys * sizeof(int));
	// Launch the get kernel.
	get<<<blocks_no, block_size>>>(hashtable, hash_size, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize(); // Wait for kernel to complete.
	
	// Copy results from device to host.
	cudaMemcpy(results, dev_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	// Free device memory.
	cudaFree(dev_keys);
	cudaFree(dev_values);
	return results; // Return pointer to host results.
}

/**
 * @brief Calculates the current load factor of the hash table.
 * The load factor is the ratio of the number of entries to the total hash table size.
 * @return The load factor as a float.
 */
float GpuHashTable::loadFactor() {
	return (float)num_entries / hash_size; // num_entries / hash_size.
}

// Macros for convenient usage of GpuHashTable functions.
#define HASH_INIT GpuHashTable GpuHashTable(1); // Initialize a GpuHashTable instance.
#define HASH_RESERVE(size) GpuHashTable.reshape(size); // Reshape the hash table.

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys) // Insert a batch.
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys) // Retrieve a batch.

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor() // Get current load factor.

#define HASH_DESTROY GpuHashTable.~GpuHashTable(); // Call destructor to free resources.

// This inclusion suggests that `test_map.cpp` might contain a host-side test harness
// that uses these macros to interact with the GpuHashTable.
#include "test_map.cpp"

// Conditional compilation: If _HASHCPU_ is not defined, these definitions are included.
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0 // Sentinel value for an invalid/empty key.

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
 * @brief Definition of the `GpuHashTable` class and its members.
 * This is a re-declaration (or forward declaration if `gpu_hashtable.hpp` is not correctly structured)
 * of the class, defining its public interface.
 */
class GpuHashTable
{
	public:
		/**
		 * @brief Structure representing a single entry in the hash table.
		 * @param key The key of the entry. `KEY_INVALID` indicates an empty slot.
		 * @param value The value associated with the key.
		 */
		struct entry {
			uint32_t key;
			uint32_t value;
		};
		int hash_size; // Total number of buckets in the hash table.
		int num_entries; // Current number of valid entries in the hash table.
		struct entry *hashtable; // Pointer to the hash table array on the device.

		GpuHashTable(int size); // Constructor: initializes the hash table.
		void reshape(int sizeReshape); // Resizes the hash table.
		
		bool insertBatch(int *keys, int* values, int numKeys); // Inserts a batch of key-value pairs.
		int* getBatch(int* key, int numItems); // Retrieves a batch of values for keys.
		
		float loadFactor(); // Calculates the current load factor.
		void occupancy(); // (Declared but not defined in the provided code snippet).
		void print(string info); // (Declared but not defined in the provided code snippet).
	
		~GpuHashTable(); // Destructor: frees resources.
};

#endif // _HASHCPU_