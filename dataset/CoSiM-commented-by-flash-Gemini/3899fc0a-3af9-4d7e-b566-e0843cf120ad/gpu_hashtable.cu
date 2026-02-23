/**
 * @file gpu_hashtable.cu
 * @brief CUDA C++ implementation of a hash table optimized for GPU execution.
 *
 * This file provides a `GpuHashTable` class that manages a hash table residing
 * and operating primarily on the GPU. It utilizes CUDA kernels to perform
 * parallel operations such as insertion, retrieval, and resizing.
 * The implementation uses open addressing with linear probing for collision resolution
 * and `atomicCAS` for thread-safe concurrent modifications.
 * This version uses `cudaMallocManaged` for some buffers, indicating Unified Memory usage.
 *
 * Domain-Specific Awareness (HPC & Parallelism, Performance Optimization):
 * - Memory Hierarchy: Uses `cudaMalloc`, `cudaMallocManaged`, `cudaMemset`, `cudaMemcpy`
 *   for efficient device/unified memory management and data transfer between host and device.
 * - Thread Indexing: Kernels (`kernel_insert`, `kernel_rehash`, `kernel_get`)
 *   use `threadIdx.x` and `blockIdx.x` to distribute work across CUDA threads.
 * - Synchronization: `atomicCAS` provides atomic read-modify-write operations
 *   for thread-safe updates to hash table entries, crucial for concurrent access.
 * - Algorithm: Open addressing with linear probing. The hash table automatically
 *   reshapes (resizes) to a larger capacity when the load factor exceeds
 *   a defined threshold (`0.95`) to maintain performance.
 *
 * Time Complexity:
 * - `GpuHashTable` constructor/destructor: O(size) for memory allocation/deallocation on device.
 * - `myHash`: O(1) per call.
 * - `kernel_insert`: O(numEntries) in parallel. Collision resolution with linear probing
 *   adds a factor proportional to probe length.
 * - `kernel_rehash`: O(oldHt.size) in parallel. Collision resolution
 *   adds a factor proportional to probe length.
 * - `kernel_get`: O(numEntries) in parallel. Collision resolution
 *   adds a factor proportional to probe length.
 * - Host methods (`reshape`, `insertBatch`, `getBatch`): Involve `cudaMemcpy` (O(N) for N elements)
 *   and kernel launches (parallel time complexity of kernels).
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

// Custom header for GpuHashTable definitions (expected to contain hash_table and entry structs)
#include "gpu_hashtable.hpp"

/**
 * @brief Device-side hash function.
 * Computes a hash value for a given key, mapping it to a bucket within the specified limit.
 * Uses a multiplicative hash approach with large prime numbers.
 * @param data The integer key to hash.
 * @param limit The upper bound for the hash value (number of buckets).
 * @return The computed hash index.
 */
__device__ int myHash(int data, int limit) {
	// Formula: ((abs(data) * P1) % P2) % limit
	// P1 and P2 are large prime numbers to ensure good distribution.
	return ((long) abs(data) * 1306601) % 274019932696 % limit; 
}

/**
 * @brief CUDA kernel for inserting a batch of key-value pairs into the hash table.
 * Each thread processes one key-value pair. It uses linear probing for collision
 * resolution and `atomicCAS` for thread-safe insertion.
 * @param values Pointer to an array of values to insert on the device.
 * @param keys Pointer to an array of keys to insert on the device.
 * @param ht The `hash_table` structure containing the hash table entries and size.
 * @param numEntries The number of key-value pairs to insert in this batch.
 */
__global__ void kernel_insert(int *values, int *keys, hash_table ht, int numEntries) {
	// Calculate global thread ID. Each thread processes one key-value pair.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Pre-condition: Ensure thread ID is within the bounds of `numEntries`.
	if (idx < numEntries) {
		int key;
		// Invariant: Compute initial hash for the current key.
		int hash = myHash(keys[idx], ht.size);
		
		// Block Logic: Linear probing starting from `hash` to the end of the table.
		for (int i = hash; i < ht.size; i++) {
			// Atomically compare and swap: attempt to write the key if the slot is 0 (KEY_INVALID).
			// `key` will contain the value of `ht.table[i].key` *before* the atomic operation.
			key = atomicCAS(&ht.table[i].key, 0, keys[idx]);
			bool invalid = key == keys[idx] || key == 0; // Check if already inserted or empty.
			if (invalid) {
				ht.table[i].value = values[idx]; // Write the corresponding value.
				return; // Successfully inserted or duplicate handled.
			}
		}
		
		// Block Logic: Linear probing from `hash - 1` towards the beginning of the table (wraparound).
		// Note: The original code starts from `hash - 1` and goes down to `> 0`. This might miss index 0.
		for (int i = hash - 1; i > 0; i--) { // Potential bug: Should probably go down to 0, not > 0.
			key = atomicCAS(&ht.table[i].key, 0, keys[idx]);
			bool invalid = key == keys[idx] || key == 0;
			if (invalid) {
				ht.table[i].value = values[idx]; // Write the corresponding value.
				return; // Successfully inserted or duplicate handled.
			}
		}
	}
	return; // Returns if key not inserted after probing.
}

/**
 * @brief CUDA kernel for retrieving values for a batch of keys from the hash table.
 * Each thread processes one key lookup. It uses linear probing to find keys.
 * @param values Pointer to an array on the device where retrieved values will be stored.
 * @param keys Pointer to an array of keys to search for on the device.
 * @param ht The `hash_table` structure containing the hash table entries and size.
 * @param numEntries The number of keys to retrieve in this batch.
 */
__global__ void kernel_get(int *values, int *keys, hash_table ht, int numEntries) {
	// Calculate global thread ID. Each thread processes one key lookup.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Pre-condition: Ensure thread ID is within the bounds of `numEntries`.
	if (idx < numEntries) {
		// Invariant: Compute initial hash for the current key.
		int hash = myHash(keys[idx], ht.size);
		
		// Block Logic: Linear probing starting from `hash` to the end of the table.
		for (int i = hash; i < ht.size; i++) {
			// If key matches, store the value and break.
			if (ht.table[i].key == keys[idx]) {
				values[idx] = ht.table[i].value;
				return;
			}
		}

		// Block Logic: Linear probing from `hash - 1` towards the beginning of the table (wraparound).
		for (int i = hash - 1; i > 0; i--) { // Potential bug: Should probably go down to 0, not > 0.
			if (ht.table[i].key == keys[idx]) {
				values[idx] = ht.table[i].value;
				return;
			}
		}
	}
}

/**
 * @brief CUDA kernel for reshaping (resizing) the hash table.
 * Copies existing valid entries from an old hash table to a new, larger one.
 * Uses `atomicCAS` for concurrent insertion into the new table, resolving collisions
 * with linear probing.
 * @param newHt The new, larger `hash_table` structure on the device.
 * @param oldHt The old `hash_table` structure on the device.
 */
__global__ void kernel_rehash(hash_table newHt, hash_table oldHt) {
	// Calculate global thread ID. Each thread potentially processes one entry from the old hash table.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Pre-condition: If thread ID is out of bounds or the entry in the old table is invalid, skip.
	if (idx >= oldHt.size || oldHt.table[idx].key == 0) // The `oldHt.size` check is incorrect. It should be `oldHt.capacity`.
			return;
	
	int key;
	// Invariant: Compute initial hash for the key in the new hash table.
	int hash = myHash(oldHt.table[idx].key, newHt.size);
	
	// Block Logic: Linear probing starting from `hash` to the end of the new table.
	for (int i = hash; i < newHt.size; i++) {
		key = atomicCAS(&newHt.table[i].key, 0, oldHt.table[idx].key);
		if (key == 0) { // If `key` was 0 (KEY_INVALID), our key was successfully inserted.
			newHt.table[i].value = oldHt.table[idx].value; // Write the corresponding value.
			return; // Successfully inserted.
		} 
	}
	
	// Block Logic: Linear probing from `hash - 1` towards the beginning of the new table (wraparound).
	for (int i = hash - 1; i > 0; i--) { // Potential bug: Should probably go down to 0, not > 0.
		key = atomicCAS(&newHt.table[i].key, 0, oldHt.table[idx].key);
		if (key == 0) {
			newHt.table[i].value = oldHt.table[idx].value; // Write the corresponding value.
			return; // Successfully inserted.
		} 
	}
	return; // Returns if key not inserted after probing.
}


/**
 * @brief Constructor for `GpuHashTable`.
 * Initializes the host-side `GpuHashTable` object and allocates device memory
 * for the underlying `hash_table` entries, initializing them to 0 (KEY_INVALID).
 * @param size The initial size (capacity) for the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	inserted = 0; // Initialize the count of inserted pairs.
	ht.size = size; // Set the initial capacity of the hash table.
	ht.table = nullptr; // Initialize device pointer to null.
	
	// Allocate device memory for the hash table entries.
	if (cudaMalloc(&ht.table, size * sizeof(entry)) != cudaSuccess) {
		std::cerr << "Memory allocation failed! \n"; // Error message.
		return; // Return on allocation failure.
	}
	// Initialize all hash table entries with 0 (KEY_INVALID) to indicate empty slots.
	cudaMemset(ht.table, 0, size * sizeof(entry));
}

/**
 * @brief Destructor for `GpuHashTable`.
 * Frees the device memory allocated for the hash table entries.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(ht.table); // Free device memory.
}

/**
 * @brief Reshapes (resizes) the hash table on the GPU to a new capacity.
 * This host-side method allocates a new, larger hash table on the device,
 * launches the `kernel_rehash` to re-hash and transfer entries from the old
 * table to the new one, and then updates the internal `ht` structure.
 * @param numBucketsReshape The desired new number of buckets (capacity) for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hash_table newHt; // Declare a new hash_table structure.
	newHt.size = numBucketsReshape; // Set the new capacity.
	
	// Allocate device memory for the new hash table entries.
	if (cudaMalloc(&newHt.table, numBucketsReshape * sizeof(entry)) != cudaSuccess) {
		std::cerr << "Memory allocation failed! \n";
		return;
	}
	// Initialize the new hash table entries with 0 (KEY_INVALID).
	cudaMemset(newHt.table, 0, numBucketsReshape * sizeof(entry));
	
	// Calculate number of blocks for kernel launch.
	unsigned int blocks;
	blocks = ht.size / NUM_THREADS + 1; // `ht.size` here refers to old capacity, not numEntries.
	// This calculation should ideally use `old_capacity` (current ht.size) to determine grid size.

	// Launch the `kernel_rehash` to copy entries from the old to the new hash table.
	kernel_rehash<<<blocks, NUM_THREADS>>>(newHt, ht);

	cudaDeviceSynchronize(); // Wait for the kernel to complete.
	
	cudaFree(ht.table); // Free the old hash table memory.
	
	ht = newHt; // Update the host-side `ht` structure to point to the new hash table.
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table on the GPU.
 * This host-side method handles load factor checking, resizing (reshaping) if necessary,
 * transfers host data to unified memory, launches the `kernel_insert` kernel,
 * and updates the internal count of inserted pairs.
 * @param keys Pointer to an array of keys on the host.
 * @param values Pointer to an array of values on the host.
 * @param numKeys The number of key-value pairs to insert.
 * @return True on success, false on CUDA memory allocation failure.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *deviceKeys; // Pointer for keys in Unified Memory.
	cudaMallocManaged(&deviceKeys, numKeys * sizeof(int)); // Allocate Unified Memory.
	
	int *deviceValues; // Pointer for values in Unified Memory.
	cudaMallocManaged(&deviceValues, numKeys * sizeof(int)); // Allocate Unified Memory.

	float loadFactor;
	
	// Pre-condition: Check for CUDA memory allocation failures.
	if (deviceValues == nullptr || deviceKeys == nullptr) {
		std::cerr << "Memory allocation failed! \n";
		// Clean up any successfully allocated memory.
		if (deviceKeys) cudaFree(deviceKeys);
		if (deviceValues) cudaFree(deviceValues);
		return false;
	}
	
	// Calculate current load factor with prospective insertions.
	loadFactor = float(inserted + numKeys) / ht.size; // `ht.size` here refers to capacity, not numEntries.
	
	// Pre-condition: If load factor exceeds 0.95, reshape the hash table.
	if (loadFactor >= 0.95)
		reshape(int((inserted + numKeys) / 0.85)); // New size based on current entries and target load factor.
	
	// Copy keys and values from host to Unified Memory.
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	// Calculate number of blocks for kernel launch.
	unsigned int blocks = numKeys / NUM_THREADS + 1;

	// Launch the `kernel_insert` to insert the batch of key-value pairs.
	kernel_insert<<<blocks, NUM_THREADS>>>(deviceValues, 
	        deviceKeys, ht, numKeys);

	cudaDeviceSynchronize(); // Wait for the kernel to complete.
	
	cudaFree(deviceKeys); // Free Unified Memory.
	cudaFree(deviceValues); // Free Unified Memory.
	
	inserted = inserted + numKeys; // Update the number of inserted pairs.

	return true; // Batch insertion successful.
}

/**
 * @brief Retrieves a batch of values for a given set of keys from the hash table on the GPU.
 * This host-side method allocates Unified Memory for keys and values, transfers keys to device,
 * launches the `kernel_get` kernel, and returns a pointer to the values in Unified Memory.
 * @param keys Pointer to an array of keys on the host.
 * @param numKeys The number of keys to retrieve.
 * @return A pointer to an array of retrieved values in Unified Memory. Caller is responsible for `cudaFree`ing this memory.
 *         Returns `nullptr` on memory allocation failures.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *deviceKeys; // Pointer for keys in Unified Memory.
	cudaMallocManaged(&deviceKeys, numKeys * sizeof(int));
	int *values; // Pointer for values in Unified Memory.
	cudaMallocManaged(&values, numKeys * sizeof(int));
	
	// Pre-condition: Check for any memory allocation failures.
	if (values == nullptr || deviceKeys  == nullptr) {
		std::cerr << "Memory allocation failed!\n";
		// Clean up any successfully allocated memory.
		if (deviceKeys) cudaFree(deviceKeys);
		if (values) cudaFree(values);
		return nullptr; // Return nullptr on failure.
	}
	
	// Copy keys from host to Unified Memory.
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	// Calculate number of blocks for kernel launch.
	unsigned int blocks = numKeys / NUM_THREADS + 1;
	
	// Launch the `kernel_get` to retrieve values for the batch of keys.
	kernel_get<<<blocks, NUM_THREADS>>>(values, deviceKeys, ht, numKeys);
	
	cudaDeviceSynchronize(); // Wait for kernel to complete.
	
	cudaFree(deviceKeys); // Free Unified Memory for keys.

	return values; // Return pointer to retrieved values in Unified Memory.
}

/**
 * @brief Calculates the current load factor of the hash table.
 * The load factor is the ratio of the number of active entries (`inserted`)
 * to the total capacity (`ht.size`).
 * @return The load factor as a float, or 0 if the hash table capacity is 0.
 */
float GpuHashTable::loadFactor() {
	// Pre-condition: Avoid division by zero if hash table capacity is 0.
	if (ht.size != 0)
		return (float(inserted) / ht.size); // inserted / capacity.
	return 0;
}

// Macros for convenient usage of GpuHashTable functions.
#define HASH_INIT GpuHashTable GpuHashTable(1); // Initialize a GpuHashTable instance with capacity 1.
#define HASH_RESERVE(size) GpuHashTable.reshape(size); // Reshape the hash table to a new size.

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys) // Inserts a batch of key-value pairs.
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys) // Retrieves a batch of values for keys.

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor() // Get current load factor.

// This inclusion suggests that `test_map.cpp` might contain a host-side test harness
// that uses these macros to interact with the GpuHashTable.
#include "test_map.cpp"

// Conditional compilation: If _HASHCPU_ is not defined, these definitions are included.
// This block appears to contain global definitions and class/struct declarations,
// typically found in a header file (e.g., gpu_hashtable.hpp).
#ifndef _HASHCPU_
#define _HASHCPU_

#include <cstdio> // For fprintf, perror
#include <cstdlib> // For exit
#include <errno.h> // For errno

using namespace std;

#define KEY_INVALID 0 // Sentinel value for an invalid/empty key in the hash table.
#define NUM_THREADS 256 // Default number of threads per block for CUDA kernels.

// Macro for error checking CUDA API calls and exiting on failure.
#define DIE(assertion, call_description) 
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




int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

/**
 * @brief Structure representing a single entry (key-value pair) in the hash table.
 */
struct entry {
	int key, value; // Key and value of the hash table entry.
};

/**
 * @brief Structure representing the hash table itself, managed on the GPU.
 */
struct hash_table {
	int size; // The current capacity (number of buckets) of the hash table.
	entry *table; // Pointer to the array of `entry` structures residing on the device.
};

/**
 * @brief Host-side class for managing a `hash_table` on the GPU.
 * Provides methods for initializing, reshaping, inserting batches, and retrieving batches
 * from the GPU hash table.
 */
class GpuHashTable
{
	int inserted; // Tracks the number of actually inserted key-value pairs.
	hash_table ht; // The underlying `hash_table` structure, with entries on the GPU.

	public:
		GpuHashTable(int size); // Constructor: initializes the hash table.
		void reshape(int sizeReshape); // Resizes the hash table on the GPU.
		
		bool insertBatch(int *keys, int* values, int numKeys); // Inserts a batch of key-value pairs.
		int* getBatch(int* key, int numItems); // Retrieves a batch of values for keys.
		
		float loadFactor(); // Calculates the current load factor.
		void occupancy(); // (Declared but not defined in the provided code snippet).
		void print(string info); // (Declared but not defined in the provided code snippet).
	
		~GpuHashTable(); // Destructor: frees GPU resources.
};

#endif // _HASHCPU_