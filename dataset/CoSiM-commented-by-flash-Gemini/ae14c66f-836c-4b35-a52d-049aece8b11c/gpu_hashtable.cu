/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA, designed for high-throughput key-value storage and retrieval.
 *
 * This file provides the core CUDA kernels and host-side functions for managing a hash table
 * on NVIDIA GPUs. It utilizes a linear probing strategy for collision resolution and
 * atomic operations to ensure correctness in parallel insertion scenarios.
 *
 * Domain: High-Performance Computing (HPC), GPU Programming, Data Structures, Concurrent Algorithms.
 */
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <cstdlib>
#include <stdio.h>
#include <errno.h>

#include "gpu_hashtable.hpp"

// @brief The total number of buckets available in the hash map.
int mapSize;
// @brief Pointer to the number of elements currently stored in the hash map, managed on the device.
int *currentNo;
// @brief Device-side pointer to a list of prime numbers used for hash function computations.
size_t *copyList = NULL;
// @brief Device-side pointer to the actual hash map structure, storing key-value pairs.
struct realMap *hashMap = NULL;


/**
 * @brief CUDA kernel for retrieving values associated with given keys in parallel.
 *
 * This kernel iterates through a batch of keys, computes their hash indices, and
 * retrieves corresponding values from the GPU hash map. It handles collisions
 * using linear probing.
 *
 * @param keys Device-side array of keys to search for.
 * @param numKeys The number of keys to search.
 * @param myMap The GPU hash map structure.
 * @param values Device-side array to store the retrieved values.
 * @param num The total number of buckets in `myMap`.
 * @param list Device-side array of prime numbers for hashing.
 */
__global__ void kernel_get(int* keys, int numKeys, struct realMap *myMap,
	int* values, int num, size_t *list) {

	// Functional Utility: Computes a unique global thread index.
	// Inline: `threadIdx.x` provides the thread's ID within its block, `blockDim.x` is the number of threads per block,
	// and `blockIdx.x` is the block's ID within the grid. This calculation yields a unique global index for each thread.
	// Invariant: `i` correctly maps to a unique key in the `keys` array for processing by this thread.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	// Block Logic: Ensures that each thread processes a valid key within the bounds of the input array.
	// Pre-condition: `i` is the global thread index calculated for this thread.
	// Invariant: Only threads with `i < numKeys` are allowed to proceed, preventing out-of-bounds access.
	if(i >= numKeys)
		return;


	// Functional Utility: Calculates the initial hash index for the current key.
	int index = hash1(keys[i], num, list);

	/**
	 * Block Logic: Implements linear probing to find the key in the hash table.
	 * This loop continues until the target key is found or it's confirmed that the key is not present.
	 * Pre-condition: `index` holds the initial hashed position for `keys[i]`.
	 * Invariant: The loop either finds the key and assigns its value or continues probing
	 *            until all possible locations have been checked (implied wrap-around).
	 */
	while(1) {
		
		// Block Logic: Checks if the current hash map entry's key matches the target key.
		// Pre-condition: `myMap[index]` contains a potential key-value pair.
		// Invariant: If `myMap[index].key` matches `keys[i]`, the associated value is retrieved, and the search terminates.
		if(myMap[index].key == keys[i]) {
			values[i] = myMap[index].value; // Inline: Assigns the found value to the output array.
			break; // Inline: Exits the probing loop as the key is found.
		} else {
			// Block Logic: Moves to the next bucket using linear probing in case of a collision or no match.
			// Pre-condition: The current bucket `myMap[index]` does not contain `keys[i]`.
			// Invariant: `index` is incremented to probe the next contiguous memory location.
			index ++; // Inline: Advances to the next potential bucket in linear probing.

			// Block Logic: Handles hash table wrap-around to continue probing from the beginning of the table.
			// Pre-condition: `index` has exceeded the maximum valid index (`num - 1`).
			// Invariant: If `index` goes out of bounds, it is reset to `0` to continue probing from the start.
			if(index >= num)
				index = 0; // Inline: Resets `index` to the start of the hash map, implementing wrap-around.
		}
	}
}


/**
 * @brief CUDA kernel for inserting key-value pairs into the hash table in parallel.
 *
 * This kernel processes a batch of key-value pairs, computes their hash indices,
 * and inserts them into the GPU hash table. It uses linear probing for collision
 * resolution and atomic operations to ensure thread-safe insertions and updates.
 *
 * @param keys Device-side array of keys to insert.
 * @param numKeys The number of keys to insert.
 * @param values Device-side array of values corresponding to the keys.
 * @param myMap The GPU hash map structure where keys and values will be stored.
 * @param num The total number of buckets (size) in `myMap`.
 * @param list Device-side array of prime numbers for hash function computations.
 * @param current Device-side pointer to an integer tracking the current number of unique elements in the hash map.
 *                This parameter can be NULL if the count is not needed for a specific insertion (e.g., during reshape).
 */
__global__ void kernel_insert(int* keys, int numKeys, int* values,
	struct realMap *myMap, int num, size_t *list, int* current) {

	// Functional Utility: Computes a unique global thread index.
	// Inline: `threadIdx.x` provides the thread's ID within its block, `blockDim.x` is the number of threads per block,
	// and `blockIdx.x` is the block's ID within the grid. This calculation yields a unique global index for each thread.
	// Invariant: `i` correctly maps to a unique key in the `keys` array for processing by this thread.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	// Block Logic: Ensures that each thread processes a valid key within the bounds of the input array.
	// Pre-condition: `i` is the global thread index calculated for this thread.
	// Invariant: Only threads with `i < numKeys` are allowed to proceed, preventing out-of-bounds access.
	if(i >= numKeys)
		return;

	// Functional Utility: Calculates the initial hash index for the current key.
	int index = hash1(keys[i], num, list);

	/**
	 * Block Logic: Implements linear probing for insertion, handling potential collisions and updates.
	 * This loop continues until the key is successfully inserted into an empty slot or its value is updated
	 * if the key already exists.
	 * Pre-condition: `index` holds the initial hashed position for `keys[i]`.
	 * Invariant: The loop guarantees that `keys[i]` is eventually inserted or its value updated,
	 *            maintaining thread-safety through atomic operations.
	 */
	while(1) {
		// Block Logic: Checks if the current hash map entry's key matches the target key, indicating an update.
		// Pre-condition: `myMap[index]` contains a potential key-value pair.
		// Invariant: If `myMap[index].key` matches `keys[i]`, an atomic exchange updates the value, and the insertion terminates.
		if(myMap[index].key == keys[i]) {
			// Inline: Atomically updates the value associated with an existing key to prevent race conditions.
			atomicExch(&myMap[index].value, values[i]);
			return; // Inline: Exits the kernel as the key's value has been updated.
		}
		
		// Block Logic: Attempts to atomically insert the key into an empty slot.
		// Pre-condition: `myMap[index].key` is either `0` (empty) or a different key.
		// Invariant: `atomicCAS` ensures that `keys[i]` is written only if the slot is truly empty (`0`),
		//            preventing multiple threads from claiming the same empty slot concurrently.
		int result = atomicCAS(&myMap[index].key, 0, keys[i]);

		// Block Logic: If the atomic compare-and-swap (CAS) operation successfully inserted the key.
		// Pre-condition: `result` is `0`, indicating `myMap[index].key` was `0` before the CAS.
		// Invariant: The value is then atomically inserted, and the global count of elements is incremented (if `current` is not NULL).
		if(result == 0) {
			// Inline: Atomically exchanges the value into the newly claimed slot to prevent race conditions.
			atomicExch(&myMap[index].value, values[i]);
			// Block Logic: Atomically increments the total count of elements if a counter is provided.
			// Pre-condition: `current` is a valid pointer to the element counter.
			// Invariant: `current` is incremented by one using an atomic operation, maintaining thread-safe counting.
			if(current != NULL)
				atomicAdd(current, 1); // Inline: Atomically increments the number of elements in the hash table.
			break; // Inline: Exits the probing loop as insertion is complete.
		}
		else {
			// Block Logic: Moves to the next bucket using linear probing if the current slot is occupied by a different key.
			// Pre-condition: The atomic CAS failed, meaning `myMap[index].key` was not `0`.
			// Invariant: `index` is incremented to probe the next contiguous memory location.
			index ++; // Inline: Advances to the next potential bucket in linear probing.

			// Block Logic: Handles hash table wrap-around to continue probing from the beginning of the table.
			// Pre-condition: `index` has exceeded the maximum valid index (`num - 1`).
			// Invariant: If `index` goes out of bounds, it is reset to `0` to continue probing from the start.
			if(index >= num)
				index = 0; // Inline: Resets `index` to the start of the hash map, implementing wrap-around.
		}
	}
}

/**
 * @brief Constructor for the GpuHashTable class.
 *
 * Initializes a new GPU-based hash table with a specified initial size.
 * It allocates managed memory for the hash map, the current element count,
 * and copies prime numbers for hashing to device memory.
 *
 * @param size The initial number of buckets for the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	// Memory Allocation: Allocates managed memory for the hash map structure.
	// Managed memory allows both host and device to access the data.
	cudaMallocManaged(&hashMap, size * sizeof(struct realMap));
	cudaDeviceSynchronize(); // Inline: Ensures memory allocation completes before proceeding.

	// Error Handling: Checks if the hash map allocation was successful.
	DIE(hashMap == NULL, "allocation for hashMap failed");
	mapSize = size; // Inline: Stores the total capacity of the hash table.

	// Memory Allocation: Allocates managed memory for storing the current number of elements.
	cudaMallocManaged(&currentNo, sizeof(int));
	cudaDeviceSynchronize(); // Inline: Ensures memory allocation completes before proceeding.
	// Error Handling: Checks if the current element count allocation was successful.
	DIE(currentNo == NULL, "allocation for currentNo failed");

	// Initialization: Sets all key-value pairs in the newly allocated hash map to 0 (empty).
	// Pre-condition: `hashMap` is successfully allocated.
	// Invariant: All buckets are marked as empty before any insertions.
	for(int i = 0; i < size; i ++) {
		hashMap[i].key = 0; // Inline: Initializes key to 0, indicating an empty slot.
		hashMap[i].value = 0; // Inline: Initializes value to 0.
	}

	// Memory Allocation: Allocates managed memory for a list of prime numbers used by hash functions.
	cudaMallocManaged(&copyList, count() * sizeof(size_t));
	cudaDeviceSynchronize(); // Inline: Ensures memory allocation completes before proceeding.

	// Data Transfer: Copies the static `primeList` from host memory to the allocated device memory (`copyList`).
	// This makes prime numbers accessible to CUDA kernels.
	cudaMemcpy(copyList, primeList, count() * sizeof(size_t), cudaMemcpyHostToDevice);
}

/**
 * @brief Destructor for the GpuHashTable class.
 *
 * Frees all CUDA managed memory allocated by the hash table
 * to prevent memory leaks.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashMap); // Inline: Frees the memory allocated for the hash map.
	cudaFree(currentNo); // Inline: Frees the memory allocated for the current element count.
	cudaFree(copyList); // Inline: Frees the memory allocated for the prime number list.
}

/**
 * @brief Reshapes (resizes) the hash table to a new number of buckets.
 *
 * This function creates a new, larger hash table, rehashes all existing
 * elements from the old table into the new one, and then replaces the old
 * table with the new one. This is typically done when the load factor
 * becomes too high.
 *
 * @param numBucketsReshape The new desired number of buckets for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	// Memory Allocation: Allocates a temporary new hash map with the reshaped size.
	struct realMap *hashMapCopy = 0;
	cudaMallocManaged(&hashMapCopy, numBucketsReshape * sizeof(struct realMap));
	cudaDeviceSynchronize(); // Inline: Ensures memory allocation completes before proceeding.

	// Memory Allocation: Allocates temporary arrays to hold existing keys and values for re-insertion.
	int *keyCopy;
	cudaMallocManaged(&keyCopy, *currentNo * sizeof(int));
	cudaDeviceSynchronize(); // Inline: Ensures memory allocation completes before proceeding.

	int *valuesCopy;
	cudaMallocManaged(&valuesCopy, *currentNo * sizeof(int));
	cudaDeviceSynchronize(); // Inline: Ensures memory allocation completes before proceeding.

	// Initialization: Sets all key-value pairs in the new temporary hash map to 0 (empty).
	// Pre-condition: `hashMapCopy` is successfully allocated.
	// Invariant: All buckets in the new table are marked as empty before re-insertion begins.
	for(int i = 0; i < numBucketsReshape; i ++) {
		hashMapCopy[i].key = 0; // Inline: Initializes key to 0, indicating an empty slot.
		hashMapCopy[i].value = 0; // Inline: Initializes value to 0.
	}

	// Functional Utility: `x` acts as a counter for the number of valid key-value pairs copied from the old map.
	int x = 0;

	// Block Logic: Iterates through the old hash map to collect all currently stored key-value pairs.
	// Pre-condition: `hashMap` contains existing elements to be rehashed.
	// Invariant: All non-empty entries from the old `hashMap` are copied into `keyCopy` and `valuesCopy`.
	for(int i = 0; i < mapSize; i ++) {
		// Block Logic: Checks if the current bucket in the old hash map contains a valid key (not 0).
		// Pre-condition: `hashMap[i].key` holds the key for the current bucket.
		// Invariant: If the key is valid, it's added to the temporary arrays, and the counter `x` is incremented.
		if(hashMap[i].key != 0) {
			keyCopy[x] = hashMap[i].key; // Inline: Copies the key from the old map.
			valuesCopy[x] = hashMap[i].value; // Inline: Copies the value from the old map.
			x ++; // Inline: Increments the count of copied elements.
		}
	}

	// CUDA Launch: Calls the `kernel_insert` to re-insert all collected key-value pairs into the new `hashMapCopy`.
	// The `currentNo` counter is not incremented during reshape by passing `NULL` because the count is already known.
	// Inline: `<<<>>>` syntax specifies the kernel launch configuration (number of blocks and threads per block).
	kernel_insert>>(keyCopy, x, valuesCopy, hashMapCopy,
		numBucketsReshape, copyList, NULL);
	cudaDeviceSynchronize(); // Inline: Ensures all kernel operations complete before proceeding on the host.

	// Memory Management: Frees the memory of the old hash map.
	cudaFree(hashMap);

	// Update Pointers: Assigns the new hash map to the main `hashMap` pointer and updates `mapSize`.
	hashMap = hashMapCopy; // Inline: Updates the global `hashMap` pointer to the newly created map.
	mapSize = numBucketsReshape; // Inline: Updates the global `mapSize` to reflect the new capacity.
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 *
 * This function first checks if resizing the hash table is necessary based on the current load factor.
 * If needed, it calls `reshape`. Then, it copies the input keys and values to device memory
 * and launches a CUDA kernel to perform the insertions in parallel.
 *
 * @param keys Host-side array of keys to insert.
 * @param values Host-side array of values corresponding to the keys.
 * @param numKeys The number of key-value pairs to insert in this batch.
 * @return `false` always, as it doesn't represent a boolean outcome but rather completes the operation.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	// Block Logic: Checks if the hash table needs to be reshaped (resized) before insertion.
	// Pre-condition: `currentNo` holds the current number of elements, `numKeys` is the number of new elements.
	// Invariant: If the addition of `numKeys` would exceed `mapSize`, `reshape` is called to expand the table.
	if(*currentNo + numKeys >= mapSize) {
		// Functional Utility: Calculates a new size for the hash table, adding 5% overhead.
		reshape(*currentNo + numKeys + (*currentNo + numKeys) * 5 / 100);
		// Inline: Updates the `mapSize` to the new, increased capacity.
		mapSize = *currentNo + numKeys + (*currentNo + numKeys) * 5 / 100;
	}

	// Memory Allocation: Allocates managed memory for temporary key input to the kernel.
	int *keyCopy;
	cudaMallocManaged(&keyCopy, numKeys * sizeof(int));
	cudaDeviceSynchronize(); // Inline: Ensures memory allocation completes before proceeding.

	// Memory Allocation: Allocates managed memory for temporary value input to the kernel.
	int *valuesCopy;
	cudaMallocManaged(&valuesCopy, numKeys * sizeof(int));
	cudaDeviceSynchronize(); // Inline: Ensures memory allocation completes before proceeding.

	// Data Transfer: Copies host-side input keys and values to the device-accessible managed memory.
	memcpy(keyCopy, keys, numKeys * sizeof(int)); // Inline: Host-to-device copy of keys.
	memcpy(valuesCopy, values, numKeys * sizeof(int)); // Inline: Host-to-device copy of values.

	// CUDA Launch: Calls the `kernel_insert` to insert the batch of key-value pairs into the hash table.
	// `currentNo` is passed to atomically update the element count during insertion.
	// Inline: `<<<>>>` syntax specifies the kernel launch configuration (number of blocks and threads per block).
	kernel_insert>>(keyCopy, numKeys, valuesCopy,
		hashMap, mapSize, copyList, currentNo);
	cudaDeviceSynchronize(); // Inline: Ensures all kernel operations complete before returning.

	return false; // Inline: Returns a dummy value, as the function's primary purpose is side-effect (insertion).
}

/**
 * @brief Retrieves a batch of values for given keys from the hash table.
 *
 * This function copies the input keys to device memory and launches a CUDA kernel
 * to perform the retrievals in parallel. The results (values) are then returned
 * as a device-side array (managed memory).
 *
 * @param keys Host-side array of keys for which to retrieve values.
 * @param numKeys The number of keys to search for in this batch.
 * @return A device-side pointer to an array of integers containing the retrieved values.
 *         The caller is responsible for freeing this memory using `cudaFree`.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	// Memory Allocation: Allocates managed memory for storing the retrieval results.
	int* results = NULL;
	cudaMallocManaged(&results, numKeys * sizeof(int));
	cudaDeviceSynchronize(); // Inline: Ensures memory allocation completes before proceeding.

	// Memory Allocation: Allocates managed memory for temporary key input to the kernel.
	int *keyCopy;
	cudaMallocManaged(&keyCopy, numKeys * sizeof(int));
	cudaDeviceSynchronize(); // Inline: Ensures memory allocation completes before proceeding.

	// Data Transfer: Copies host-side input keys to the device-accessible managed memory.
	memcpy(keyCopy, keys, numKeys * sizeof(int)); // Inline: Host-to-device copy of keys.

	// CUDA Launch: Calls the `kernel_get` to retrieve values for the batch of keys.
	// Inline: `<<<>>>` syntax specifies the kernel launch configuration (number of blocks and threads per block).
	kernel_get>>(keyCopy, numKeys, hashMap,
		results, mapSize, copyList);
	cudaDeviceSynchronize(); // Inline: Ensures all kernel operations complete before returning.

	return results; // Inline: Returns the pointer to the device-side array of results.
}

/**
 * @brief Calculates the current load factor of the hash table.
 *
 * The load factor is defined as the ratio of the number of occupied buckets
 * to the total number of available buckets. It indicates how full the hash table is.
 *
 * @return A float representing the current load factor.
 */
float GpuHashTable::loadFactor() {
	// Functional Utility: Calculates the load factor by dividing the number of current elements by the total map size.
	return (float)(*currentNo) / (float)(mapSize); // Inline: Explicitly casts to float for floating-point division.
}

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0

// Macro for error checking and reporting.
#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)
	
/**
 * @brief A static array of prime numbers used in hash function calculations.
 * These primes are chosen to help distribute keys evenly across hash table buckets.
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
 * @brief Structure representing a single bucket in the hash map.
 * Each bucket stores a key-value pair.
 * A key of `0` typically indicates an empty bucket.
 */
struct realMap {
	int key;   ///< The key stored in this bucket.
	int value; ///< The value associated with the key in this bucket.
};


/**
 * @brief Computes the first hash function for a given data element.
 *
 * This function takes an integer data, a limit for the hash table size, and a list of prime numbers
 * to compute a hash index. It uses multiplication with a large prime from `primeList` and
 * then applies modulo operations to fit the hash within the `limit`.
 *
 * @param data The integer data (key) to hash.
 * @param limit The maximum index (size) of the hash table.
 * @param primeList1 Device-side array of prime numbers used for hashing.
 * @return An integer representing the computed hash index.
 */
__host__ __device__  int hash1(int data, int limit, size_t *primeList1) {
	// Functional Utility: Computes a hash index using a prime number from `primeList1`.
	// Inline: Uses absolute value of data, multiplies by a specific prime (index 64),
	// and applies modulo operations with other primes (index 90) and the hash table limit.
	// This helps distribute keys evenly and maps the hash to a valid bucket index.
	return ((long)abs(data) * primeList1[64]) % primeList1[90] % limit;
}

/**
 * @brief Computes the second hash function for a given data element.
 *
 * Similar to `hash1`, this function uses a different set of prime numbers from the
 * static `primeList` to generate an alternative hash index. This is typically
 * used in multi-hashing or double hashing schemes for collision resolution.
 *
 * @param data The integer data (key) to hash.
 * @param limit The maximum index (size) of the hash table.
 * @return An integer representing the computed hash index.
 */
int hash2(int data, int limit) {
	// Functional Utility: Computes a hash index using a prime number from the static `primeList`.
	// Inline: Uses absolute value of data, multiplies by a specific prime (index 67),
	// and applies modulo operations with other primes (index 91) and the hash table limit.
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}

/**
 * @brief Computes the third hash function for a given data element.
 *
 * Similar to `hash1` and `hash2`, this function uses yet another set of prime numbers
 * from the static `primeList` to generate a third alternative hash index.
 * Useful for advanced collision resolution strategies.
 *
 * @param data The integer data (key) to hash.
 * @param limit The maximum index (size) of the hash table.
 * @return An integer representing the computed hash index.
 */
int hash3(int data, int limit) {
	// Functional Utility: Computes a hash index using a prime number from the static `primeList`.
	// Inline: Uses absolute value of data, multiplies by a specific prime (index 70),
	// and applies modulo operations with other primes (index 93) and the hash table limit.
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}


/**
 * @brief Counts the number of elements in the `primeList` array.
 *
 * This function iterates through the `primeList` until it encounters a null terminator
 * (which is not standard for `size_t` arrays, implying a specific sentinel value usage).
 *
 * @return The number of elements found in the `primeList` plus one (likely to account for the sentinel).
 */
int count() {
	int i = 0;

	// Block Logic: Iterates through the `primeList` array.
	// Pre-condition: `primeList` is a valid array.
	// Invariant: The loop continues until a sentinel value (interpreted as `\0`) is found.
	while(primeList[i] != '\0') { // Inline: Checks for a 'null terminator' sentinel.
		i ++; // Inline: Increments counter for each non-sentinel element.
	}
	return i + 1; // Inline: Returns the count, adding one possibly for the sentinel itself.
}

/**
 * @brief Declaration of the GpuHashTable class.
 * This class provides an interface for managing a hash table on the GPU.
 * It includes methods for construction, destruction, resizing, batch insertion,
 * batch retrieval, and querying the load factor.
 */
class GpuHashTable
{
	public:
		/**
		 * @brief Constructor for GpuHashTable.
		 * @param size The initial size of the hash table.
		 */
		GpuHashTable(int size);

		/**
		 * @brief Resizes the hash table to a new capacity.
		 * @param sizeReshape The new desired number of buckets.
		 */
		void reshape(int sizeReshape);
		
		/**
		 * @brief Inserts a batch of key-value pairs into the hash table.
		 * @param keys Host-side array of keys to insert.
		 * @param values Host-side array of values to insert.
		 * @param numKeys The number of key-value pairs in the batch.
		 * @return Always returns `false`.
		 */
		bool insertBatch(int *keys, int* values, int numKeys);

		/**
		 * @brief Retrieves a batch of values for given keys.
		 * @param key Host-side array of keys to search for.
		 * @param numItems The number of keys to search for.
		 * @return A device-side pointer to an array of retrieved values.
		 */
		int* getBatch(int* key, int numItems);
		
		/**
		 * @brief Calculates the current load factor of the hash table.
		 * @return The current load factor as a float.
		 */
		float loadFactor();

		/**
		 * @brief Placeholder function for occupancy calculation (currently not implemented).
		 */
		void occupancy();

		/**
		 * @brief Placeholder function for printing hash table information (currently not implemented).
		 * @param info A string containing information to print.
		 */
		void print(string info);
	
		/**
		 * @brief Destructor for GpuHashTable.
		 */
		~GpuHashTable();
};

#endif
