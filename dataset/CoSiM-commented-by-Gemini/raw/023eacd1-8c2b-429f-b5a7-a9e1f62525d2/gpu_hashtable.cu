/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA for parallel insertion and retrieval.
 *
 * This file provides a `GpuHashTable` class designed to leverage NVIDIA GPUs
 * for high-performance hash table operations. It includes host-side management
 * functions (constructor, destructor, reshape, insertBatch, getBatch, loadFactor)
 * and device-side CUDA kernels (`kernel_insert`, `kernel_get`) that perform
 * parallel hash table insertions and lookups, respectively.
 * The implementation uses linear probing for collision resolution and atomic operations
 * for thread-safe access to the hash table.
 *
 * Data Structures:
 * - `struct node`: Represents a key-value pair in the hash table.
 * - `GpuHashTable`: Manages the hash table's state on both host and device.
 *
 * Algorithms:
 * - Hashing: Custom `hash_function` for distributing keys.
 * - Collision Resolution: Linear probing.
 * - Parallel Operations: CUDA kernels utilize atomic operations for concurrent updates.
 *
 * Time Complexity:
 * - `insertBatch`, `getBatch`: O(numKeys / num_threads) in ideal case (no collisions),
 *   O(numKeys * average_probe_length) in worst case on device.
 * - `reshape`: O(old_size + new_size) for copying and re-inserting elements.
 * Space Complexity: O(hashtable_size) for the hash table, O(numKeys) for batch operations.
 */

#include "gpu_hashtable.hpp"

// Functional Utility: A constant representing an invalid key, used to denote empty slots in the hash table.
#define KEY_INVALID 0

/**
 * @brief Constructs a new GpuHashTable instance.
 * @param size The initial desired size for the hash table (number of buckets).
 *
 * This constructor initializes the hash table on unified memory (accessible by both CPU and GPU).
 * It allocates space for the hash table nodes and an integer to track its current filled size.
 * Each bucket's key is initialized to KEY_INVALID to mark it as empty.
 */
GpuHashTable::GpuHashTable(int size) {
	// Functional Utility: Stores the initial size of the hash table.
	hashtable_size = size;

	// HPC: Allocates memory for hashtable_current_size on unified memory.
	// This allows both host and device to access and update the count of elements.
	cudaMallocManaged((void **) &hashtable_current_size, sizeof(int));
	// Functional Utility: Initializes the current size of the hash table to zero.
	*hashtable_current_size = 0;

	// HPC: Allocates memory for the hash table (array of struct node) on unified memory.
	cudaMallocManaged((void **) &hashtable, size * sizeof(struct node));
	// Block Logic: Initializes all keys in the newly allocated hash table to KEY_INVALID.
	// This marks all buckets as empty and available.
	for (int i = 0; i < size; i++)
		hashtable[i].key = KEY_INVALID;
}

/**
 * @brief Destroys the GpuHashTable instance.
 *
 * This destructor frees all CUDA-allocated memory associated with the hash table,
 * including the hash table itself and the current size counter.
 */
GpuHashTable::~GpuHashTable() {
	// HPC: Frees the memory allocated for the hash table on the GPU.
	cudaFree(hashtable);
	// HPC: Frees the memory allocated for the current size counter on the GPU.
	cudaFree(hashtable_current_size);
}

void GpuHashTable::reshape(int numBucketsReshape) {
	// Functional Utility: Calculates the new size for the hash table, applying a load factor heuristic.
	int new_size = numBucketsReshape * 1.07f;

	// Functional Utility: Stores the current number of elements before reshaping.
	int old_size = *hashtable_current_size;

	// Functional Utility: Allocates host memory to temporarily store existing keys and values.
	int *old_keys = (int *)malloc(*(hashtable_current_size) * sizeof(int));
	int *old_values = (int *)malloc(*(hashtable_current_size) *sizeof(int));
	
	int index = 0;

	/**
	 * Block Logic: Iterates through the old hash table to extract all active key-value pairs.
	 * These pairs are temporarily stored in host arrays (`old_keys`, `old_values`).
	 * Pre-condition: The old hash table contains valid data up to `hashtable_size`.
	 * Invariant: All non-invalid key-value pairs are transferred to the temporary host arrays.
	 */
	for (int i = 0; i < hashtable_size; i++) {
		// Block Logic: Checks if the current hash table entry is active (not KEY_INVALID).
		if (hashtable[i].key != KEY_INVALID) {
			old_keys[index] = hashtable[i].key;
			old_values[index] = hashtable[i].value;
			index++;
		}
	}

	// HPC: Frees the memory of the old hash table and its current size counter on the GPU.
	cudaFree(hashtable);
	cudaFree(hashtable_current_size);

	// Functional Utility: Updates the hash table's size to the new calculated size.
	hashtable_size = new_size;

	// HPC: Allocates new memory for `hashtable_current_size` on unified memory.
	cudaMallocManaged((void **) &hashtable_current_size, sizeof(int));
	// Functional Utility: Resets the current size counter for the new hash table.
	*hashtable_current_size = 0;

	// HPC: Allocates new memory for the hash table with the `new_size` on unified memory.
	cudaMallocManaged((void **) &hashtable, new_size * sizeof(struct node));
	// Block Logic: Initializes all keys in the new hash table to KEY_INVALID, marking all buckets as empty.
	for (int i = 0; i < new_size; i++)
		hashtable[i].key = KEY_INVALID;

	// Functional Utility: Re-inserts all saved key-value pairs into the newly resized hash table.
	insertBatch(old_keys, old_values, old_size);

	// Functional Utility: Frees the host memory used for temporary storage of old key-value pairs.
	free(old_keys);
	free(old_values);
}

/**
 * @brief CUDA kernel for parallel insertion of key-value pairs into the hash table.
 * @param keys Pointer to an array of keys to insert (device memory).
 * @param values Pointer to an array of corresponding values to insert (device memory).
 * @param numKeys The total number of key-value pairs to insert.
 * @param hashtable Pointer to the GPU hash table (unified memory).
 * @param hashtable_size The total number of buckets in the hash table.
 * @param hashtable_current_size Pointer to an integer tracking the current number of elements in the hash table (unified memory).
 *
 * This kernel assigns one thread per key to be inserted. It uses linear probing for
 * collision resolution and atomic compare-and-swap (atomicCAS) to ensure thread-safe
 * insertion of keys and atomic exchange (atomicExch) for values, and atomic addition
 * (atomicAdd) to update the current size.
 */
__global__ void kernel_insert(int *keys, int *values, int numKeys,
struct node *hashtable, int hashtable_size,
int *hashtable_current_size) {
	// HPC: Calculates a unique global index for the current thread.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// Block Logic: Ensures that the thread index is within the bounds of the `numKeys` to process.
	if (idx >= numKeys)
		return;

	// Functional Utility: Retrieves the key and value for the current thread to insert.
	int my_key = keys[idx];
	int my_value = values[idx];

	// Functional Utility: Computes the initial hash index for the key.
	int hash = hash_function(my_key, hashtable_size);
	
	/**
	 * Block Logic: Implements linear probing for insertion.
	 * Invariant: The loop continues until the key is successfully inserted or its value updated.
	 */
	while (1) {
		// HPC: Attempts to atomically insert the key into the hash table.
		// `atomicCAS` compares `hashtable[hash].key` with `KEY_INVALID`. If they are equal,
		// `hashtable[hash].key` is updated to `my_key`. The original value of `hashtable[hash].key` is returned.
		int init_val = atomicCAS(&(hashtable[hash].key), KEY_INVALID,
			my_key);

		// Block Logic: If the slot was empty (`KEY_INVALID`), the key was successfully inserted.
		if (init_val == KEY_INVALID) {
			// HPC: Atomically updates the value for the newly inserted key.
			automicExch(&hashtable[hash].value, my_value);
			// HPC: Atomically increments the count of elements in the hash table.
			automicAdd(hashtable_current_size, 1);

			// HPC: Synchronizes all threads in the current block to ensure memory consistency.
			__syncthreads();
			break;
		}
		// Block Logic: If the slot already contains the same key, update its value.
		else if (init_val == my_key) {
			// HPC: Atomically updates the value associated with the existing key.
			automicExch(&hashtable[hash].value, my_value);

			// HPC: Synchronizes all threads in the current block.
			__syncthreads();
                        break;
		}

		// Functional Utility: Moves to the next bucket (linear probing) in case of collision or existing key mismatch.
		hash = (hash + 1) % hashtable_size;
	}
}

bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	// Block Logic: Checks if there are any keys to insert. If not, returns true immediately.
	if (numKeys <= 0)
		return true;

	// Block Logic: Checks if the current hash table size is sufficient for the new insertions.
	// If not, it triggers a reshape operation to resize the hash table.
	if (*hashtable_current_size + numKeys > hashtable_size)
		reshape(hashtable_size + numKeys);

	// HPC: Declares device pointers for keys and values.
	int *device_keys, *device_values;

	// HPC: Allocates device memory for the input keys and values.
	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));

	// HPC: Copies the keys from host memory to device memory.
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int),
		cudaMemcpyHostToDevice);
	// HPC: Copies the values from host memory to device memory.
	cudaMemcpy(device_values, values, numKeys * sizeof(int),
		cudaMemcpyHostToDevice);
	
	
	// HPC: Launches the CUDA kernel for parallel insertion.
	// The kernel is configured to use a grid and block size suitable for `numKeys`.
	// Each thread processes one key-value pair.
	kernel_insert>>>(device_keys, device_values, numKeys,
		hashtable, hashtable_size, 
		hashtable_current_size);
	// HPC: Synchronizes the device to ensure all kernel operations complete before proceeding on the host.
	cudaDeviceSynchronize();

	// HPC: Frees the device memory allocated for input keys and values.
	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}

/**
 * @brief CUDA kernel for parallel retrieval of values from the hash table.
 * @param keys Pointer to an array of keys to search for (device memory).
 * @param values Pointer to an array where retrieved values will be stored (device memory).
 * @param numKeys The total number of keys to search for.
 * @param hashtable Pointer to the GPU hash table (unified memory).
 * @param hashtable_size The total number of buckets in the hash table.
 *
 * This kernel assigns one thread per key to be retrieved. It uses linear probing
 * to find the key in the hash table and writes the corresponding value into the
 * `values` array. If a key is not found, the corresponding value in the `values`
 * array will remain its initialized state (typically 0 from `calloc`).
 */
__global__ void kernel_get(int *keys, int *values, int numKeys,
struct node *hashtable, int hashtable_size) {
	// HPC: Calculates a unique global index for the current thread.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// Block Logic: Ensures that the thread index is within the bounds of the `numKeys` to process.
	if (idx >= numKeys)
		return;

	// Functional Utility: Retrieves the key for the current thread to search for.
	int my_key = keys[idx];
	// Functional Utility: Computes the initial hash index for the key.
	int hash = hash_function(my_key, hashtable_size);

	/**
	 * Block Logic: Implements linear probing to find the key.
	 * Invariant: The loop continues until the key is found. Assumes key will always be found
	 * if present, or it will eventually wrap around to an empty spot or its own initial spot.
	 */
	while (1) {
		// Block Logic: If the key at the current hash index matches `my_key`, the value is retrieved.
		if (hashtable[hash].key == my_key) {
			values[idx] = hashtable[hash].value;

			// HPC: Synchronizes all threads in the current block to ensure memory consistency.
			__syncthreads();
			break;; 
		}

		// Functional Utility: Moves to the next bucket (linear probing) if the current bucket does not contain the key.
        hash = (hash + 1) % hashtable_size;
	}
}

/**
 * @brief Performs a batch retrieval of values for a given set of keys.
 * @param keys Pointer to an array of keys to retrieve (host memory).
 * @param numKeys The total number of keys to retrieve.
 * @return A pointer to a newly allocated array of values (host memory), corresponding to the input keys.
 *         The caller is responsible for freeing this memory.
 *
 * This method allocates device memory for keys and values, copies the input keys
 * to the device, launches a CUDA kernel for parallel retrieval, copies the results
 * back to host memory, and then frees the temporary device memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	// Functional Utility: Allocates host memory for the result values, initialized to zero.
	int *values = (int *)calloc(numKeys, sizeof(int));

	// HPC: Declares device pointers for keys and values.
	int *device_keys, *device_values;

	// HPC: Allocates device memory for the input keys and output values.
	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));

	// HPC: Copies the input keys from host memory to device memory.
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int),
		cudaMemcpyHostToDevice);
	// HPC: Copies the initialized values array from host memory to device memory.
	cudaMemcpy(device_values, values, numKeys * sizeof(int),
		cudaMemcpyHostToDevice);

	// HPC: Launches the CUDA kernel for parallel retrieval.
	// The kernel is configured to use a grid and block size suitable for `numKeys`.
	// Each thread processes one key.
	kernel_get>>>(device_keys, device_values, numKeys,
		hashtable, hashtable_size);
	// HPC: Synchronizes the device to ensure all kernel operations complete before proceeding on the host.
	cudaDeviceSynchronize();

	// HPC: Copies the retrieved values from device memory back to host memory.
	cudaMemcpy(values, device_values, numKeys * sizeof(int),
		cudaMemcpyDeviceToHost);

	// HPC: Frees the device memory allocated for temporary storage of keys and values.
	cudaFree(device_keys);
	cudaFree(device_values);

	// Functional Utility: Returns the pointer to the host array containing the retrieved values.
	return values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float (current_size / hashtable_size).
 *
 * The load factor indicates how full the hash table is. A high load factor
 * can lead to increased collision rates and slower performance.
 */
float GpuHashTable::loadFactor() {
	// Functional Utility: Computes the ratio of currently stored elements to the total number of buckets.
	return *(hashtable_current_size) / (float)hashtable_size;
}



// Functional Utility: Macro to initialize a GpuHashTable instance with a default size of 1.
#define HASH_INIT GpuHashTable GpuHashTable(1);
// Functional Utility: Macro to reshape the GpuHashTable to accommodate a specified number of buckets.
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

// Functional Utility: Macro to perform a batch insertion of key-value pairs into the GpuHashTable.
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
// Functional Utility: Macro to perform a batch retrieval of values for given keys from the GpuHashTable.
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

// Functional Utility: Macro to get the current load factor of the GpuHashTable.
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

// Functional Utility: Includes test-related functionalities, likely for benchmarking or validation.
#include "test_map.cpp"


#ifndef _HASHCPU_
#define _HASHCPU_

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <chrono>

using namespace std;

// Functional Utility: Constant representing an invalid key, used to denote empty slots.
#define KEY_INVALID 0

// Functional Utility: Macro for error handling and exiting the program.
// It prints an error message including file and line number, and the system error.
#define DIE(assertion, call_description)              
	do                                            
	{
		if (assertion)                        
		{
			fprintf(stderr, "(%s, %d): ", 
					__FILE__, __LINE__);  
			perror(call_description);     
			exit(errno);                  
		}
	}
	while (0)

/**
 * @brief An array of large prime numbers.
 *
 * This list of prime numbers is likely used in the hash function
 * to reduce collisions and ensure a more uniform distribution of keys
 * across the hash table. The use of large primes is a common technique
 * in hashing for better performance and randomness.
 */
const size_t primeList[] = {
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
	11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu};

__device__ int hash_function(int data, int limit) {
        return ((long)abs(data) * 21407807219llu) % 139193449418173llu % limit;
}

struct node {
	int key;
	int value;
};


class GpuHashTable
{
public:
	struct node *hashtable;
	int hashtable_size;
	int *hashtable_current_size;
	
GpuHashTable(int size);
	void reshape(int sizeReshape);

	bool insertBatch(int *keys, int *values, int numKeys);
	int *getBatch(int *key, int numItems);

	float loadFactor();
	void occupancy();
	void print(string info);

	~GpuHashTable();
};

#endif