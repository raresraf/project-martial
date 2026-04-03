
/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA, supporting insertion, retrieval, and dynamic resizing.
 *
 * This file contains the CUDA device functions and global kernels for parallel hash table operations,
 * as well as the host-side GpuHashTable class methods to manage memory and orchestrate kernel launches.
 * This particular implementation utilizes `struct Node` and `struct HashTable` for device-side data representation.
 *
 * Domain: High-Performance Computing (HPC), GPU Computing, Data Structures (Hash Table).
 * Algorithm: Linear probing for collision resolution, `atomicCAS` for thread-safe operations.
 * Time Complexity:
 *   - Insertion/Retrieval (average): O(1)
 *   - Insertion/Retrieval (worst case, on collision): O(N) where N is table size
 *   - Reshape: O(M) where M is number of elements, as elements are rehashed.
 * Space Complexity: O(S) where S is the size of the hash table.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief Device-side hash function for mapping a key to a hash table index.
 * @param key The integer value to be hashed.
 * @param capacity The total number of buckets in the hash table (upper bound for the hash index).
 * @return An integer representing the hash value within the specified capacity.
 *
 * This function computes a hash value for a given key, ensuring the result
 * fits within the hash table's capacity. It uses a multiplicative hashing approach
 * with large prime numbers to minimize collisions and distribute keys uniformly.
 */
__device__ int getHashCode(int key, int capacity) {
	return ((long long)abs(key) * 1646237llu) % 67965551447llu % capacity;
}

/**
 * @brief CUDA kernel for parallel insertion of key-value pairs into the hash table.
 * @param keys Pointer to an array of keys to insert.
 * @param values Pointer to an array of values corresponding to the keys.
 * @param N The total number of key-value pairs to insert.
 * @param hashtable The HashTable struct containing device pointers and metadata.
 *
 * Each CUDA thread processes one key-value pair. It calculates an initial hash
 * and then attempts to insert the pair using linear probing with `atomicCAS`
 * to handle collisions and ensure atomicity. If a key already exists or a slot
 * is empty (indicated by key == 0), the value is updated.
 *
 * Block Logic:
 *   - Thread Mapping: `i` uniquely identifies the key-value pair processed by each thread.
 *   - Collision Resolution: Implements linear probing, wrapping around the table if necessary.
 *   - Concurrency Control: Uses `atomicCAS` (Compare-And-Swap) to ensure that
 *     only one thread successfully writes to a slot if multiple threads
 *     attempt to write to the same slot simultaneously, or to insert into an
 *     empty slot (key == 0).
 */
__global__ void kernel_insert(int *keys, int *values, int N, HashTable hashtable) {

	// Inline: Calculate a unique global index for each thread.
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// Pre-condition: Ensure the thread's index is within the bounds of the number of keys.
	if (i < N) {

		// Block Logic: Calculate the initial hash value for the current key.
		int idx = getHashCode(keys[i], hashtable.capacity);

		// Block Logic: Linear probing starting from the calculated `idx` to `hashtable.capacity`.
		// Invariant: Attempts to find an empty slot (key == 0) or a slot with the same key to update.
		for (int k = idx; k < hashtable.capacity; k++) {
			// Inline: Atomically attempt to insert the key or retrieve the existing key.
			// `0` is used as the sentinel for an empty slot.
			int old_key = atomicCAS(&hashtable.nodes[k].key, 0, keys[i]);
			// Post-condition: If the slot was empty or contained the same key, update the value and return.
			if (!old_key || old_key == keys[i]) {
				hashtable.nodes[k].value = values[i];
				return;
			}
		}

		// Block Logic: If no slot was found in the first loop, continue linear probing
		// from the beginning of the table up to the initial `idx` (wrap-around).
		for (int k = 0; k < idx; k++) {
			int old_key = atomicCAS(&hashtable.nodes[k].key, 0, keys[i]);
			// Post-condition: If the slot was empty or contained the same key, update the value and return.
			if (!old_key || old_key == keys[i]) {
				hashtable.nodes[k].value = values[i];
				return;
			}
		}
	}
}

/**
 * @brief CUDA kernel for reshaping (resizing) the hash table.
 * @param old_hashtable The old HashTable struct.
 * @param new_hashtable The new, larger HashTable struct.
 *
 * Each CUDA thread processes one entry from the `old_hashtable`. If the entry
 * is valid (key != 0), it calculates a new hash value for it based on the
 * `new_hashtable.capacity` and attempts to insert it into the `new_hashtable`
 * using linear probing and `atomicCAS` to ensure thread-safe insertion.
 *
 * Block Logic:
 *   - Thread Mapping: `i` uniquely identifies an entry in the old hash table
 *     to be rehashed by each thread.
 *   - Pre-condition: Checks if the entry in the old hash table is valid (key != 0).
 *   - Re-hashing: Recalculates the hash for valid entries using the new table's capacity.
 *   - Collision Resolution: Implements linear probing in the `new_hashtable`,
 *     wrapping around if necessary.
 *   - Concurrency Control: Uses `atomicCAS` to ensure thread-safe insertion
 *     of elements from the old table into the new table, preventing race conditions.
 */
__global__ void kernel_reshape(HashTable old_hashtable, HashTable new_hashtable) {

	// Inline: Calculate a unique global index for each thread.
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// Pre-condition: Ensure the thread's index is within the bounds of the old table's capacity.
	if (i < old_hashtable.capacity) {

		// Pre-condition: Process only valid entries from the old hash table (key != 0).
		if (old_hashtable.nodes[i].key != 0) {

			// Block Logic: Calculate the new hash value for the key based on the new table's capacity.
			int idx = getHashCode(old_hashtable.nodes[i].key, new_hashtable.capacity);

			// Block Logic: Linear probing starting from `idx` to `new_hashtable.capacity`.
			// Invariant: Attempts to find an empty slot (key == 0) in the new hash table for the current key.
			for (int k = idx; k < new_hashtable.capacity; k++) {
				// Inline: Atomically attempt to insert the key into an empty slot in the new table.
				int old_key = atomicCAS(&new_hashtable.nodes[k].key, 0, old_hashtable.nodes[i].key);
	
				// Post-condition: If the slot was empty, place the value and return.
				if (!old_key) {
					new_hashtable.nodes[k].value = old_hashtable.nodes[i].value;
					return;
				}
			}
	
			// Block Logic: If no slot was found in the first loop, continue linear probing
			// from the beginning of the new table up to `idx` (wrap-around).
			for (int k = 0; k < idx; k++) {
				int old_key = atomicCAS(&new_hashtable.nodes[k].key, 0, old_hashtable.nodes[i].key);
	
				// Post-condition: If the slot was empty, place the value and return.
				if (!old_key) {
					new_hashtable.nodes[k].value = old_hashtable.nodes[i].value;
					return;
				}
			}
		}
	}
}

/**
 * @brief CUDA kernel for parallel retrieval of values from the hash table.
 * @param keys Pointer to an array of keys to search for.
 * @param values Pointer to an array where the retrieved values will be stored.
 * @param N The total number of keys to search for.
 * @param hashtable The HashTable struct containing device pointers and metadata.
 *
 * Each CUDA thread processes one key. It calculates an initial hash
 * and then searches for the key using linear probing. If the key is found,
 * its corresponding value is written to the `values` array. If not found,
 * the corresponding entry in `values` remains unchanged (or default initialized to 0).
 *
 * Block Logic:
 *   - Thread Mapping: `i` uniquely identifies the key searched by each thread.
 *   - Collision Resolution: Implements linear probing, wrapping around the table if necessary,
 *     to locate the target key.
 */
__global__ void kernel_get(int *keys, int *values, int N, HashTable hashtable) {
	// Inline: Calculate a unique global index for each thread.
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// Pre-condition: Ensure the thread's index is within the bounds of the number of keys.
	if (i < N) {

		// Block Logic: Calculate the initial hash value for the current key.
		int idx = getHashCode(keys[i], hashtable.capacity);

		// Block Logic: Linear probing starting from `idx` to `hashtable.capacity`.
		// Invariant: Searches for a slot containing the target key.
		for (int k = idx; k < hashtable.capacity; k++) {
			// Pre-condition: Check if the current slot's key matches the target key.
			if (hashtable.nodes[k].key == keys[i]) {
				values[i] = hashtable.nodes[k].value;
				return;
			}
		}

		// Block Logic: If not found in the first loop, continue linear probing
		// from the beginning of the table up to `idx` (wrap-around).
		for (int k = 0; k < idx; k++) {
			// Pre-condition: Check if the current slot's key matches the target key.
			if (hashtable.nodes[k].key == keys[i]) {
				values[i] = hashtable.nodes[k].value;
				return;
			}
		}
	}
}


GpuHashTable::GpuHashTable(int size) {

	/**
	 * @brief Initializes a new instance of the GpuHashTable class.
	 * @param size The initial capacity for the hash table.
	 *
	 * This constructor sets up the host-side representation (`hashtable` struct)
	 * and allocates memory for the hash table nodes on the GPU. It also
	 * initializes all allocated device memory to 0 to mark nodes as empty.
	 */
	// Block Logic: Initialize the hash table size and capacity.
	hashtable.size = 0;
	hashtable.capacity = size;

	const int num_bytes = size * sizeof(Node);

	// Block Logic: Allocate device memory for the hash table nodes.
	cudaMalloc(&hashtable.nodes, num_bytes);
	if (!hashtable.nodes) {
		// Pre-condition: Check for successful memory allocation.
		return; // Error handling
	}

	// Block Logic: Initialize all allocated device memory to 0 (KEY_INVALID).
	cudaMemset(hashtable.nodes, 0, num_bytes);
}


GpuHashTable::~GpuHashTable() {
	/**
	 * @brief Destroys the GpuHashTable instance, freeing allocated GPU memory.
	 *
	 * This destructor deallocates the device memory previously allocated for the hash table nodes.
	 */
	cudaFree(hashtable.nodes);
}


void GpuHashTable::reshape(int numBucketsReshape) {

	HashTable new_hashtable;
	
	/**
	 * @brief Reshapes the hash table to a new, larger capacity.
	 * @param numBucketsReshape The suggested new capacity for the hash table.
	 *
	 * This method creates a new `HashTable` instance with an increased capacity,
	 * typically `numBucketsReshape / 0.8f` to maintain a desired load factor.
	 * It allocates and initializes new device memory, then launches the `kernel_reshape`
	 * to rehash and transfer all valid entries from the old table to the new one.
	 * Finally, it frees the old table's device memory and updates the `hashtable` member
	 * to point to the new table.
	 */
	// Block Logic: Calculate the actual new capacity based on a desired load factor (0.8).
	new_hashtable.capacity = (int)(numBucketsReshape / 0.8f);

	// Block Logic: Preserve the current number of elements from the old hash table.
	new_hashtable.size = hashtable.size;

	const int num_bytes = new_hashtable.capacity * sizeof(Node);

	// Block Logic: Allocate device memory for the new hash table nodes.
	cudaMalloc(&new_hashtable.nodes, num_bytes);
	if (!new_hashtable.nodes) {
		// Pre-condition: Check for successful memory allocation.
		return; // Error handling
	}

	// Block Logic: Initialize all allocated device memory for the new table to 0 (KEY_INVALID).
	cudaMemset(new_hashtable.nodes, 0, num_bytes);

	// Block Logic: Calculate the number of blocks needed for the kernel launch.
	// Invariant: Ensures enough blocks are launched to cover all elements in the old hash table.
	unsigned int blocks_no = hashtable.capacity / 1024;

	// Inline: Adjust block count if there's a remainder.
	if (hashtable.capacity % 1024) {
		++blocks_no;
	}

	// Block Logic: Launch the kernel to rehash and transfer elements from the old table to the new.
	kernel_reshape<<<blocks_no, 1024>>>(hashtable, new_hashtable);

	// Block Logic: Synchronize the device to ensure all kernel operations complete before proceeding.
	cudaDeviceSynchronize();

	// Block Logic: Free the memory of the old hash table.
	cudaFree(hashtable.nodes);

	// Block Logic: Update the host-side `hashtable` struct to point to the newly created hash table.
	hashtable = new_hashtable;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {

	const int num_bytes = numKeys * sizeof(int);

	int *device_keys = 0;
	int *device_values = 0;

	/**
	 * @brief Inserts a batch of key-value pairs into the GPU hash table.
	 * @param keys Pointer to an array of keys on the host.
	 * @param values Pointer to an array of values corresponding to the keys.
	 * @param numKeys The number of key-value pairs to insert.
	 * @return True if the insertion is successful, false otherwise.
	 *
	 * This method manages the entire batch insertion process, including:
	 * 1. Allocating temporary device memory for input keys and values.
	 * 2. Checking the hash table's load factor and triggering a reshape operation if necessary.
	 * 3. Copying host data to device memory.
	 * 4. Launching the `kernel_insert` to perform parallel insertions on the GPU.
	 * 5. Synchronizing the device to ensure all kernel operations complete.
	 * 6. Updating the internal count of inserted pairs.
	 * 7. Freeing temporary device memory.
	 */
	// Block Logic: Allocate temporary device memory for keys.
	cudaMalloc(&device_keys, num_bytes);
	// Block Logic: Allocate temporary device memory for values.
	cudaMalloc(&device_values, num_bytes);

	// Pre-condition: Check if device memory allocation was successful.
	if (!device_keys || !device_values) {
		// Inline: Clean up any allocated memory before returning false.
		if (device_keys) cudaFree(device_keys);
		if (device_values) cudaFree(device_values);
		return false;
	}

	// Block Logic: Check if the current load factor exceeds a threshold,
	// triggering a reshape operation to a larger table if necessary.
	// The new capacity is calculated to maintain a desired load factor after insertion.
	float new_factor = (float)(hashtable.size + numKeys) / (float)hashtable.capacity;
	if (new_factor > 0.8f) {
		reshape(hashtable.size + numKeys);
	}

	// Block Logic: Copy keys from host memory to device memory.
	cudaMemcpy(device_keys, keys, num_bytes, cudaMemcpyHostToDevice);
	// Block Logic: Copy values from host memory to device memory.
	cudaMemcpy(device_values, values, num_bytes, cudaMemcpyHostToDevice);

	// Block Logic: Calculate the number of blocks needed for the kernel launch.
	// Invariant: Ensures enough blocks are launched to cover all key-value pairs in the batch.
	unsigned int blocks_no = numKeys / 1024;

	// Inline: Adjust block count if there's a remainder.
	if (numKeys % 1024) {
		++blocks_no;
	}

	// Block Logic: Launch the kernel to perform parallel insertions.
	kernel_insert<<<blocks_no, 1024>>>(device_keys, 
											device_values, 
											numKeys, 
											hashtable);

	// Block Logic: Synchronize the device to ensure all kernel operations complete before proceeding.
	cudaDeviceSynchronize();

	// Block Logic: Update the internal count of inserted pairs in the hash table.
	hashtable.size += numKeys;

	// Block Logic: Free temporary device memory for keys.
	cudaFree(device_keys);
	// Block Logic: Free temporary device memory for values.
	cudaFree(device_values);

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {

	int *host_values = 0;
	int *device_keys = 0;
	int *device_values = 0;

	const int num_bytes = numKeys * sizeof(int);

	/**
	 * @brief Retrieves a batch of values corresponding to a batch of keys from the GPU hash table.
	 * @param keys Pointer to an array of keys on the host to search for.
	 * @param numKeys The number of keys to search for.
	 * @return A pointer to an array of retrieved values on the host. Returns NULL on error.
	 *
	 * This method manages the batch retrieval process, including:
	 * 1. Allocating host memory for results and temporary device memory for keys and values.
	 * 2. Copying keys from host to device memory.
	 * 3. Launching the `kernel_get` to perform parallel lookups on the GPU.
	 * 4. Synchronizing the device to ensure all kernel operations complete.
	 * 5. Copying the results (values) back to host memory.
	 * 6. Freeing temporary device memory.
	 */
	// Block Logic: Allocate host memory for the retrieved values.
	host_values = (int *) malloc(num_bytes);

	// Block Logic: Allocate temporary device memory for keys.
	cudaMalloc(&device_keys, num_bytes);
	// Block Logic: Allocate temporary device memory for values.
	cudaMalloc(&device_values, num_bytes);

	// Pre-condition: Check if memory allocation was successful for all pointers.
	if(!host_values || !device_keys || !device_values) {
		// Inline: Clean up any allocated memory before returning NULL.
		if (host_values) free(host_values);
		if (device_keys) cudaFree(device_keys);
		if (device_values) cudaFree(device_values);
		return NULL;
	}

	// Block Logic: Copy keys from host memory to device memory.
	cudaMemcpy(device_keys, keys, num_bytes, cudaMemcpyHostToDevice);

	// Block Logic: Calculate the number of blocks needed for the kernel launch.
	// Invariant: Ensures enough blocks are launched to cover all keys in the batch.
	unsigned int blocks_no = numKeys / 1024;
	
	// Inline: Adjust block count if there's a remainder.
	if (numKeys % 1024) {
		++blocks_no;
	}

	// Block Logic: Launch the kernel to perform parallel value retrievals.
	kernel_get<<<blocks_no, 1024>>>(device_keys,
										device_values, 
										numKeys, 
										hashtable);

	// Block Logic: Synchronize the device to ensure all kernel operations complete before proceeding.
	cudaDeviceSynchronize();

	// Block Logic: Copy the retrieved values from device memory to host memory.
	cudaMemcpy(host_values, device_values, num_bytes, cudaMemcpyDeviceToHost);

	// Block Logic: Free temporary device memory for keys.
	cudaFree(device_keys);
	// Block Logic: Free temporary device memory for values.
	cudaFree(device_values);

	return host_values;
}


float GpuHashTable::loadFactor() {
	/**
	 * @brief Calculates the current load factor of the hash table.
	 * @return The current load factor as a float.
	 *
	 * The load factor is defined as the ratio of the number of inserted pairs (`hashtable.size`)
	 * to the total capacity of the hash table (`hashtable.capacity`). This metric indicates
	 * how full the hash table is and is used to determine when a reshape operation
	 * is necessary to maintain performance.
	 */
	return (float)hashtable.size / (float)hashtable.capacity;
}



#define HASH_INIT GpuHashTable GpuHashTable(1); /**< @brief Macro to initialize a GpuHashTable instance with a default size of 1. */
#define HASH_RESERVE(size) GpuHashTable.reshape(size); /**< @brief Macro to reshape the GpuHashTable to a specified size. */

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys) /**< @brief Macro to insert a batch of key-value pairs into the hash table. */
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys) /**< @brief Macro to retrieve a batch of values for given keys from the hash table. */

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor() /**< @brief Macro to get the current load factor of the hash table. */

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0 /**< @brief Sentinel value indicating an invalid or empty key in the hash table. */

/**
 * @brief Macro for error checking and termination.
 * @param assertion A boolean expression that, if true, indicates an error.
 * @param call_description A string describing the function call or operation being checked.
 *
 * This macro checks an assertion; if it's true, it prints an error message
 * including the file and line number, a system error description, and then
 * exits the program.
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
 * @brief An array of large prime numbers used in various hash functions.
 *
 * This list provides a selection of prime numbers, likely to be used as
 * moduli or multipliers in hashing algorithms to ensure better distribution
 * and reduce collisions. These are for CPU-side hashing, potentially for testing.
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
 * @brief Auxiliary hash function 1.
 * @param data The integer value to be hashed.
 * @param limit The upper limit for the hash result.
 * @return A hash value within the specified limit.
 *
 * This function uses a specific pair of prime numbers from `primeList`
 * for hash calculation, providing an alternative hashing mechanism.
 */
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
/**
 * @brief Auxiliary hash function 2.
 * @param data The integer value to be hashed.
 * @param limit The upper limit for the hash result.
 * @return A hash value within the specified limit.
 *
 * This function uses a specific pair of prime numbers from `primeList`
 * for hash calculation, providing another alternative hashing mechanism.
 */
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
/**
 * @brief Auxiliary hash function 3.
 * @param data The integer value to be hashed.
 * @param limit The upper limit for the hash result.
 * @return A hash value within the specified limit.
 *
 * This function uses a specific pair of prime numbers from `primeList`
 * for hash calculation, providing yet another alternative hashing mechanism.
 */
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

/**
 * @brief Represents a single node (entry) in the hash table.
 *
 * This struct holds a key-value pair. The `key` field is used to store
 * the identifier, and the `value` field stores its associated data.
 * A `key` of `0` typically denotes an empty or invalid slot.
 */
struct Node {
	int key;
	int value;
};

/**
 * @brief Device-side structure for the hash table.
 *
 * This struct encapsulates the essential metadata for the hash table on the GPU,
 * including its current number of elements (`size`), total capacity (`capacity`),
 * and a pointer to the array of `Node` elements (`nodes`) in device memory.
 */
struct HashTable {
	int capacity;
	int size;
	Node *nodes;
};

/**
 * @brief Host-side class for managing a GPU-accelerated hash table.
 *
 * This class provides the host-side interface to create, manage, and interact
 * with a hash table stored in GPU device memory. It encapsulates the `HashTable`
 * struct and orchestrates memory allocations/deallocations, data transfers
 * between host and device, and the launching of CUDA kernels for hash table
 * operations (insert, get, reshape).
 */
class GpuHashTable
{
	/// @brief Instance of the HashTable struct, representing the device-side hash table.
	HashTable hashtable;

	public:
		/**
		 * @brief Constructor for GpuHashTable.
		 * @param size The initial capacity of the hash table.
		 */
		GpuHashTable(int size);
		/**
		 * @brief Reshapes the hash table to a new size.
		 * @param sizeReshape The new size for the hash table.
		 */
		void reshape(int sizeReshape);
		
		/**
		 * @brief Inserts a batch of key-value pairs into the hash table.
		 * @param keys Array of keys to insert.
		 * @param values Array of values to insert.
		 * @param numKeys The number of key-value pairs in the batch.
		 * @return True if the batch insertion was successful, false otherwise.
		 */
		bool insertBatch(int *keys, int* values, int numKeys);
		/**
		 * @brief Retrieves a batch of values for given keys from the hash table.
		 * @param key Array of keys to retrieve values for.
		 * @param numItems The number of keys to retrieve.
		 * @return A pointer to an array of retrieved values on the host.
		 */
		int* getBatch(int* key, int numItems);
		
		/**
		 * @brief Calculates the current load factor of the hash table.
		 * @return The load factor as a float.
		 */
		float loadFactor();
		// void occupancy(); // Not implemented in this file, likely a placeholder for future functionality.
		// void print(string info); // Not implemented in this file, likely a placeholder for future functionality.
	
		/**
		 * @brief Destructor for GpuHashTable, frees GPU memory.
		 */
		~GpuHashTable();
};

#endif

