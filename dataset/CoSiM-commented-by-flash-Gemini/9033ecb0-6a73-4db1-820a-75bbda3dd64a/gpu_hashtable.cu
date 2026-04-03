

/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA, supporting insertion, retrieval, and dynamic resizing.
 *
 * This file contains the CUDA device functions and global kernels for parallel hash table operations,
 * as well as the host-side GpuHashTable class methods to manage memory and orchestrate kernel launches.
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

#define FIRST_PRIME 13169977 /**< @brief First prime number for hash calculation. */
#define SECOND_PRIME 5351951779 /**< @brief Second prime number for hash calculation, larger than FIRST_PRIME. */


/**
 * @brief Device-side hash function for mapping data to a hash table index.
 * @param data The integer value to be hashed.
 * @param limit The size of the hash table (upper bound for the hash index).
 * @return An integer representing the hash value within the specified limit.
 *
 * This function uses a multiplicative hashing approach with two large prime numbers
 * to reduce collisions and distribute keys evenly across the hash table.
 * It's executed on the GPU device.
 */
__device__ int device_hash(int data, int limit)
{
	return ((long long)abs(data) * FIRST_PRIME) % SECOND_PRIME % limit;
}

/**
 * @brief CUDA kernel for parallel insertion of key-value pairs into the hash table.
 * @param keys Pointer to an array of keys to insert.
 * @param values Pointer to an array of values corresponding to the keys.
 * @param numKeys The total number of key-value pairs to insert.
 * @param hash_table Pointer to the hash table on the device.
 * @param size The current size of the hash table.
 *
 * Each CUDA thread processes one key-value pair. It calculates an initial hash
 * and then attempts to insert the pair using linear probing with `atomicCAS`
 * to handle collisions and ensure atomicity. If a key already exists or a slot
 * is invalid, the value is updated.
 *
 * Block Logic:
 *   - Thread Mapping: `idx` uniquely identifies the key-value pair processed by each thread.
 *   - Collision Resolution: Implements linear probing, wrapping around the table if necessary.
 *   - Concurrency Control: Uses `atomicCAS` (Compare-And-Swap) to ensure that
 *     only one thread successfully writes to a slot if multiple threads
 *     attempt to write to the same slot simultaneously, or to insert into an
 *     empty slot (KEY_INVALID).
 */
__global__ void kernel_insert(int *keys, int *values, int numKeys,
			      entry *hash_table, int size)
{
	// Inline: Calculate a unique global index for each thread.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, hash_value, old_value;

	// Pre-condition: Ensure the thread's index is within the bounds of the number of keys.
	if (idx >= numKeys)
		return;

	// Block Logic: Calculate the initial hash value for the current key.
	hash_value = device_hash(keys[idx], size);

	// Block Logic: Linear probing starting from the calculated hash_value to 'size'.
	// Invariant: Attempts to find an empty slot or a slot with the same key to update.
	for (i = hash_value; i < size; i++) {
		// Inline: Atomically attempt to insert the key or retrieve the existing key.
		old_value =
			atomicCAS(&hash_table[i].key, KEY_INVALID, keys[idx]);
		// Post-condition: If the slot was empty or contained the same key, update the value and return.
		if (old_value == keys[idx] || old_value == KEY_INVALID) {
			hash_table[i].value = values[idx];
			return;
		}
	}
	// Block Logic: If no slot was found in the first loop, continue linear probing
	// from the beginning of the table up to the initial hash_value (wrap-around).
	for (i = 0; i < hash_value; i++) {
		old_value =
			atomicCAS(&hash_table[i].key, KEY_INVALID, keys[idx]);
		// Post-condition: If the slot was empty or contained the same key, update the value and return.
		if (old_value == keys[idx] || old_value == KEY_INVALID) {
			hash_table[i].value = values[idx];
			return;
		}
	}
}

/**
 * @brief CUDA kernel for parallel retrieval of values from the hash table.
 * @param keys Pointer to an array of keys to search for.
 * @param values Pointer to an array where the retrieved values will be stored.
 * @param numKeys The total number of keys to search for.
 * @param hash_table Pointer to the hash table on the device.
 * @param size The current size of the hash table.
 *
 * Each CUDA thread processes one key. It calculates an initial hash
 * and then searches for the key using linear probing. If the key is found,
 * its corresponding value is written to the `values` array. If not found,
 * the corresponding entry in `values` remains unchanged (or default initialized).
 *
 * Block Logic:
 *   - Thread Mapping: `idx` uniquely identifies the key searched by each thread.
 *   - Collision Resolution: Implements linear probing, wrapping around the table if necessary,
 *     to locate the target key.
 */
__global__ void kernel_get(int *keys, int *values, int numKeys,
			   entry *hash_table, int size)
{
	// Inline: Calculate a unique global index for each thread.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, hash_value;

	// Pre-condition: Ensure the thread's index is within the bounds of the number of keys.
	if (idx >= numKeys)
		return;

	// Block Logic: Calculate the initial hash value for the current key.
	hash_value = device_hash(keys[idx], size);

	// Block Logic: Linear probing starting from the calculated hash_value to 'size'.
	// Invariant: Searches for a slot containing the target key.
	for (i = hash_value; i < size; i++) {
		// Pre-condition: Check if the current slot's key matches the target key.
		if (hash_table[i].key == keys[idx]) {
			values[idx] = hash_table[i].value;
			return;
		}
	}

	// Block Logic: If not found in the first loop, continue linear probing
	// from the beginning of the table up to the initial hash_value (wrap-around).
	for (i = 0; i < hash_value; i++) {
		// Pre-condition: Check if the current slot's key matches the target key.
		if (hash_table[i].key == keys[idx]) {
			values[idx] = hash_table[i].value;
			return;
		}
	}
}

/**
 * @brief CUDA kernel for reshaping (resizing) the hash table.
 * @param hash_table Pointer to the old hash table on the device.
 * @param new_hash_table Pointer to the new, larger hash table on the device.
 * @param old_size The size of the old hash table.
 * @param new_size The size of the new hash table.
 *
 * Each CUDA thread processes one entry from the old hash table. If the entry
 * is valid (not KEY_INVALID), it calculates a new hash value for it based on
 * the `new_size` and attempts to insert it into the `new_hash_table` using
 * linear probing and `atomicCAS` to ensure thread-safe insertion into the
 * new table.
 *
 * Block Logic:
 *   - Thread Mapping: `idx` uniquely identifies an entry in the old hash table
 *     to be rehashed by each thread.
 *   - Re-hashing: Recalculates the hash for valid entries using the new table size.
 *   - Collision Resolution: Implements linear probing in the `new_hash_table`,
 *     wrapping around if necessary.
 *   - Concurrency Control: Uses `atomicCAS` to ensure thread-safe insertion
 *     of elements from the old table into the new table, preventing race conditions.
 */
__global__ void kernel_reshape(entry *hash_table, entry *new_hash_table,
			       int old_size, int new_size)
{
	// Inline: Calculate a unique global index for each thread.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, new_hash, old_value;

	// Pre-condition: Ensure the thread's index is within the bounds of the old table size
	// and that the entry at `idx` in the old table is valid (not KEY_INVALID).
	if (idx >= old_size || hash_table[idx].key == KEY_INVALID)
		return;

	// Block Logic: Calculate the new hash value for the key based on the new table size.
	new_hash = device_hash(hash_table[idx].key, new_size);

	// Block Logic: Linear probing starting from the new_hash to new_size in the new table.
	// Invariant: Attempts to find an empty slot in the new hash table for the current key.
	for (i = new_hash; i < new_size; i++) {
		// Inline: Atomically attempt to insert the key into an empty slot in the new table.
		old_value = atomicCAS(&new_hash_table[i].key, KEY_INVALID,
				      hash_table[idx].key);
		// Post-condition: If the slot was empty, place the value and return.
		if (old_value == KEY_INVALID) {
			new_hash_table[i].value = hash_table[idx].value;
			return;
		}
	}
	// Block Logic: If no slot was found in the first loop, continue linear probing
	// from the beginning of the new table up to new_hash (wrap-around).
	for (i = 0; i < new_hash; i++) {
		old_value = atomicCAS(&new_hash_table[i].key, KEY_INVALID,
				      hash_table[idx].key);
		// Post-condition: If the slot was empty, place the value and return.
		if (old_value == KEY_INVALID) {
			new_hash_table[i].value = hash_table[idx].value;
			return;
		}
	}
}


GpuHashTable::GpuHashTable(int size)
{
	cudaError_t err;

	/**
	 * @brief Initializes a new instance of the GpuHashTable class.
	 * @param size The initial size (number of entries) for the hash table.
	 *
	 * This constructor allocates memory for the hash table on the GPU and
	 * initializes all keys to `KEY_INVALID` to mark them as empty.
	 * It also sets the initial number of inserted pairs to zero and stores
	 * the table size.
	 */
	this->numberInsertedPairs = 0;
	this->size = size;

	// Block Logic: Allocate device memory for the hash table.
	err = cudaMalloc(&this->hash_table, size * sizeof(entry));
	DIE(err != cudaSuccess, "cudaMalloc init");

	// Block Logic: Initialize all allocated device memory to 0 (KEY_INVALID).
	err = cudaMemset(this->hash_table, 0, size * sizeof(entry));
	DIE(err != cudaSuccess, "cudaMemset init");
}

/**
 * @brief Destroys the GpuHashTable instance, freeing allocated GPU memory.
 *
 * This destructor deallocates the device memory previously allocated for the hash table.
 */
GpuHashTable::~GpuHashTable()
{
	cudaFree(this->hash_table);
}


void GpuHashTable::reshape(int numBucketsReshape)
{
	entry *new_hash_table;
	cudaError_t err;
	unsigned int numBlocks;
	int old_size = this->size;

	/**
	 * @brief Reshapes the hash table to a new size.
	 * @param numBucketsReshape The new desired size for the hash table.
	 *
	 * This method reallocates a new hash table on the GPU with the specified
	 * `numBucketsReshape`. It then launches the `kernel_reshape` to transfer
	 * and re-insert all existing valid key-value pairs from the old table
	 * to the new one. Finally, it frees the old table's memory and updates
	 * the internal pointer to the new table.
	 */
	this->size = numBucketsReshape;

	// Block Logic: Allocate device memory for the new hash table.
	err = cudaMalloc(&new_hash_table, numBucketsReshape * sizeof(entry));
	DIE(err != cudaSuccess, "cudaMalloc reshape");

	// Block Logic: Initialize all allocated device memory for the new table to 0 (KEY_INVALID).
	err = cudaMemset(new_hash_table, 0, numBucketsReshape * sizeof(entry));
	DIE(err != cudaSuccess, "cudaMemset reshape");

	// Block Logic: Calculate the number of blocks needed for the kernel launch.
	// Invariant: Ensures enough blocks are launched to cover all elements in the old hash table.
	numBlocks = numBucketsReshape / THREADS_PER_BLOCK;
	if (numBucketsReshape % THREADS_PER_BLOCK)
		numBlocks++;

	// Block Logic: Launch the kernel to rehash and transfer elements from the old table to the new.
	kernel_reshape<<<numBlocks, THREADS_PER_BLOCK>>>(
		this->hash_table, new_hash_table, old_size, numBucketsReshape);

	// Block Logic: Synchronize the device to ensure all kernel operations complete before proceeding.
	cudaDeviceSynchronize();

	// Block Logic: Free the memory of the old hash table.
	err = cudaFree(this->hash_table);
	DIE(err != cudaSuccess, "cudaFree reshape");

	this->hash_table = new_hash_table;
}


bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys)
{
	int *device_keys;
	int *device_values;
	size_t total_size = numKeys * sizeof(int);
	unsigned int numBlocks;
	cudaError_t err;

	/**
	 * @brief Inserts a batch of key-value pairs into the hash table on the GPU.
	 * @param keys Pointer to an array of keys on the host.
	 * @param values Pointer to an array of values on the host.
	 * @param numKeys The number of key-value pairs in the batch.
	 * @return True if the insertion is successful, false otherwise.
	 *
	 * This method first allocates temporary device memory for the input keys and values,
	 * then checks if the hash table needs to be reshaped based on load factors.
	 * It copies the data from host to device, launches the `kernel_insert` to perform
	 * the parallel insertions, synchronizes the device, and finally frees the temporary
	 * device memory.
	 */
	// Block Logic: Allocate temporary device memory for keys.
	err = cudaMalloc(&device_keys, total_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc device_keys");
		return false;
	}
	// Block Logic: Allocate temporary device memory for values.
	err = cudaMalloc(&device_values, total_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc device_values");
		return false;
	}

	// Block Logic: Check if the current load factor exceeds the maximum threshold,
	// triggering a reshape operation to a larger table if necessary.
	if (float(numberInsertedPairs + numKeys) / this->size >=
	    MAXIMUM_LOAD_FACTOR)
		reshape(int((numberInsertedPairs + numKeys) /
			    MINIMUM_LOAD_FACTOR));

	// Block Logic: Copy keys from host memory to device memory.
	err = cudaMemcpy(device_keys, keys, total_size, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy device_keys");
	// Block Logic: Copy values from host memory to device memory.
	err = cudaMemcpy(device_values, values, total_size,
			 cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy device_values");

	// Block Logic: Calculate the number of blocks needed for the kernel launch.
	// Invariant: Ensures enough blocks are launched to cover all key-value pairs in the batch.
	numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK)
		numBlocks++;

	// Block Logic: Launch the kernel to perform parallel insertions.
	kernel_insert<<<numBlocks, THREADS_PER_BLOCK>>>(
		device_keys, device_values, numKeys, this->hash_table,
		this->size);

	// Block Logic: Synchronize the device to ensure all kernel operations complete before proceeding.
	cudaDeviceSynchronize();

	// Block Logic: Update the count of inserted pairs.
	numberInsertedPairs += numKeys;

	// Block Logic: Free temporary device memory for keys.
	err = cudaFree(device_keys);
	DIE(err != cudaSuccess, "cudaFree device_keys");

	// Block Logic: Free temporary device memory for values.
	err = cudaFree(device_values);
	DIE(err != cudaSuccess, "cudaFree device_values");

	return true;
}


int *GpuHashTable::getBatch(int *keys, int numKeys)
{
	int *device_keys;
	int *values, *device_values;
	size_t total_size = numKeys * sizeof(int);
	unsigned int numBlocks;
	cudaError_t err;

	/**
	 * @brief Retrieves a batch of values corresponding to a batch of keys from the GPU hash table.
	 * @param keys Pointer to an array of keys on the host to search for.
	 * @param numKeys The number of keys to search for.
	 * @return A pointer to an array of retrieved values on the host. Returns nullptr on error.
	 *
	 * This method allocates temporary device memory for keys and values, copies keys
	 * from host to device, launches the `kernel_get` to perform parallel lookups,
	 * synchronizes the device, and then copies the results (values) back to host memory.
	 * Finally, it frees temporary device memory.
	 */
	// Block Logic: Allocate temporary device memory for keys.
	err = cudaMalloc(&device_keys, total_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc device_keys");
		return nullptr;
	}
	// Block Logic: Allocate temporary device memory for values (to be populated by the kernel).
	err = cudaMalloc(&device_values, total_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc values");
		return nullptr;
	}

	// Block Logic: Initialize device_values to 0. This is important to ensure that if a key is not found,
	// the corresponding value in the result array is consistently 0.
	err = cudaMemset(device_values, 0, total_size);
	DIE(err != cudaSuccess, "cudaMemset getbatch");

	// Block Logic: Copy keys from host memory to device memory.
	err = cudaMemcpy(device_keys, keys, total_size, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy device_keys");

	// Block Logic: Allocate host memory for the retrieved values.
	values = (int *)malloc(total_size);
	DIE(values == NULL, "malloc values");

	// Block Logic: Calculate the number of blocks needed for the kernel launch.
	// Invariant: Ensures enough blocks are launched to cover all keys in the batch.
	numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK)
		numBlocks++;

	// Block Logic: Launch the kernel to perform parallel value retrievals.
	kernel_get<<<numBlocks, THREADS_PER_BLOCK>>>(device_keys,
						       device_values, numKeys,
						       this->hash_table,
						       this->size);

	// Block Logic: Synchronize the device to ensure all kernel operations complete before proceeding.
	cudaDeviceSynchronize();

	// Block Logic: Copy the retrieved values from device memory to host memory.
	err = cudaMemcpy(values, device_values, total_size,
			 cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, "cudaMemcpy values");

	// Block Logic: Free temporary device memory for keys.
	err = cudaFree(device_keys);
	DIE(err != cudaSuccess, "cudaFree device_keys");

	// Block Logic: Free temporary device memory for values.
	err = cudaFree(device_values);
	DIE(err != cudaSuccess, "cudaFree device_values");

	return values;
}


float GpuHashTable::loadFactor()
{
	/**
	 * @brief Calculates the current load factor of the hash table.
	 * @return The current load factor as a float. Returns 0 if the table size is 0.
	 *
	 * The load factor is defined as the ratio of the number of inserted pairs
	 * to the total size of the hash table. This metric indicates how full
	 * the hash table is and is used to determine when a reshape operation
	 * is necessary to maintain performance.
	 */
	// Pre-condition: Avoid division by zero if the hash table size is 0.
	if (this->size == 0)
		return 0;
	else
		return (float(numberInsertedPairs) / this->size);
}



#define HASH_INIT GpuHashTable GpuHashTable(1); /**< @brief Macro to initialize a GpuHashTable instance with a size of 1. */
#define HASH_RESERVE(size) GpuHashTable.reshape(size); /**< @brief Macro to reshape the GpuHashTable to a specified size. */

#define HASH_BATCH_INSERT(keys, values, numKeys)                               \
	GpuHashTable.insertBatch(keys, values, numKeys) /**< @brief Macro to insert a batch of key-value pairs. */
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys) /**< @brief Macro to retrieve a batch of values for given keys. */

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor() /**< @brief Macro to get the current load factor of the hash table. */

#include "test_map.cpp"

#ifndef _HASHCPU_
#define _HASHCPU_

#include 
#include 

using namespace std;


#define	KEY_INVALID			0 /**< @brief Sentinel value indicating an invalid or empty key in the hash table. */
#define THREADS_PER_BLOCK 	1024 /**< @brief Number of threads per block for CUDA kernel launches. */
#define MINIMUM_LOAD_FACTOR	0.8 /**< @brief Minimum load factor threshold for hash table resizing. */
#define MAXIMUM_LOAD_FACTOR	0.9 /**< @brief Maximum load factor threshold for hash table resizing. */

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
 * and reduce collisions.
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
 * This function uses a different pair of prime numbers from `primeList`
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
 * This function uses a different pair of prime numbers from `primeList`
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
 * This function uses a different pair of prime numbers from `primeList`
 * for hash calculation, providing yet another alternative hashing mechanism.
 */
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}


/**
 * @brief Represents a single entry in the hash table.
 *
 * This struct holds a key-value pair. The `key` field is used to store
 * the identifier, and the `value` field stores its associated data.
 * `KEY_INVALID` is a special value for `key` to denote an empty slot.
 */
typedef struct entry {
	int key;
	int value;
} entry;

/**
 * @brief Host-side class for managing a GPU-accelerated hash table.
 *
 * This class provides the host-side interface to create, manage, and interact
 * with a hash table stored in GPU device memory. It orchestrates memory
 * allocations/deallocations, data transfers between host and device, and
 * the launching of CUDA kernels for hash table operations (insert, get, reshape).
 */
class GpuHashTable
{
	public:
		/// @brief The number of currently inserted key-value pairs in the hash table.
		int numberInsertedPairs;
		/// @brief The current size (number of buckets) of the hash table.
		int size;
		/// @brief Pointer to the hash table data structure in GPU device memory.
		entry *hash_table;

		/**
		 * @brief Constructor for GpuHashTable.
		 * @param size The initial size of the hash table.
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
		 * @return A pointer to an array of retrieved values.
		 */
		int* getBatch(int* key, int numItems);
		
		/**
		 * @brief Calculates the current load factor of the hash table.
		 * @return The load factor as a float.
		 */
		float loadFactor();
		// void occupancy(); // Not implemented in this file
		// void print(string info); // Not implemented in this file
	
		/**
		 * @brief Destructor for GpuHashTable, frees GPU memory.
		 */
		~GpuHashTable();
};

#endif
