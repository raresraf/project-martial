
/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table leveraging CUDA for parallel key-value storage and retrieval.
 *
 * This module provides a high-performance hash table designed for NVIDIA GPUs,
 * offering functionalities for batch insertion, batch retrieval, and dynamic resizing.
 * It utilizes CUDA kernels and atomic operations to ensure thread-safe and efficient
 * data manipulation in a parallel computing environment.
 *
 * Domain: High-Performance Computing, Parallel Algorithms, Data Structures, CUDA.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"






/**
 * @brief CUDA kernel for parallel insertion of key-value pairs into the hash table.
 *
 * This kernel executes on the GPU, with each thread attempting to insert a single
 * key-value pair. It employs a linear probing strategy for collision resolution
 * and utilizes atomic operations to ensure thread safety during concurrent writes
 * to the hash table.
 *
 * Parameters:
 *   - keys: Pointer to an array of integer keys to be inserted.
 *   - values: Pointer to an array of integer values corresponding to the keys.
 *   - numKeys: Total number of key-value pairs to attempt to insert.
 *   - hmap: The hash map structure (device-side representation) where data will be stored.
 */
__global__ void thread_insert(int *keys, int *values, int numKeys, Hmap hmap) {

  	// Calculate a unique global index for each thread.
  	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  	// Pre-condition: Ensure the thread index is within the bounds of the number of keys to process.
  	if(i >= numKeys)
  		return;

  	// Pre-condition: Skip insertion if the key is invalid (KEY_INVALID is 0).
  	if( keys[i] == 0)
  		return;

  	// Compute the initial hash index for the current key.
  	int hashed_idx = hash1(keys[i], hmap.size);

  	/**
     * Block Logic: Implements linear probing for collision resolution.
     * Invariant: The loop continues until the key is successfully inserted or its value is updated.
     */
  	while(1) {

		/**
         * Block Logic: Attempt to insert the key if the slot is empty (key is 0).
         * This uses atomicCompareAndSwap (atomicCAS) to ensure only one thread
         * successfully writes to an empty slot, preventing race conditions.
         * If successful, the value is then atomically exchanged.
         */
		if( atomicCAS(&hmap.keys[hashed_idx], 0, keys[i]) == 0 ){

			atomicExch(&hmap.values[hashed_idx], values[i]);
			break;

		}

		/**
         * Block Logic: If the slot is not empty, check if the key already exists.
         * If the key matches an existing entry, atomically update its value.
         * This handles the case of inserting a duplicate key with a new value.
         */
		if( atomicCAS(&hmap.keys[hashed_idx], keys[i], keys[i]) == keys[i]){

			atomicExch(&hmap.values[hashed_idx], values[i]);
			break;

		}

    	// Move to the next slot (linear probing).
    	hashed_idx++;

    	// Wrap around to the beginning of the hash table if the end is reached.
     	hashed_idx %= hmap.size;
    }

}



/**
 * @brief CUDA kernel for parallel retrieval of values from the hash table.
 *
 * This kernel executes on the GPU, with each thread attempting to retrieve the value
 * associated with a given key. It employs a linear probing strategy to find the key
 * and uses atomic operations to ensure safe access to hash table elements.
 *
 * Parameters:
 *   - keys: Pointer to an array of integer keys to query.
 *   - numKeys: Total number of keys to query.
 *   - hmap: The hash map structure (device-side representation) to search within.
 *   - results: Pointer to an array where the retrieved values will be stored.
 */
__global__ void thread_get(int *keys, int numKeys, Hmap hmap, int *results) {

  	// Calculate a unique global index for each thread.
  	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

  	// Pre-condition: Ensure the thread index is within the bounds of the number of keys to process.
  	if(i >= numKeys)
  		return;
  	// Compute the initial hash index for the current key.
  	int hashed_idx = hash1(keys[i], hmap.size);

  	/**
     * Block Logic: Implements linear probing to search for the key.
     * Invariant: The loop continues until the key is found or the search wraps around
     * (implying the key is not present, though this implementation assumes key presence).
     */
  	while(1) {

    	/**
         * Block Logic: Atomically check if the current slot contains the target key.
         * If the key matches, atomically exchange the found value into the results array.
         * This handles concurrent reads and ensures the correct value is retrieved.
         */
    	if(atomicCAS(&hmap.keys[hashed_idx], keys[i], keys[i]) == keys[i]){
    		atomicExch(&results[i], hmap.values[hashed_idx]);
    		break;
    	}

    	// Move to the next slot (linear probing).
    	hashed_idx++;

    	// Wrap around to the beginning of the hash table if the end is reached.
     	hashed_idx %= hmap.size;

    }

}



/**
 * @brief Constructor for GpuHashTable.
 *
 * Initializes a new hash table on the GPU. This involves allocating device memory
 * for both keys and values arrays and setting all entries to 0 (KEY_INVALID for keys).
 *
 * Parameters:
 *   - size: The initial capacity of the hash table (number of buckets).
 */
GpuHashTable::GpuHashTable(int size) {

	hm.current_no_of_pairs = 0;
	hm.size = size;
	// Allocate memory for keys array on the GPU.
	cudaMalloc((void **) &hm.keys, size * sizeof(int));
	// Allocate memory for values array on the GPU.
	cudaMalloc((void **) &hm.values, size * sizeof(int));

	// Initialize all key entries to 0 (representing empty/invalid).
	cudaMemset(hm.keys, 0, size * sizeof(int));
	// Initialize all value entries to 0.
	cudaMemset(hm.values, 0, size * sizeof(int));
}


/**
 * @brief Destructor for GpuHashTable.
 *
 * Frees the device memory allocated for the keys and values arrays
 * to prevent memory leaks on the GPU.
 */
GpuHashTable::~GpuHashTable() {

	cudaFree(hm.keys);
	cudaFree(hm.values);
}



/**
 * @brief Resizes the hash table to a new capacity.
 *
 * This function creates a new hash table with `numBucketsReshape` buckets,
 * rehashes all existing key-value pairs from the current table into the new table,
 * and then replaces the old table with the new one. This is typically called
 * when the load factor exceeds a certain threshold.
 *
 * Parameters:
 *   - numBucketsReshape: The new desired capacity of the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {

	Hmap new_hm;

	new_hm.size = numBucketsReshape;

	// Allocate memory for the new keys array on the GPU.
	cudaMalloc((void **) &new_hm.keys, numBucketsReshape * sizeof(int));
	// Allocate memory for the new values array on the GPU.
	cudaMalloc((void **) &new_hm.values, numBucketsReshape * sizeof(int));

	// Initialize new keys and values arrays to 0.
	cudaMemset(new_hm.keys, 0, numBucketsReshape * sizeof(int));
	cudaMemset(new_hm.values, 0, numBucketsReshape * sizeof(int));

	// Determine the grid and block dimensions for the kernel launch.
	unsigned int no_of_threads_in_block = 256;
	unsigned int no_of_blocks = hm.size / no_of_threads_in_block;

	// Adjust block count if the size is not a perfect multiple of threads per block.
	if(hm.size % no_of_threads_in_block != 0)
		no_of_blocks ++;

	/**
     * Block Logic: Re-inserts all existing key-value pairs from the old hash table
     * into the newly allocated hash table using the `thread_insert` kernel.
     * The `hm.keys` and `hm.values` here refer to the *source* data.
     */
	thread_insert<<<no_of_blocks, no_of_threads_in_block>>>(hm.keys, hm.values, hm.size, new_hm);

	// Ensures all GPU operations (specifically the re-insertion) are complete before proceeding.
	cudaDeviceSynchronize();

	// Update the count of pairs in the new hash map.
	new_hm.current_no_of_pairs += hm.current_no_of_pairs;

	// Free the memory of the old hash table.
	cudaFree(hm.keys);

	// Free the memory of the old hash table.
	cudaFree(hm.values);

	// Replace the old hash map structure with the new one.
	hm = new_hm;


}


/**
 * @brief Inserts a batch of key-value pairs into the GPU hash table.
 *
 * This function prepares a batch of key-value pairs from host memory,
 * transfers them to device memory, checks the load factor to determine
 * if a reshape operation is necessary, launches the `thread_insert` kernel
 * to perform parallel insertion, and then cleans up device memory.
 *
 * Parameters:
 *   - keys: Host pointer to an array of integer keys to insert.
 *   - values: Host pointer to an array of integer values to insert.
 *   - numKeys: The number of key-value pairs in the batch.
 *
 * Returns:
 *   - true if the insertion batch operation was successful.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys, *device_values;

	// Define block size for CUDA kernel launch.
	unsigned int no_of_threads_in_block = 256;
	// Calculate the total size in bytes for the key/value arrays.
	unsigned int size = numKeys * sizeof(int);
	// Calculate the number of blocks needed based on the number of keys.
	unsigned int no_of_blocks = numKeys / no_of_threads_in_block;

	// Adjust block count if numKeys is not a perfect multiple of threads per block.
	if(numKeys % no_of_threads_in_block != 0)
		no_of_blocks++;

	// Allocate device memory for the batch of keys.
	cudaMalloc(&device_keys, size);
	// Allocate device memory for the batch of values.
	cudaMalloc(&device_values, size);

	/**
     * Block Logic: Checks if the current load factor of the hash table exceeds 0.9.
     * If it does, triggers a reshape operation to expand the hash table capacity
     * to accommodate new insertions and maintain performance.
     * Invariant: After this block, the hash table has sufficient capacity or has
     * been resized to a larger capacity.
     */
	if (float(hm.current_no_of_pairs + numKeys) / hm.size >= 0.9)
		reshape(int((hm.current_no_of_pairs + numKeys) / 0.8));

	// Copy the batch of keys from host to device memory.
	cudaMemcpy(device_keys, keys, size, cudaMemcpyHostToDevice);
	// Copy the batch of values from host to device memory.
	cudaMemcpy(device_values, values, size, cudaMemcpyHostToDevice);

	/**
     * Block Logic: Launches the `thread_insert` CUDA kernel to perform
     * parallel insertion of the batch of key-value pairs into the hash table.
     * Each thread attempts to insert one key-value pair.
     */
	thread_insert<<<no_of_blocks, no_of_threads_in_block>>>(device_keys, device_values, numKeys, hm);

	// Synchronizes the host with the device, ensuring all kernel operations are complete.
	cudaDeviceSynchronize();

	// Update the total count of key-value pairs in the hash table.
	hm.current_no_of_pairs += numKeys;

	// Free device memory allocated for the batch keys.
	cudaFree(device_keys);
	// Free device memory allocated for the batch values.
	cudaFree(device_values);


	return true;
}


/**
 * @brief Retrieves a batch of values for given keys from the GPU hash table.
 *
 * This function takes an array of keys from host memory, transfers them to
 * device memory, launches the `thread_get` kernel to perform parallel retrieval,
 * and then returns a pointer to the device array containing the results.
 *
 * Parameters:
 *   - keys: Host pointer to an array of integer keys to query.
 *   - numKeys: The number of keys in the batch.
 *
 * Returns:
 *   - A device pointer to an array containing the retrieved values. The caller
 *     is responsible for freeing this memory (e.g., using `cudaFree`).
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {

	int *device_keys;

	// Define block size for CUDA kernel launch.
	unsigned int no_of_threads_in_block = 256;
	// Calculate the total size in bytes for the key/value arrays.
	unsigned int size = numKeys * sizeof(int);
	// Calculate the number of blocks needed based on the number of keys.
	unsigned int no_of_blocks = numKeys / no_of_threads_in_block;

	// Adjust block count if numKeys is not a perfect multiple of threads per block.
	if(numKeys % no_of_threads_in_block != 0)
		no_of_blocks++;

	// Allocate device memory for the batch of keys.
	cudaMalloc(&device_keys, size);
	// Copy the batch of keys from host to device memory.
	cudaMemcpy(device_keys, keys, size, cudaMemcpyHostToDevice);


	int *results;
	// Allocate managed memory for results, accessible by both host and device.
	cudaMallocManaged(&results, size);

	/**
     * Block Logic: Launches the `thread_get` CUDA kernel to perform
     * parallel retrieval of values for the batch of keys.
     */
	thread_get<<<no_of_blocks, no_of_threads_in_block>>>(device_keys, numKeys, hm, results);

	// Synchronizes the host with the device, ensuring all kernel operations are complete.
	cudaDeviceSynchronize();




	// Free device memory allocated for the batch keys.
	cudaFree(device_keys);

	// Returns a pointer to the device array containing the results.
	return results;


}


/**
 * @brief Calculates the current load factor of the GPU hash table.
 *
 * The load factor is a measure of how full the hash table is, calculated as
 * the number of stored key-value pairs divided by the total number of buckets.
 * A higher load factor typically indicates more collisions and potentially
 * slower performance, prompting a reshape operation.
 *
 * Returns:
 *   - A float representing the current load factor. Returns 0 if the table size is 0 to avoid division by zero.
 */
float GpuHashTable::loadFactor() {
	// Pre-condition: Avoid division by zero if the hash map size is 0.
	if(hm.size == 0)
		return 0;
	// Calculate the load factor.
	return float(hm.current_no_of_pairs)/hm.size;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
/**
 * @brief Macro for initializing the GpuHashTable.
 *
 * This macro provides a convenient way to declare and initialize a
 * `GpuHashTable` instance with an initial size of 1.
 */
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

/**
 * @brief Macro for reserving space in the GpuHashTable.
 *
 * This macro calls the `reshape` method of the `GpuHashTable` instance
 * to expand its capacity to the specified `size`.
 */
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
/**
 * @brief Macro for inserting a batch of key-value pairs.
 *
 * This macro calls the `insertBatch` method of the `GpuHashTable` instance
 * to insert multiple key-value pairs in parallel.
 */
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)
/**
 * @brief Macro for retrieving a batch of values.
 *
 * This macro calls the `getBatch` method of the `GpuHashTable` instance
 * to retrieve multiple values in parallel for the given keys.
 */
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()
/**
 * @brief Macro for getting the current load factor of the GpuHashTable.
 *
 * This macro calls the `loadFactor` method to get an indication of
 * how full the hash table is.
 */

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

/**
 * @brief Defines KEY_INVALID as 0, which signifies an empty or unused hash table slot.
 */
#define	KEY_INVALID		0

/**
 * @brief A utility macro for error handling and program termination.
 *
 * This macro checks an assertion; if false, it prints an error message
 * to stderr, including the file and line number, a description of the
 * failed call, and then terminates the program with the current errno.
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
 * @brief A constant array of prime numbers used in hashing.
 *
 * This list provides a selection of large prime numbers suitable for
 * modulo operations in hash functions, helping to distribute hash values
 * more evenly and reduce collisions.
 */
__device__ const size_t primeList[] =
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
 * @brief Device-side hash function for mapping data to an index within the hash table limit.
 *
 * This function computes a hash index for a given integer `data` using a combination
 * of multiplication by a prime number from `primeList` and modulo operations.
 * The result is then taken modulo `limit` to fit within the hash table's bounds.
 *
 * Parameters:
 *   - data: The integer data to be hashed.
 *   - limit: The size of the hash table (number of buckets).
 *
 * Returns:
 *   - An integer hash index within the range [0, limit-1].
 */
__device__ int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}



/**
 * @brief Structure representing the internal state of a GPU hash map.
 *
 * This structure holds pointers to the device-side arrays for keys and values,
 * as well as metadata about the current state of the hash map, such as the
 * number of stored pairs and its total capacity.
 */
typedef struct hashmap {
	int *keys;
	int *values;
	unsigned int current_no_of_pairs;
	unsigned int size;
} Hmap;



/**
 * @brief Declaration of the GpuHashTable class.
 *
 * This class provides a host-side interface for managing a GPU-accelerated hash table.
 * It encapsulates the device-side hash map (`Hmap`) and provides methods for
 * initialization, resizing, batch operations (insert and get), and querying
 * the hash table's load factor.
 */
class GpuHashTable
{
	Hmap hm;
	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);

		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);

		float loadFactor();
		void occupancy();
		void print(string info);

		~GpuHashTable();
};

#endif

