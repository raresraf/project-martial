/**
 * @file gpu_hashtable.cu
 * @brief CUDA implementation of a GPU-resident hash table.
 *
 * @details This file provides a high-performance hash table designed to be entirely managed and
 * accessed on the GPU. It uses a 2-way set-associative structure combined with open addressing
 * to handle collisions. The key operations—insertion, retrieval, and resizing (reshape)—are
 * implemented as CUDA kernels for parallel execution.
 *
 * Algorithm:
 * - Hashing: A multiplicative hashing scheme using a pre-computed list of prime numbers.
 * - Collision Handling: The hash table consists of two parallel arrays (node[0] and node[1]).
 *   A key-value pair can be stored in either array at the hash location. This is a form of
 *   2-way cuckoo hashing. If both slots are occupied, this implementation uses linear probing
 *   (wrapping around the table) to find the next available slot.
 * - Concurrency: Atomic Compare-And-Swap (atomicCAS) operations are used during insertion
 *   to ensure thread-safe writes, preventing race conditions when multiple threads attempt
 *   to claim the same hash slot.
 * - Dynamic Resizing: The hash table automatically resizes (reshape) when the load factor
 *   exceeds a predefined threshold (OCCUPIED_FACTOR). This is done by launching a kernel
 *   that re-hashes all existing elements into a new, larger table.
 *
 * HPC & Parallelism:
 * - Memory Hierarchy: The primary hash table data (`hashmap.node`) resides in global GPU memory.
 *   The `copyList` of primes is stored in device constant memory, which provides faster read
 *   access for all threads in a warp.
 * - Thread Indexing: Kernels use the standard `blockIdx.x * blockDim.x + threadIdx.x` pattern
 *   to assign a unique data element (key-value pair) to each thread for batch operations.
 * - Synchronization: `cudaDeviceSynchronize()` is used on the host side to ensure that
 *   kernels complete before the host proceeds, which is crucial for correctness during
 *   reshape operations and before freeing memory.
 */
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief A large list of prime numbers stored in device (constant) memory.
 * Functional Utility: These primes are used as multipliers and moduli in the hashing
 * function (`myHash`) to ensure a good distribution of keys across the hash table.
 * Using constant memory is an optimization as it is cached and provides fast, uniform
 * access for all threads in a warp.
 */
__device__ const std::size_t copyList[] =
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
 * @brief Computes a hash value for a given integer key.
 * @param data The input key to hash.
 * @param limit The size of the hash table, used to scale the hash value.
 * @return An integer hash value in the range [0, limit - 1].
 * @note This is a device function, callable only from kernels or other device functions.
 */
__device__ int myHash(int data, int limit) {
	// Block Logic: Multiplicative hashing using large prime numbers to distribute keys.
	// The absolute value of data is used to handle negative keys.
	// The result is taken modulo the table size to map it to a valid bucket index.
	return ((long long) abs(data) * copyList[55]) % copyList[99] % limit;
}


/**
 * @brief Attempts to insert a key-value pair into the hash table using atomic operations.
 * @param value The value to be inserted.
 * @param newTable The hash table to insert into.
 * @param begin The starting index for the search (typically the initial hash location).
 * @param end The ending index for the search (typically the end of the table).
 * @param newKey The key to be inserted.
 * @return `true` if the insertion was successful, `false` otherwise.
 * @note Implements linear probing and uses atomicCAS for thread-safe insertion.
 */
__device__ bool found(int value, hashTable newTable, int begin, int end, int newKey) {

	int key;
	// Block Logic: Implements linear probing to find an empty slot.
	// Pre-condition: `begin` is the ideal hash slot. The loop scans from `begin` to `end`.
	// Invariant: The loop continues as long as no empty slot has been found and `i` is in bounds.
	for (int i = begin; i < end; i++) {
		// Inline: Atomically attempt to claim the slot in the first table array.
		// `atomicCAS` ensures that only one thread can successfully write its key
		// if multiple threads collide on the same slot. It returns the old value.
		key = atomicCAS(&newTable.node[0][i].key, KEY_INVALID, newKey);
		// If the original key was our key (already inserted) or the slot was empty, the insertion is successful.
		if (key == newKey || key == KEY_INVALID) {
			newTable.node[0][i].value = (uint32_t) value;
			return true;
		}
		else {
			// Inline: If the first slot was taken, try the second slot in the cuckoo-style structure.
			key = atomicCAS(&newTable.node[1][i].key, KEY_INVALID, newKey);
			// If this succeeds, store the value and return.
			if (key == newKey || key == KEY_INVALID) {
				newTable.node[1][i].value = (uint32_t) value;
				return true;
			}
		
		}		

	}
	// The key could not be inserted in the given range.
	return false;
}



/**
 * @brief Searches for a key within a specified range of the hash table.
 * @param Table The hash table to search in.
 * @param begin The starting index of the search.
 * @param end The ending index of the search.
 * @param key The key to search for.
 * @return The value associated with the key, or -1 if the key is not found.
 */
__device__ int searchKey(hashTable Table, int begin, int end, int key) {
	
	int value = -1;
	// Block Logic: Linear scan for the key within the given range [begin, end).
	// It checks both parallel arrays at each index.
	for (int i = begin; i < end; i++) {
		if (Table.node[0][i].key == key) {
				value = Table.node[0][i].value;
				return value;
		}
		else if (Table.node[1][i].key == key) {
				value = Table.node[1][i].value;
				return value;
		}
	}
	return value;
}


/**
 * @brief CUDA kernel to resize the hash table.
 * @param old_hash The original hash table.
 * @param new_hash The new, larger hash table.
 * @details Each thread is responsible for one bucket in the old hash table. It reads the
 *          key-value pairs from that bucket and re-inserts them into the new table.
 */
__global__ void reshape_global(hashTable old_hash, hashTable new_hash) {
	// Thread Indexing: Assign one thread per bucket of the old hash table.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Pre-condition: Ensure the thread index is within the bounds of the old table size.
	if (idx >= old_hash.size) 
		return;

	// Block Logic: Iterate through the two possible slots at the given bucket index.
	for (int index = 0; index <= 1 && old_hash.node[index][idx].key != KEY_INVALID; index++) {

		int newKey = old_hash.node[index][idx].key;

		// Re-hash the key for the new table size.
		int hash = myHash(newKey, new_hash.size);

		int value = old_hash.node[index][idx].value;

		// Attempt to insert the item, starting from its new hash location.
		bool done = found(value, new_hash, hash, new_hash.size, newKey);
		// If insertion fails (e.g., due to high contention or a full probe sequence),
		// wrap around and try to insert starting from the beginning of the table.
		if (!done)
			found(value, new_hash, 0, hash, newKey);
	
	}
}

/**
 * @brief CUDA kernel for batch insertion of key-value pairs.
 * @param keys Device pointer to an array of keys to insert.
 * @param values Device pointer to an array of values to insert.
 * @param numEntries The number of key-value pairs in the batch.
 * @param hashmap The target hash table.
 */
__global__ void insert_global(int *keys, int *values, int numEntries, hashTable hashmap) {
	// Thread Indexing: Assign one thread per key-value pair to be inserted.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numEntries)
		return;

	int newKey = keys[idx];

	// Calculate the initial hash position for the key.
	int hash = myHash(newKey, hashmap.size);
	bool done = false;

	// Block Logic: Tries to insert into one of the two slots.
	// This loop is not strictly necessary as `found` already checks both slots,
	// but it ensures two full search passes are attempted.
	for (int index = 0; index <= 1; index++) {

		if (!done)
			// First attempt: linear probe from the ideal hash location to the end.
			done = found(values[idx], hashmap, hash, hashmap.size, newKey);
		if (!done)
			// Second attempt: wrap around and probe from the beginning to the hash location.
			done = found(values[idx], hashmap, 0, hash, newKey);
	}
}

/**
 * @brief CUDA kernel for batch retrieval of values based on keys.
 * @param keys Device pointer to an array of keys to look up.
 * @param values Device pointer to an array where the found values will be stored.
 * @param numEntries The number of keys to look up.
 * @param hashmap The hash table to search in.
 */
__global__ void get_global(int *keys, int *values, int numEntries, hashTable hashmap) {
	// Thread Indexing: Assign one thread per key to be retrieved.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numEntries)
		return;

	int key = keys[idx];

	int hash = myHash(keys[idx], hashmap.size);

	// First search attempt: probe from the ideal hash location to the end of the table.
	values[idx] = searchKey(hashmap, hash, hashmap.size, key);
	// If not found, wrap around and search from the beginning to the hash location.
	if (values[idx] < 0)
		values[idx] = searchKey(hashmap, 0, hash, key);

}

/**
 * @brief Host-side constructor for the GpuHashTable class.
 * @param size The initial number of buckets in the hash table.
 */
GpuHashTable::GpuHashTable(int size) {

	added = 0;
	hashmap.size = size;
	hashmap.node[0] = NULL;
	hashmap.node[1] = NULL;

	// Block Logic: Allocate global GPU memory for the two parallel node arrays.
	DIE(cudaMalloc(&hashmap.node[0], size * sizeof(item)) != cudaSuccess, "Malloc Error");
	DIE(cudaMalloc(&hashmap.node[1], size * sizeof(item)) != cudaSuccess, "Malloc Error");
	
	// Block Logic: Initialize the allocated GPU memory to zero. This sets all keys to KEY_INVALID (0).
	DIE(cudaMemset(hashmap.node[0], 0, size * sizeof(item)) != cudaSuccess, "Memset Error");
	DIE(cudaMemset(hashmap.node[1], 0, size * sizeof(item)) != cudaSuccess, "Memset Error");
}

/**
 * @brief Host-side destructor for the GpuHashTable class.
 */
GpuHashTable::~GpuHashTable() {
	// Frees the GPU memory allocated for the hash table nodes.
	DIE(cudaFree(hashmap.node[0]) != cudaSuccess, "Free Error");
	DIE(cudaFree(hashmap.node[1]) != cudaSuccess, "Free Error");
}

/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The new number of buckets for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {

	hashTable hashmap_new;
	hashmap_new.size = numBucketsReshape;

	// Allocate memory for the new, larger hash table.
	DIE(cudaMalloc(&hashmap_new.node[0], numBucketsReshape * sizeof(item)) != cudaSuccess, "Malloc Error");
	DIE(cudaMalloc(&hashmap_new.node[1], numBucketsReshape * sizeof(item)) != cudaSuccess, "Malloc Error");

	DIE(cudaMemset(hashmap_new.node[0], 0, numBucketsReshape * sizeof(item)) != cudaSuccess, "Memset Error");
	DIE(cudaMemset(hashmap_new.node[1], 0, numBucketsReshape * sizeof(item)) != cudaSuccess, "Memset Error");

	// Determine the grid size for the reshape kernel launch.
	unsigned int numBlocks = hashmap.size / THREADS_PER_BLOCK;
	if (hashmap.size % THREADS_PER_BLOCK != 0) numBlocks++;

	cudaDeviceSynchronize();

	// Launch the kernel to re-hash elements from the old table to the new one.
	reshape_global<<>>(hashmap, hashmap_new);

	cudaDeviceSynchronize();

	// Free the memory of the old hash table.
	DIE(cudaFree(hashmap.node[0]) != cudaSuccess, "Free Error");
	DIE(cudaFree(hashmap.node[1]) != cudaSuccess, "Free Error");

	// Point the current hashmap to the new one.
	hashmap = hashmap_new;
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys Host pointer to an array of keys.
 * @param values Host pointer to an array of values.
 * @param numKeys The number of pairs to insert.
 * @return `true` on successful insertion.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *deviceKeys, *deviceValues;

	// Pre-condition: Check if the load factor will exceed the threshold after insertion.
	// If so, trigger a reshape to increase the table's capacity first.
	if (float (added + numKeys)/hashmap.size > OCCUPIED_FACTOR)
		reshape(added + numKeys);

	// Allocate temporary device memory for the batch of keys and values.
	DIE(cudaMalloc(&deviceKeys, numKeys * sizeof(int)) != cudaSuccess, "Malloc Error");
	DIE(cudaMalloc(&deviceValues, numKeys * sizeof(int)) != cudaSuccess, "Malloc Error");

	// Copy the keys and values from host memory to device memory.
	DIE(cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess, "Memcpy Error");
	DIE(cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess, "Memcpy Error");

	// Determine the grid size for the insertion kernel.
	unsigned int numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK != 0) numBlocks++;

	cudaDeviceSynchronize();

	// Launch the batch insertion kernel.
	insert_global<<>>(deviceKeys, deviceValues, numKeys, hashmap);

	cudaDeviceSynchronize();

	// Update the count of added elements.
	added += numKeys;

	// Free the temporary device memory.
	DIE(cudaFree(deviceKeys) != cudaSuccess, "Free Error");
	DIE(cudaFree(deviceValues) != cudaSuccess, "Free Error");

	return true;
}


/**
 * @brief Retrieves a batch of values corresponding to a batch of keys.
 * @param keys Host pointer to an array of keys to look up.
 * @param numKeys The number of keys in the batch.
 * @return A host pointer to an array of values. The value for a key not found will be -1.
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
	int *deviceKeys;
	int *values;

	// Allocate temporary device memory for keys and managed memory for values.
	// Managed memory (`cudaMallocManaged`) is accessible from both host and device.
	DIE(cudaMalloc(&deviceKeys, numKeys * sizeof(int)) != cudaSuccess, "Malloc Error");
	DIE(cudaMallocManaged(&values, numKeys * sizeof(int)) != cudaSuccess, "Malloc Error");

	// Copy keys from host to device.
	DIE(cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess, "Memcpy Error");

	// Determine grid size for the get kernel.
	unsigned int numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK != 0) numBlocks++;

	cudaDeviceSynchronize();

	// Launch the batch retrieval kernel.
	get_global<<>>(deviceKeys, values, numKeys, hashmap);

	cudaDeviceSynchronize();

	// Free the temporary device memory for keys.
	DIE(cudaFree(deviceKeys) != cudaSuccess, "Free Error");

	// Return the host-accessible pointer to the results.
	return values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float (added elements / total capacity).
 */
float GpuHashTable::loadFactor() {
	if (hashmap.size)
		return float(added) / hashmap.size;

	return -1;	
}


// C-style macros for interfacing with the hash table, likely for a specific test harness.
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

// Includes a test file, which suggests this is part of a larger project with a test suite.
#include "test_map.cpp"
