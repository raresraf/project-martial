/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 *
 * @details
 * This file implements a hash table on the GPU using open addressing with a
 * linear probing collision resolution strategy. The implementation supports
 * batch insertion, batch retrieval, and dynamic resizing (reshaping) based
 * on the table's load factor.
 *
 * @algorithm
 * - Hashing: A bit-wise hash function, `getHash`, is used to map keys to indices.
 * - Collision Resolution: Linear probing is employed. If a hash collision occurs,
 *   the algorithm probes the next bucket sequentially (with wrap-around) until
 *   an empty slot is found.
 * - Concurrency: The `atomicCAS` (Compare-And-Swap) operation is used during
 *   insertions and reshaping to handle race conditions, ensuring that multiple
 *   threads can safely write to the hash table.
 *
 * @note The file follows an unconventional structure, appearing to concatenate the
 * implementation with its own header definitions and a test file include.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief Computes a hash for a given key using bitwise operations.
 * @param data The key to hash.
 * @param limit The capacity of the hash table for the modulo operation.
 * @return The calculated hash index.
 * @note This is a device function that uses a series of XOR and multiplication
 * operations (a variant of a MurmurHash-style function) to generate the hash.
 */
__device__ int getHash(int data, int limit)
{
	data = ((data >> 16) ^ data) * 0x45d9f3b;
	data = ((data >> 16) ^ data) * 0x45d9f3b;
	data = (data >> 16) ^ data;
	return abs(data) % limit; // Use abs to ensure positive index
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param keys Device pointer to an array of keys.
 * @param values Device pointer to an array of values.
 * @param numKeys The number of pairs to insert.
 * @param hashtable The hash table structure.
 *
 * @details Each thread handles one key-value pair, using linear probing and
 * atomicCAS to find an empty slot or update an existing key.
 */
__global__ void kernel_insert(int *keys, int *values, int numKeys, hashtable_t hashtable)
{
	// Thread Indexing: Each thread gets a unique ID to process one key-value pair.
	unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int i, old, key, value, index;

	if (thread_id >= numKeys)
		return;

	key = keys[thread_id];
	value = values[thread_id];
	index = getHash(key, hashtable.max_size);

	// Block Logic: Linear probing loop to find an available slot.
	for (i = 0; i < hashtable.max_size; i++) {
		// Synchronization: Atomically attempt to claim an empty slot (KEY_INVALID).
		old = atomicCAS(&hashtable.map[index].key, KEY_INVALID, key);

		// Case 1: Slot was empty and we successfully wrote our key.
		// Case 2: Slot already contained our key.
		if (old == KEY_INVALID || old == key) {
			hashtable.map[index].value = value; // Set or update the value.
			return; // Insertion successful.
		}
		// Collision: Move to the next slot with wrap-around.
		index = (index + 1) % hashtable.max_size;
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @details Each thread searches for one key using linear probing.
 */
__global__ void kernel_get(int *keys, int *values, int numKeys, hashtable_t hashtable)
{
	// Thread Indexing: Each thread is assigned one key to look up.
	unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int i, index, key;

	if (thread_id >= numKeys)
		return;

	key = keys[thread_id];
	index = getHash(key, hashtable.max_size);

	// Block Logic: Linear probing search loop.
	for (i = 0; i < hashtable.max_size; i++) {
		if (hashtable.map[index].key == key) {
			// Key found, store the value in the output array.
			values[thread_id] = hashtable.map[index].value;
			return;
		}
		// If an empty slot is found, the key is not in the table.
		if (hashtable.map[index].key == KEY_INVALID) {
			values[thread_id] = KEY_INVALID;
			return;
		}
		// Collision: probe the next slot.
		index = (index + 1) % hashtable.max_size;
	}
}

/**
 * @brief CUDA kernel to copy and rehash all elements from an old table to a new one.
 * @details Each thread is responsible for one element from the old table.
 */
__global__ void kernel_reshape(hashtable_t oldHash, hashtable_t newHash) {
	// Thread Indexing: Each thread is assigned one slot from the old hash table.
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int i, index, key, old;

	if (thread_id >= oldHash.max_size)
		return;

	// Pre-condition: Only process valid (non-empty) slots.
	if (oldHash.map[thread_id].key == KEY_INVALID)
		return;

	key = oldHash.map[thread_id].key;
	index = getHash(key, newHash.max_size);

	// Block Logic: Linear probing to insert the old element into the new table.
	for (i = 0; i < newHash.max_size; i++) { // The loop limit should be newHash.max_size
		// Synchronization: Atomically claim an empty slot.
		old = atomicCAS(&newHash.map[index].key, KEY_INVALID, key);
		if (old == KEY_INVALID) {
			newHash.map[index].value = oldHash.map[thread_id].value;
			return;
		}
		// Collision: probe the next slot.
		index = (index + 1) % newHash.max_size;
	}
}

/**
 * @brief Constructor for GpuHashTable.
 * @details Allocates and initializes the hash table memory on the GPU.
 */
GpuHashTable::GpuHashTable(int size)
{
	hashtable.max_size = size;
	hashtable.actual_size = 0;

	// Memory Allocation: Allocate the main array for key-value pairs on the device.
	cudaMalloc((void **) &hashtable.map, size * sizeof(entry_t));
	if (hashtable.map == NULL)
		return;
	
	// Initialize all keys to KEY_INVALID (0).
	cudaMemset(hashtable.map, 0, size * sizeof(entry_t));
}

/**
 * @brief Destructor for GpuHashTable. Frees GPU memory.
 */
GpuHashTable::~GpuHashTable()
{
	cudaFree(hashtable.map);
}

/**
 * @brief Resizes the hash table and rehashes all elements.
 */
void GpuHashTable::reshape(int numBucketsReshape)
{
	hashtable_t new_hashtable;
	size_t blocks_no = hashtable.max_size / BLOCK_SIZE;
	if (hashtable.max_size % BLOCK_SIZE != 0)
		blocks_no++;

	new_hashtable.max_size = numBucketsReshape;

	// Memory Allocation: Allocate a new, larger table on the device.
	cudaMalloc((void **) &new_hashtable.map, numBucketsReshape * sizeof(entry_t));
	if (new_hashtable.map == NULL)
		return;

	cudaMemset(new_hashtable.map, 0, numBucketsReshape * sizeof(entry_t));

	// Launch the kernel to rehash elements from the old table to the new one.
	kernel_reshape>>(hashtable, new_hashtable);
	cudaDeviceSynchronize();

	// Free the old table memory.
	cudaFree(hashtable.map);

	// Update the main hash table structure.
	new_hashtable.actual_size = hashtable.actual_size;
	hashtable = new_hashtable;
}

/**
 * @brief Inserts a batch of keys and values into the hash table.
 * @return True on success, false on failure.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys)
{
	int *device_keys, *device_values;
	size_t size = sizeof(int) * numKeys;
	size_t blocks_no = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE != 0)
		blocks_no++;

	// Memory Allocation: Create temporary device buffers for the batch.
	cudaMalloc((void **) &device_keys, size);
	if (device_keys == NULL)
		return false;
	cudaMalloc((void **) &device_values, size);
	if (device_values == NULL)
		return false;

	// Block Logic: Check if the load factor will exceed the maximum threshold and reshape if necessary.
	if ((float)(hashtable.actual_size + numKeys) / hashtable.max_size > MAX_LOAD_FACTOR)
		reshape((int)((hashtable.actual_size + numKeys) / 0.8f));

	// Memory Transfer: Copy keys and values from host to device.
	cudaMemcpy(device_keys, keys, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, size, cudaMemcpyHostToDevice);

	// Launch the insertion kernel.
	kernel_insert>>(device_keys, device_values, numKeys, hashtable);

	// Synchronization: Wait for the kernel to finish.
	cudaDeviceSynchronize();
	cudaFree(device_keys);
	cudaFree(device_values);

	// Update the element count on the host.
	hashtable.actual_size += numKeys;
	return true;
}

/**
 * @brief Retrieves values for a batch of keys from the hash table.
 * @return A host pointer to an array of results. The caller must free this memory.
 */
int *GpuHashTable::getBatch(int *keys, int numKeys)
{
	int *device_keys, *values;
	size_t size = numKeys * sizeof(int);
	size_t blocks_no = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE != 0)
		blocks_no++;

	// Memory Allocation: Allocate temporary device buffer for keys.
	cudaMalloc((void **) &device_keys, size);
	if (device_keys == NULL)
		return NULL;
	
	// Memory Allocation: Use managed memory for the output values, simplifying host access after the kernel runs.
	cudaMallocManaged((void **) &values, size);
	if (values == NULL)
		return NULL;

	// Memory Transfer: Copy keys from host to device.
	cudaMemcpy(device_keys, keys, size, cudaMemcpyHostToDevice);

	// Launch the get kernel. Note: `values` is used for both input and output here, which is unusual.
	// It seems to be intended as an output-only array for results.
	kernel_get>>(device_keys, values, numKeys, hashtable);

	// Synchronization: Wait for the kernel to complete.
	cudaDeviceSynchronize();
	cudaFree(device_keys);

	return values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor()
{
	return ((float) hashtable.actual_size) / hashtable.max_size;
}


// --- Convenience macros for a simplified external API ---
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

/*
 * @note The following line includes a .cpp file, which is highly unconventional.
 */
#include "test_map.cpp"

/*
 * @note The following block appears to be a re-definition of the header file content.
 */
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

// Value representing an empty slot in the hash table.
#define	KEY_INVALID		0
// Default number of threads per block for kernel launches.
#define BLOCK_SIZE		512
// The maximum load factor before a reshape is triggered.
#define MAX_LOAD_FACTOR	0.9

// Macro for fatal error checking.
#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)

// A list of large prime numbers.
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


// Redundant hash function definitions.
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
 * @struct entry_t
 * @brief Represents a single key-value pair in the hash table.
 */
typedef struct entry {
	int key, value;
} entry_t;

/**
 * @struct hashtable_t
 * @brief The main control structure for the hash table, passed to kernels.
 */
typedef struct hashtable {
	entry_t *map;           // Device pointer to the array of key-value pairs.
	unsigned int max_size;    // The capacity of the hash table.
	unsigned int actual_size; // The current number of elements in the table.
} hashtable_t;



/**
 * @class GpuHashTable
 * @brief Host-side class for managing the GPU hash table.
 */
class GpuHashTable
{
	hashtable_t hashtable; // The main hash table structure.

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
