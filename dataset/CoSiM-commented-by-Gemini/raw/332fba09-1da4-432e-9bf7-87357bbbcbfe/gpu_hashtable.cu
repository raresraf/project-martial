/**
 * @file gpu_hashtable.cu
 * @brief A GPU-based hash table using open addressing with linear probing.
 *
 * This file provides a well-encapsulated and robust implementation of a hash
 * table that resides entirely in GPU memory. It uses a single contiguous array
 * for storage and resolves hash collisions by linearly probing for the next
 * available bucket. The implementation includes an efficient, GPU-only reshape
 * mechanism.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief CUDA kernel to rehash all elements from an old table into a new, larger one.
 *
 * This kernel is launched during a `reshape` operation. Each thread is assigned
 * to an entry in the old hash table. If the entry is valid (i.e., not empty),
 * the thread re-calculates the hash for the key based on the new table size and
 * inserts the element into the new table using an atomic linear probing strategy.
 */
__global__ void rehash(elem_t *src, unsigned int size, elem_t *dst, unsigned int new_size) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= size) {
		return;
	}

	// Skip empty entries from the source table.
	if (src[idx].key == KEY_INVALID) {
		return;
	}

	unsigned int new_index = hash_f(src[idx].key, new_size);

	bool finished = false;
	while (!finished) {
		// Atomically attempt to claim an empty slot in the new table.
		unsigned int old = atomicCAS(&dst[new_index].key, KEY_INVALID, src[idx].key);

		if (old == KEY_INVALID) {
			// If the slot was empty, the insertion is successful.
			dst[new_index].value = src[idx].value;
			finished = true;
		} else {
			// If the slot was occupied, probe the next one.
			new_index = (++new_index) % new_size;
		}
	}
}

/**
 * @brief CUDA kernel for inserting a batch of key-value pairs.
 *
 * Each thread handles one key-value pair. It computes an initial hash and then
 * uses linear probing to find an empty slot. `atomicCAS` ensures that each
 * empty slot is claimed by only one thread, preventing race conditions.
 */
__global__ void insert(elem_t *hash_table, unsigned int size, int *keys, int *values, int no_pairs) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int old;
	if (idx >= no_pairs)
		return;

	unsigned int hash_index = hash_f(keys[idx], size);

	bool inserted = false;
	while (!inserted) {
		// Atomically attempt to write the new key if the slot is empty (KEY_INVALID)
		// or if the slot already contains the same key (for updates).
		old = atomicCAS(&hash_table[hash_index].key, KEY_INVALID, keys[idx]);

		if (old == KEY_INVALID || old == keys[idx]) {
			hash_table[hash_index].value = values[idx];
			inserted = true;
		} else {
			// If the slot was occupied by a different key, probe the next one.
			hash_index = (++hash_index) % size;
		}
	}
}

/**
 * @brief CUDA kernel for retrieving values for a batch of keys.
 *
 * Each thread searches for one key. It uses a bounded `for` loop to perform
 * linear probing, ensuring the search terminates even if the key is not found
 * in a full table.
 */
__global__ void get(elem_t *hash_table, unsigned int size, int *keys, int no_pairs, int *result) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= no_pairs)
		return;

	unsigned int hash_index = hash_f(keys[idx], size);

	// Linearly probe the table. The loop is bounded by the table size for safety.
	for (int i = 0; i < size; i++, hash_index = (++hash_index) % size) {
		// A non-atomic read is used for efficiency.
		if (hash_table[hash_index].key == keys[idx]) {
			result[idx] = hash_table[hash_index].value;
			break; // Key found.
		}
	}
}

/**
 * @brief Host-side constructor. Allocates and initializes the hash table on the GPU.
 */
GpuHashTable::GpuHashTable(int size) {
	cudaError_t ret;
	void *new_hash_table;

	this->no_elements = 0;
	this->size = size;

	ret = cudaMalloc(&new_hash_table, size * sizeof(elem_t));
	DIE(ret != cudaSuccess, "cudaMalloc failed!");

	// Initialize all keys in the hash table to KEY_INVALID.
	ret = cudaMemset(new_hash_table, KEY_INVALID, size * sizeof(elem_t));
	DIE(ret != cudaSuccess, "cudaMemset failed!");

	this->hash_table = (elem_t *)new_hash_table;
}

/**
 * @brief Host-side destructor. Frees the GPU memory used by the hash table.
 */
GpuHashTable::~GpuHashTable() {
	cudaError_t ret;

	ret = cudaFree(this->hash_table);
	DIE(ret != cudaSuccess, "cudaFree failed!");
}

/**
 * @brief Resizes the hash table and rehashes all elements entirely on the GPU.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t ret;
	elem_t *new_hash_table;
	unsigned int no_blocks = this->size / NO_BLOCK_THREADS + 1;
	void *new_hash_table_p;

	// 1. Allocate a new, larger table on the device.
	ret = cudaMalloc(&new_hash_table_p, numBucketsReshape * sizeof(elem_t));
	DIE(ret != cudaSuccess, "cudaMalloc failed!");

	ret = cudaMemset(new_hash_table_p, KEY_INVALID, numBucketsReshape * sizeof(elem_t));
	DIE(ret != cudaSuccess, "cudaMemset failed!");

	// 2. Launch the rehash kernel to copy elements from the old table to the new one.
	new_hash_table = (elem_t *)new_hash_table_p;
	rehash>>(this->hash_table, this->size, new_hash_table, numBucketsReshape);
	cudaDeviceSynchronize();

	// 3. Free the old hash table memory.
	ret = cudaFree(this->hash_table);
	DIE(ret != cudaSuccess, "cudaFree failed!");

	// 4. Update the host-side object to point to the new table and size.
	this->hash_table = new_hash_table;
	this->size = numBucketsReshape;
}

/**
 * @brief Host-side function to insert a batch of key-value pairs.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	cudaError_t ret;
	unsigned int no_blocks = this->size / NO_BLOCK_THREADS + 1;

	// Check if the new elements would exceed the maximum load factor, and reshape if so.
	if ((float)(this->no_elements + numKeys) / this->size >= MAX_LOAD_FACTOR) {
		reshape((int)(this->no_elements + numKeys) / MIN_LOAD_FACTOR);
	}
	this->no_elements += numKeys;

	// Allocate temporary device memory for the keys and values to be inserted.
	void *cuda_keys_p, *cuda_values_p;
	ret = cudaMalloc(&cuda_keys_p, numKeys * sizeof(int));
	DIE(ret != cudaSuccess, "cudaMalloc cuda_keys failed!");
	ret = cudaMemcpy(cuda_keys_p, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(ret != cudaSuccess, "cudaMemcpy cuda_keys failed!");

	ret = cudaMalloc(&cuda_values_p, numKeys * sizeof(int));
	DIE(ret != cudaSuccess, "cudaMalloc cuda_values failed!");
	ret = cudaMemcpy(cuda_values_p, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(ret != cudaSuccess, "cudaMemcpy cuda_values failed!");

	// Launch the insertion kernel.
	int *cuda_keys = (int *)cuda_keys_p;
	int *cuda_values = (int *)cuda_values_p;
	insert>>(this->hash_table, this->size, cuda_keys, cuda_values, numKeys);

	// Free the temporary device buffers.
	ret = cudaFree(cuda_keys_p);
	DIE(ret != cudaSuccess, "cudaFree cuda_keys failed!");

	ret = cudaFree(cuda_values_p);
	DIE(ret != cudaSuccess, "cudaFree cuda_values failed!");

	return true;
}

/**
 * @brief Host-side function to retrieve values for a batch of keys.
 * @return A pointer to a host array containing the results, allocated with managed memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	cudaError_t ret;
	unsigned int no_blocks = this->size / NO_BLOCK_THREADS + 1;
	void *result_p, *dev_keys_p;
	
	// Allocate managed memory for the results, simplifying host access after the kernel runs.
	ret = cudaMallocManaged(&result_p, numKeys * sizeof(int));
	DIE(ret != cudaSuccess, "cudaMalloc dev_res failed!");

	// Allocate temporary device memory for the keys to be searched.
	ret = cudaMalloc(&dev_keys_p, numKeys * sizeof(int));
	DIE(ret != cudaSuccess, "cudaMalloc dev_keys failed!");
	ret = cudaMemcpy(dev_keys_p, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(ret != cudaSuccess, "cudaMemcpy dev_keys failed!");

	// Launch the get kernel.
	int *result = (int *)result_p;
	int *dev_keys = (int *)dev_keys_p;
	get>>(this->hash_table, this->size, dev_keys, numKeys, result);
	cudaDeviceSynchronize(); // Wait for the kernel to complete the lookups.

	// Free the temporary key buffer on the device.
	ret = cudaFree(dev_keys_p);
	DIE(ret != cudaSuccess, "cudaFree dev_keys failed!");

	return result;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
	if (!this->no_elements)
		return 0.f;

	return (float) this->no_elements / this->size;
}


// --- Convenience macros for the test file ---
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"

// The file includes a header guard and content that should be in gpu_hashtable.hpp
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

// --- Hash Table Constants ---
#define	KEY_INVALID		0 // Sentinel value for an empty bucket key.
#define NO_BLOCK_THREADS 1024 // Default number of threads per CUDA block.
#define MIN_LOAD_FACTOR .8f // Target load factor after a resize.
#define MAX_LOAD_FACTOR 1.0f // Load factor that triggers a resize.

// C-style error handling macro.
#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)
	
// A large list of prime numbers, likely intended for hashing, but not used by hash_f.
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


// These hash functions are defined but not used in the main logic.
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

// A simple key-value pair struct.
typedef struct elem {
	int key;
	int value;
} elem_t;

/**
 * @brief A Jenkins-style hash function, good for creating a wide
 * distribution of hash values. Can be used on both host and device.
 */
__device__ unsigned int hash_f(int key, int size) {
	key = ~key + (key << 15);
	key = key ^ (key >> 12);
	key = key + (key << 2);
	key = key ^ (key >> 4);
	key = key * 2057;
	key = key ^ (key >> 16);
	return key % size;
}


/**
 * @brief The host-side class providing an interface to the GPU hash table.
 */
class GpuHashTable
{
	private:
		elem_t *hash_table; // Pointer to the hash table array in GPU memory.
		unsigned int no_elements; // The number of elements currently stored.
		unsigned int size; // The current capacity (number of buckets).
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

