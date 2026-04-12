
/**
 * @file gpu_hashtable.cu
 * @brief CUDA-accelerated hash table using linear probing and atomic synchronization.
 * 
 * This module implements a parallel hash table residing in GPU global memory.
 * It uses linear probing for collision resolution and atomic operations (atomicCAS)
 * to ensure thread-safe concurrent access from thousands of GPU threads.
 */

#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "gpu_hashtable.hpp"

/**
 * @brief CUDA kernel to migrate elements during table resizing.
 * 
 * Algorithm: Parallel re-hashing with linear probing.
 * 
 * @param src Source hash table buffer.
 * @param size Source capacity.
 * @param dst Destination hash table buffer.
 * @param new_size Target capacity.
 */
__global__ void rehash(elem_t *src, unsigned int size, elem_t *dst, unsigned int new_size) {
	/**
	 * Thread Indexing: Maps 1D grid/block coordinates to the source table buckets.
	 */
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Pre-condition: Thread must be within source table boundaries.
	if (idx >= size) {
		return;
	}

	// Optimization: Skip empty slots.
	if (src[idx].key == KEY_INVALID) {
		return;
	}

	unsigned int new_index = hash_f(src[idx].key, new_size);

	/**
	 * Block Logic: Re-inserts existing keys into the expanded table.
	 * Synchronization: atomicCAS ensures atomic ownership of a bucket.
	 */
	bool finished = false;
	while (!finished) {
		// Atomic: Attempt to claim an empty slot (key 0).
		unsigned int old = atomicCAS(&dst[new_index].key, KEY_INVALID, src[idx].key);

		if (old == KEY_INVALID) {
			// Success: Map the value to the newly acquired slot.
			dst[new_index].value = src[idx].value;
			finished = true;
		} else {
			// Conflict: Linear probing step.
			new_index = (++new_index) % new_size;
		}
	}
}

/**
 * @brief CUDA kernel for parallel batch insertion of key-value pairs.
 * 
 * @param hash_table Pointer to the hash map in global memory.
 * @param size Current table capacity.
 * @param keys Array of input keys.
 * @param values Array of corresponding values.
 * @param no_pairs Number of elements in the batch.
 */
__global__ void insert(elem_t *hash_table, unsigned int size, int *keys, int *values, int no_pairs) {
	/**
	 * Thread Indexing: Each thread handles one entry in the input batch.
	 */
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int old;
	if (idx >= no_pairs)
		return;

	unsigned int hash_index = hash_f(keys[idx], size);

	/**
	 * Block Logic: Insertion with conflict resolution.
	 * Invariant: Probing continues until an empty slot is claimed or an existing key is matched.
	 */
	bool inserted = false;
	while (!inserted) {
		// Atomic: Attempt to write key to a 0 slot.
		old = atomicCAS(&hash_table[hash_index].key, KEY_INVALID, keys[idx]);

		if (old == KEY_INVALID || old == keys[idx]) {
			// Success/Update: Slot now belongs to this key.
			hash_table[hash_index].value = values[idx];
			inserted = true;
		} else {
			// Collision: Move to the next bucket.
			hash_index = (++hash_index) % size;
		}
	}
}

/**
 * @brief CUDA kernel for parallel batch retrieval.
 * 
 * Algorithm: Linear probing search path traversal.
 */
__global__ void get(elem_t *hash_table, unsigned int size, int *keys, int no_pairs, int *result) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= no_pairs)
		return;

	unsigned int hash_index = hash_f(keys[idx], size);

	/**
	 * Block Logic: Search loop along the probe path.
	 * Invariant: Terminates when the specific key is located.
	 */
	for (int i = 0; i < size; i++, hash_index = (++hash_index) % size) {
		if (hash_table[hash_index].key == keys[idx]) {
			result[idx] = hash_table[hash_index].value;
			break;
		}
	}
}


GpuHashTable::GpuHashTable(int size) {
	/**
	 * Functional Utility: Initializes the hash table state.
	 * Allocates global memory for buckets and zero-initializes keys.
	 */
	cudaError_t ret;
	void *new_hash_table;

	this->no_elements = 0;
	this->size = size;

	ret = cudaMalloc(&new_hash_table, size * sizeof(elem_t));
	DIE(ret != cudaSuccess, "cudaMalloc failed!");

	// Invariant: Keys initialized to 0 (KEY_INVALID) to mark buckets as available.
	ret = cudaMemset(new_hash_table, KEY_INVALID, size * sizeof(elem_t));
	DIE(ret != cudaSuccess, "cudaMemset failed!");

	this->hash_table = (elem_t *)new_hash_table;
}


GpuHashTable::~GpuHashTable() {
	/**
	 * Functional Utility: Releases GPU resources.
	 */
	cudaError_t ret;

	ret = cudaFree(this->hash_table);
	DIE(ret != cudaSuccess, "cudaFree failed!");
}


void GpuHashTable::reshape(int numBucketsReshape) {
	/**
	 * Functional Utility: Dynamic re-hashing orchestrator.
	 * Logic: Scales capacity, migrates elements via parallel kernel, and replaces the buffer.
	 */
	cudaError_t ret;
	elem_t *new_hash_table;
	void *new_hash_table_p;

	ret = cudaMalloc(&new_hash_table_p, numBucketsReshape * sizeof(elem_t));
	DIE(ret != cudaSuccess, "cudaMalloc failed!");

	ret = cudaMemset(new_hash_table_p, KEY_INVALID, numBucketsReshape * sizeof(elem_t));
	DIE(ret != cudaSuccess, "cudaMemset failed!");

	new_hash_table = (elem_t *)new_hash_table_p;
	
	// Execution: Dispatch data migration requests.
	unsigned int no_blocks = this->size / NO_BLOCK_THREADS + 1;
	rehash<<<no_blocks, NO_BLOCK_THREADS>>>(this->hash_table, this->size, new_hash_table, numBucketsReshape);
	
	// Synchronization: Ensure re-hashing is complete before freeing old memory.
	cudaDeviceSynchronize();

	ret = cudaFree(this->hash_table);
	DIE(ret != cudaSuccess, "cudaFree failed!");

	this->hash_table = new_hash_table;
	this->size = numBucketsReshape;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	/**
	 * Functional Utility: Host-side interface for bulk insertions.
	 * Logic: monitors occupancy, triggers reshape if load factor reaches 100%, and launches kernel.
	 */
	cudaError_t ret;

	// Optimization: Enforces a minimum load factor of 80% after expansion.
	if ((float)(this->no_elements + numKeys) / this->size >= MAX_LOAD_FACTOR) {
		reshape((int)(this->no_elements + numKeys) / MIN_LOAD_FACTOR);
	}
	this->no_elements += numKeys;

	void *cuda_keys_p, *cuda_values_p;

	ret = cudaMalloc(&cuda_keys_p, numKeys * sizeof(int));
	DIE(ret != cudaSuccess, "cudaMalloc cuda_keys failed!");
	ret = cudaMemcpy(cuda_keys_p, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(ret != cudaSuccess, "cudaMemcpy cuda_keys failed!");

	ret = cudaMalloc(&cuda_values_p, numKeys * sizeof(int));
	DIE(ret != cudaSuccess, "cudaMalloc cuda_values failed!");
	ret = cudaMemcpy(cuda_values_p, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(ret != cudaSuccess, "cudaMemcpy cuda_values failed!");

	int *cuda_keys = (int *)cuda_keys_p;
	int *cuda_values = (int *)cuda_values_p;
	
	// Execution: Concurrent batch insertion.
	unsigned int no_blocks = (numKeys + NO_BLOCK_THREADS - 1) / NO_BLOCK_THREADS;
	insert<<<no_blocks, NO_BLOCK_THREADS>>>(this->hash_table, this->size, cuda_keys, cuda_values, numKeys);

	ret = cudaFree(cuda_keys_p);
	DIE(ret != cudaSuccess, "cudaFree cuda_keys failed!");

	ret = cudaFree(cuda_values_p);
	DIE(ret != cudaSuccess, "cudaFree cuda_values failed!");

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	/**
	 * Functional Utility: Parallel batch retrieval interface.
	 * Logic: Uses Managed Memory for result consistency.
	 */
	cudaError_t ret;
	void *result_p, *dev_keys_p;
	
	// Optimization: Managed memory allows host to access device-populated array after sync.
	ret = cudaMallocManaged(&result_p, numKeys * sizeof(int));
	DIE(ret != cudaSuccess, "cudaMalloc dev_res failed!");

	ret = cudaMalloc(&dev_keys_p, numKeys * sizeof(int));
	DIE(ret != cudaSuccess, "cudaMalloc dev_keys failed!");
	ret = cudaMemcpy(dev_keys_p, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(ret != cudaSuccess, "cudaMemcpy dev_keys failed!");

	int *result = (int *)result_p;
	int *dev_keys = (int *)dev_keys_p;
	
	// Execution: Concurrent search threads.
	unsigned int no_blocks = (numKeys + NO_BLOCK_THREADS - 1) / NO_BLOCK_THREADS;
	get<<<no_blocks, NO_BLOCK_THREADS>>>(this->hash_table, this->size, dev_keys, numKeys, result);
	
	cudaDeviceSynchronize();

	ret = cudaFree(dev_keys_p);
	DIE(ret != cudaSuccess, "cudaFree dev_keys failed!");

	return result;
}


float GpuHashTable::loadFactor() {
	/**
	 * Functional Utility: Accessor for the occupancy ratio.
	 */
	if (!this->no_elements)
		return 0.f;

	return (float) this->no_elements / this->size;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

#include <iostream>
#include <limits.h>

using namespace std;

#define	KEY_INVALID		0
#define NO_BLOCK_THREADS 1024
#define MIN_LOAD_FACTOR .8f
#define MAX_LOAD_FACTOR 1.0f

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
 * Data Storage: Prime numbers for universal hash coefficients.
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
 * @brief Host-side hash calculations using precomputed primes.
 */
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
 * @struct elem
 * @brief Representation of a single hash bucket.
 */
typedef struct elem {
	int key;
	int value;
} elem_t;

/**
 * @brief Device-side non-cryptographic hash function for key-to-bucket mapping.
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
 * @struct hashtable
 * @brief Descriptor for the GPU-resident hash table allocation.
 */
struct hashtable {
	int size;
	int real_size;
	elem_t *table;
};

class GpuHashTable
{
	private:
		hashtable h;
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
/* Update complete: gpu_hashtable.cu processed with semantic documentation. */
