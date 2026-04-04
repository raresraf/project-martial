/**
 * @file gpu_hashtable.cu
 * @brief A self-contained implementation of a GPU-accelerated hash table.
 * @details This file provides both the host-side management class (`GpuHashTable`)
 * and the device-side CUDA kernels. The implementation uses open addressing with
 * linear probing for collision resolution. A key feature is its use of the
 * CUDA Occupancy API (`cudaOccupancyMaxPotentialBlockSize`) to optimize kernel
 * launch parameters for improved GPU utilization.
 */

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gpu_hashtable.hpp"

/**
 * @brief Computes a hash value for a given integer key on the device.
 * @param data The integer key to hash.
 * @param limit The capacity of the hash table, used for the modulo operation.
 * @return An integer hash value within the range [0, limit-1].
 * @details The function uses a multiplicative hashing scheme with large prime numbers.
 */
__device__ int my_hash(int data, int limit) {
	return ((long) abs(data) * 163307llu) % 135931102921llu % limit;
}



/**
 * @brief Constructs a GpuHashTable object.
 * @param size The initial capacity (number of buckets) for the hash table.
 * @details Allocates global device memory for the hash table and initializes all
 * slots to an empty state (key = 0).
 */
GpuHashTable::GpuHashTable(int size) {
	cudaError_t cuda_malloc_ERROR_it;
	hash_table.size = size;
	hash_table.nr_inserted_pairs = 0;
	hash_table.map = NULL;

	// Memory Hierarchy: Allocate the main hash table storage in global device memory.
	cuda_malloc_ERROR_it = cudaMalloc(&hash_table.map,
		size * sizeof(pair_t));
	DIE(cuda_malloc_ERROR_it != cudaSuccess, "Eroare malloc init");

	// Initialize all table slots to empty (0).
	cudaMemset(hash_table.map, 0, size * sizeof(pair_t));
}


/**
 * @brief Destroys the GpuHashTable object, freeing its device memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hash_table.map);
}


/**
 * @brief CUDA kernel to rehash all entries from an old table to a new one.
 * @param old_htable The source hash table.
 * @param new_htable The destination hash table (must be pre-allocated and empty).
 * @details Each thread migrates one valid entry. It re-computes the hash for the
 * new table size and uses linear probing with `atomicCAS` to find an empty slot.
 * The probing logic is split into two loops to handle wrap-around.
 */
__global__ void cuda_reshape(hash_table_t old_htable, hash_table_t new_htable) {

	// Thread Indexing: Each thread is responsible for one potential entry in the old table.
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int block_value, key;

	// Pre-condition: Process only if the thread corresponds to a valid, occupied slot.
	if (id < old_htable.size && old_htable.map[id].key != 0) {
		key = old_htable.map[id].key;
		// Hashing: Re-calculate hash based on the new table's size.
		int hash = my_hash(key, new_htable.size);

		// Block Logic: First pass of linear probing, from the hash value to the end of the table.
		for (int i = hash; i < new_htable.size; i++) {
			// Atomic Slot Acquisition: Attempt to claim an empty slot (key=0).
			block_value = atomicCAS(&new_htable.map[i].key, 0,
					key);
			// Invariant: If the original value was 0, this thread has claimed the slot.
			if (block_value == 0) {
				new_htable.map[i].value =
				old_htable.map[id].value;
				return;
			}
		}

		// Block Logic: Second pass of linear probing, wrapping around from the beginning of the table.
		for (int i = 0; i < hash; i++) {
			block_value = atomicCAS(&new_htable.map[i].key, 0,
					key);
			if (block_value == 0) {
				new_htable.map[i].value =
				old_htable.map[id].value;
				return;
			}
		}
	}
}


/**
 * @brief Resizes the hash table to a new capacity, rehashing all elements.
 * @param numBucketsReshape The new capacity of the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t cuda_malloc_ERROR_r;
	hash_table_t new_htable;
	new_htable.size = numBucketsReshape;
	int blockSize;
	int minGridSize;
	int gridSize;

	// Allocate memory for the new table structure on the device.
	cuda_malloc_ERROR_r = cudaMalloc(&new_htable.map,
		numBucketsReshape * sizeof(pair_t));
	DIE(cuda_malloc_ERROR_r != cudaSuccess, "Eroare malloc reshape");

	cudaMemset(new_htable.map, 0, numBucketsReshape * sizeof(pair_t));

	// Performance Optimization: Use the CUDA Occupancy API to determine optimal
	// launch parameters (block size) for the reshape kernel.
	cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
		cuda_reshape, 0, 0);
	// Kernel Launch Configuration: Calculate grid size based on the determined block size.
	gridSize = (hash_table.size + blockSize - 1) / blockSize;
	cuda_reshape<<<gridSize, blockSize>>>(hash_table, new_htable);
	cudaDeviceSynchronize();

	// Atomically swap to the new table.
	cudaFree(hash_table.map);
	new_htable.nr_inserted_pairs = hash_table.nr_inserted_pairs;
	hash_table = new_htable;
}

/**
 * @brief CUDA kernel for parallel insertion of a batch of key-value pairs.
 * @param hash_table The target hash table.
 * @param keys Device pointer to the array of keys to insert.
 * @param values Device pointer to the array of values to insert.
 * @param numEntries The number of pairs in the batch.
 * @details Each thread handles one pair, using linear probing and `atomicCAS`
 * to find an empty slot or update an existing key. The probing logic wraps around
 * the table boundary.
 */
__global__ void cuda_insert(hash_table_t hash_table, int *keys, int *values,
	 int numEntries) {

	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int block_value, key, value;
	
	// Pre-condition: Ensure thread is within the bounds of the input batch.
	if (id < numEntries) {
		key = keys[id];
		value = values[id];
		int hash = my_hash(key, hash_table.size);

		// Block Logic: Linear probing from hash to the end of the table.
		for (int i = hash; i < hash_table.size; i++) {
			block_value = atomicCAS(&hash_table.map[i].key, 0,
					key);

			// Invariant: If slot was empty or held the same key, the update is successful.
			if (block_value == 0 || block_value == key) {
				hash_table.map[i].value = value;
				return;
			}
		}
		// Block Logic: Wrap-around linear probing from the start of the table.
		for (int i = 0; i < hash; i++) {
			block_value = atomicCAS(&hash_table.map[i].key, 0,
					key);

			if (block_value == 0 || block_value == key) {
				hash_table.map[i].value = value;
				return;
			}
		}
	}
}


/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys A host pointer to an array of keys.
 * @param values A host pointer to an array of values.
 * @param numKeys The number of pairs in the batch.
 * @return `true` upon successful completion.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	cudaError_t cuda_malloc_ERROR_k;
	cudaError_t cuda_malloc_ERROR_v;
	int *vector_keys;
	int *vector_values;
	size_t memory_size = numKeys * sizeof(int);
	int blockSize;
	int minGridSize;
	int gridSize;

	cuda_malloc_ERROR_k = cudaMalloc(&vector_keys, memory_size);
	cuda_malloc_ERROR_v = cudaMalloc(&vector_values, memory_size);
	DIE(cuda_malloc_ERROR_k != cudaSuccess, "Eroare malloc chei insertB");
	DIE(cuda_malloc_ERROR_v != cudaSuccess, "Eroare malloc valori insertB");

	// Pre-condition: Check if the load factor will exceed a threshold.
	// If so, resize the table to maintain performance.
	if (numKeys + hash_table.nr_inserted_pairs >= hash_table.size * 0.85f)
		reshape((int)((numKeys + hash_table.nr_inserted_pairs)
			* 100.f / 85.f));

	// Data Transfer: Copy the batch from host memory to device memory.
	cudaMemcpy(vector_keys, keys, memory_size, cudaMemcpyHostToDevice);
	cudaMemcpy(vector_values, values, memory_size, cudaMemcpyHostToDevice);

	// Performance Optimization: Use Occupancy API to find optimal block size.
	cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
		cuda_insert, 0, 0);
	// Kernel Launch Configuration: Calculate grid size accordingly.
	gridSize = (numKeys + blockSize - 1) / blockSize;
	cuda_insert<<<gridSize, blockSize>>>(hash_table, vector_keys, vector_values,
		numKeys);
	cudaDeviceSynchronize();

	hash_table.nr_inserted_pairs = hash_table.nr_inserted_pairs + numKeys;

	cudaFree(vector_keys);
	cudaFree(vector_values);

	return true;
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param hash_table The hash table to search.
 * @param keys Device pointer to an array of keys to look up.
 * @param values Device pointer to an array where found values will be written.
 * @param numEntries The number of keys in the batch.
 * @details Each thread searches for one key using linear probing. This is a
 * read-only operation and does not require atomic instructions.
 */
__global__ void cuda_get(hash_table_t hash_table, int *keys, int *values, int numEntries) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (id < numEntries) {
		int key = keys[id];
		int hash = my_hash(keys[id], hash_table.size);

		// Block Logic: Linear probe from hash to end of table.
		for (int i = hash; i < hash_table.size; i++) {
			if (hash_table.map[i].key == key) {
				values[id] = hash_table.map[i].value;
				return;
			}
		}
		// Block Logic: Wrap-around linear probe from start of table.
		for (int i = 0; i < hash; i++) {
			if (hash_table.map[i].key == key) {
				values[id] = hash_table.map[i].value;
				return;
			}
		}
	}
}


/**
 * @brief Retrieves values for a batch of keys from the hash table.
 * @param keys A host pointer to an array of keys to look up.
 * @param numKeys The number of keys in the batch.
 * @return A host pointer to a newly allocated array containing the values.
 *         The caller is responsible for freeing this memory with `free()`.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	cudaError_t cuda_malloc_ERROR_k;
	cudaError_t cuda_malloc_ERROR_v;
	int *vector_keys;
	int *vector_values;
	size_t memory_size = numKeys * sizeof(int);
	int blockSize;
	int minGridSize;
	int gridSize;

	// Allocate temporary device buffers for keys and values.
	cuda_malloc_ERROR_k = cudaMalloc(&vector_keys, memory_size);
	cuda_malloc_ERROR_v = cudaMalloc(&vector_values, memory_size);
	DIE(cuda_malloc_ERROR_k != cudaSuccess, "Eroare malloc chei getB");
	DIE(cuda_malloc_ERROR_v != cudaSuccess, "Eroare malloc valori getB");

	int *values = (int*)malloc(numKeys * sizeof(int));

	// Data Transfer: Copy lookup keys from host to device.
	cudaMemcpy(vector_keys, keys, memory_size, cudaMemcpyHostToDevice);

	// Performance Optimization: Use Occupancy API to determine launch parameters.
	cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
		cuda_get, 0, 0);

	gridSize = (numKeys + blockSize - 1) / blockSize;
	cuda_get<<<gridSize, blockSize>>>(hash_table, vector_keys, vector_values,
		numKeys);
	cudaDeviceSynchronize();

	// Data Transfer: Copy results from device back to host.
	cudaMemcpy(values, vector_values, memory_size, cudaMemcpyDeviceToHost);

	cudaFree(vector_keys);
	cudaFree(vector_values);

	return values;
}



/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float (number of items / capacity).
 */
float GpuHashTable::loadFactor() {
	return (float)hash_table.nr_inserted_pairs / hash_table.size;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"



#ifndef _HASHCPU_
#define _HASHCPU_


using namespace std;

#define	KEY_INVALID		0

#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)

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





typedef struct pair {
	int key, value;
} pair_t;

typedef struct hash_table {
	int size;
	pair_t *map;
	int nr_inserted_pairs;
} hash_table_t;


class GpuHashTable
{
	hash_table_t hash_table;

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
