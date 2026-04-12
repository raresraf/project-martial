
/**
 * @file gpu_hashtable.cu
 * @brief High-performance GPU hash table implementation using linear probing and Managed Memory.
 * 
 * This module provides a thread-safe parallel hash table residing in CUDA Managed Memory.
 * It uses linear probing for collision resolution and atomic operations (atomicCAS, atomicExch)
 * to maintain consistency during high-concurrency batch insertions and lookups.
 */

#include "gpu_hashtable.hpp"

GpuHashTable::GpuHashTable(int size) {
	/**
	 * Functional Utility: Initializes the GPU hash table state.
	 * Allocates managed memory for the bucket array and the occupancy counter.
	 */
	hashtable_size = size;

	// Optimization: Counter in managed memory for seamless host-device synchronization.
	cudaMallocManaged((void **) &hashtable_current_size, sizeof(int));
	*hashtable_current_size = 0;

	cudaMallocManaged((void **) &hashtable, size * sizeof(struct node));
	// Initial State: Key 0 (KEY_INVALID) indicates an available slot.
	for (int i = 0; i < size; i++)
		hashtable[i].key = KEY_INVALID;
}

GpuHashTable::~GpuHashTable() {
	/**
	 * Functional Utility: Deallocates device resources.
	 */
	cudaFree(hashtable);
	cudaFree(hashtable_current_size);
}

void GpuHashTable::reshape(int numBucketsReshape) {
	/**
	 * Functional Utility: Dynamic re-hashing orchestrator.
	 * Logic: 
	 * 1. Extracts all active key-value pairs from the current managed buffer.
	 * 2. Re-allocates a larger managed buffer.
	 * 3. Re-inserts the extracted elements using the batch insertion interface.
	 */
	int new_size = numBucketsReshape * 1.07f; // Optimization: Add a safety margin to the new capacity.
	int old_size = *hashtable_current_size;

	// Extraction: Temporal storage for existing entries.
	int *old_keys = (int *)malloc(*(hashtable_current_size) * sizeof(int));
	int *old_values = (int *)malloc(*(hashtable_current_size) *sizeof(int));
	
	int index = 0;
	for (int i = 0; i < hashtable_size; i++) {
		if (hashtable[i].key != KEY_INVALID) {
			old_keys[index] = hashtable[i].key;
			old_values[index] = hashtable[i].value;
			index++;
		}
	}

	cudaFree(hashtable);
	cudaFree(hashtable_current_size);

	hashtable_size = new_size;

	cudaMallocManaged((void **) &hashtable_current_size, sizeof(int));
	*hashtable_current_size = 0;

	cudaMallocManaged((void **) &hashtable, new_size * sizeof(struct node));
	for (int i = 0; i < new_size; i++)
		hashtable[i].key = KEY_INVALID;

	// Restoration: Re-insert all previously stored data.
	insertBatch(old_keys, old_values, old_size);

	free(old_keys);
	free(old_values);
}

/**
 * @brief CUDA kernel for parallel batch insertion of key-value pairs.
 * 
 * Algorithm: Linear probing with atomic resolution.
 * 
 * @param keys Input keys buffer.
 * @param values Corresponding values buffer.
 * @param numKeys Batch size.
 * @param hashtable Pointer to the managed hash map.
 * @param hashtable_size Capacity of the map.
 * @param hashtable_current_size Global pointer to occupancy counter.
 */
__global__ void kernel_insert(int *keys, int *values, int numKeys,
struct node *hashtable, int hashtable_size,
int *hashtable_current_size) {
	
	/**
	 * Thread Indexing: Maps 1D grid/block coordinates to the input batch.
	 */
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numKeys)
		return;

	int my_key = keys[idx];
	int my_value = values[idx];

	int hash = hash_function(my_key, hashtable_size);
	
	/**
	 * Block Logic: Atomic slot acquisition loop.
	 * Synchronization: 
	 * 1. atomicCAS ensures only one thread claims an empty bucket.
	 * 2. atomicExch handles value assignment atomically.
	 * 3. atomicAdd tracks the total number of unique keys.
	 */
	while (1) {
		// Atomic: Attempt to write current key to a 0 slot.
		int init_val = atomicCAS(&(hashtable[hash].key), KEY_INVALID,
			my_key);

		if (init_val == KEY_INVALID) {
			// Success: Claimed an empty slot.
			atomicExch(&hashtable[hash].value, my_value);
			atomicAdd(hashtable_current_size, 1);

			__syncthreads();
			break;
		}
		
		else if (init_val == my_key) {
			// Update: Slot already belongs to this key.
			atomicExch(&hashtable[hash].value, my_value);

			__syncthreads();
                        break;
		}

		// Linear Probing: Circular increment on conflict.
		hash = (hash + 1) % hashtable_size;
	}
}

bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	/**
	 * Functional Utility: Host-side interface for parallel batch insertion.
	 * Logic: Triggers expansion if occupancy exceeds threshold, transfers data, and launches kernel.
	 */
	if (numKeys <= 0)
		return true;

	// Capacity enforcement: Reshape if current batch exceeds available space.
	if (*hashtable_current_size + numKeys > hashtable_size)
		reshape(hashtable_size + numKeys);

	int *device_keys, *device_values;

	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int),
		cudaMemcpyHostToDevice);
	
	// Execution: Dispatch concurrent insertion.
	kernel_insert<<< (numKeys + 1023) / 1024, 1024 >>>(device_keys, device_values, numKeys,
		hashtable, hashtable_size, 
		hashtable_current_size);
	cudaDeviceSynchronize();

	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}

/**
 * @brief CUDA kernel for parallel batch retrieval.
 * 
 * Algorithm: Linear probing search.
 */
__global__ void kernel_get(int *keys, int *values, int numKeys,
struct node *hashtable, int hashtable_size) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numKeys)
		return;

	int my_key = keys[idx];
	int hash = hash_function(my_key, hashtable_size);

	/**
	 * Block Logic: Linear search path.
	 * Invariant: Search terminates when the key is found.
	 */
	while (1) {
		if (hashtable[hash].key == my_key) {
			values[idx] = hashtable[hash].value;
			__syncthreads();
			break;;
		}

                hash = (hash + 1) % hashtable_size;
	}
}

int* GpuHashTable::getBatch(int* keys, int numKeys) {
	/**
	 * Functional Utility: Parallel batch retrieval.
	 */
	int *values = (int *)calloc(numKeys, sizeof(int));

	int *device_keys, *device_values;

	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int),
		cudaMemcpyHostToDevice);

	// Execution: Concurrent lookup.
	kernel_get<<< (numKeys + 1023) / 1024, 1024 >>>(device_keys, device_values, numKeys,
		hashtable, hashtable_size);
	cudaDeviceSynchronize();

	cudaMemcpy(values, device_values, numKeys * sizeof(int),
		cudaMemcpyDeviceToHost);

	cudaFree(device_keys);
	cudaFree(device_values);

	return values;
}

float GpuHashTable::loadFactor() {
	/**
	 * Functional Utility: Returns current table occupancy percentage.
	 */
	return *(hashtable_current_size) / (float)hashtable_size;
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
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

using namespace std;

#define KEY_INVALID 0

#define DIE(assertion, call_description)              \
	do                                            \
	{                                             \
		if (assertion)                        \
		{                                     \
			fprintf(stderr, "(%s, %d): ", \
				__FILE__, __LINE__);  \
			perror(call_description);     \
			exit(errno);                  \
		}                                     \
	} while (0)

/**
 * Data Storage: Prime constants for universal hashing.
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
	11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu};


/**
 * @brief Device-side hash function mapping a key to a bucket index.
 */
__device__ int hash_function(int data, int limit) {
        return ((long)abs(data) * 21407807219llu) % 139193449418173llu % limit;
}

/**
 * @struct node
 * @brief Representation of a single key-value entry in the map.
 */
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
/* Update complete: gpu_hashtable.cu processed with semantic documentation. */
