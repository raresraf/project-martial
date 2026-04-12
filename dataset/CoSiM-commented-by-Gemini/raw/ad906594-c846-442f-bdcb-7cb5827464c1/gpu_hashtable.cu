
/**
 * @file gpu_hashtable.cu
 * @brief High-throughput GPU-based hash table using linear probing and atomic synchronization.
 * 
 * This module implements a parallel hash table in CUDA, optimized for batch operations.
 * It employs a linear probing strategy for collision resolution and utilizes atomicCAS
 * to ensure thread-safe slot claiming in GPU global memory.
 */

#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "gpu_hashtable.hpp"

/**
 * @brief Device-side hash function using a linear congruential mapping with prime constants.
 * 
 * @param data Input key to be hashed.
 * @param limit Maximum bucket index (capacity).
 * @return uint32_t Calculated bucket index.
 */
__device__ uint32_t hashCode(int data, int limit) {
	return ((unsigned int) abs(data) * primeList[67]) % primeList[91] % limit;
}

/**
 * @brief CUDA kernel for parallel batch insertion into the hash table.
 * 
 * Algorithm: Linear probing with atomic synchronization.
 * Invariant: A slot is considered empty if its key equals KEY_INVALID (0).
 * 
 * @param ht Pointer to the hash table in global memory.
 * @param keys Array of input keys for insertion.
 * @param values Array of corresponding input values.
 * @param numKeys Number of elements in the batch.
 * @param capacity Current capacity of the hash table.
 */
__global__ void kernel_insert_batch(HashNode* ht, int *keys, int *values, int numKeys, int capacity) {
    /**
     * Thread Indexing: Maps 1D grid/block coordinates to the input batch indices.
     */
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // Pre-condition: Thread index must be within the batch boundaries.
    if (tid >= numKeys)
        return;

    uint32_t key = keys[tid];
    uint32_t old_key;
    uint32_t value = values[tid];
    uint32_t index = hashCode(key, capacity);
    
    // Invariant: Key 0 is reserved as the invalid/empty sentinel.
    if (key == 0)
        return;

    /**
     * Block Logic: Probe loop for slot acquisition.
     * Synchronization: atomicCAS ensures atomic ownership of an empty slot or update of an existing key.
     */
    while (true) {
        old_key = atomicCAS(&ht[index].key, KEY_INVALID, key);

        if (old_key == key || old_key == KEY_INVALID) {
            ht[index].value = value;
            return;
        }

        // Linear probing: Circular increment of the bucket index.
        index++;
        index %= capacity;
    }

}

/**
 * @brief CUDA kernel for parallel batch retrieval of values.
 * 
 * @param ht Source hash map in global memory.
 * @param keys Array of search keys.
 * @param values Output array for retrieved values.
 * @param capacity Current table capacity.
 */
__global__ void kernel_get_batch(HashNode* ht, int* keys, int* values, int capacity) {

    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= capacity)
        return;
    
    uint32_t key = keys[tid];
    uint32_t index = hashCode(key, capacity);

    /**
     * Block Logic: Linear probing search path.
     * Invariant: Search continues until the key matches. Empty slot detection is implicit.
     */
    while (true) {
        if (ht[index].key == key) {
            values[tid] = ht[index].value;
            return;
        }

        index++;
        index %= capacity; 
    }

}

/**
 * @brief CUDA kernel to migrate elements from an old hash table to a new larger one.
 * 
 * @param old_h Source hash table.
 * @param new_h Target hash table.
 * @param old_capacity Capacity of the source.
 * @param capacity Capacity of the target.
 */
__global__ void kernel_reshape(HashNode* old_h, HashNode* new_h, int old_capacity, int capacity) {

    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= old_capacity)
        return;

    uint32_t key = old_h[tid].key;
    uint32_t old_key;
    uint32_t value = old_h[tid].value;
    
    // Logic: Skip empty slots during migration.
    if (key == KEY_INVALID) return;

    uint32_t index = hashCode(key, capacity);

    /**
     * Block Logic: Re-inserts existing elements using the linear probing strategy.
     */
    while (true) {
        old_key = atomicCAS(&new_h[index].key, KEY_INVALID, key);

        if (old_key == key || old_key == KEY_INVALID) {
            new_h[index].value = value;
            return;
        }

        index++;
        index %= capacity; 
    }
}


GpuHashTable::GpuHashTable(int size) {
    /**
     * Functional Utility: Initializes the GPU hash table and clears the device memory.
     */
    this->capacity = size;

    cudaMalloc(&ht, size * sizeof(HashNode));
    if (ht == 0)
        cout << "cudaMalloc failed\n";

    current_size = 0;
    // Functional Utility: Set all keys to 0 (KEY_INVALID) to mark buckets as available.
    cudaMemset(ht, KEY_INVALID, size * sizeof(HashNode));
}


GpuHashTable::~GpuHashTable() {
    /**
     * Functional Utility: Resource cleanup for device-side allocations.
     */
    cudaFree(ht);
}



void GpuHashTable::reshape(int numBucketsReshape) {
    /**
     * Functional Utility: Dynamic re-hashing orchestrator.
     * Logic: Allocates a new buffer, invokes migration kernel, and synchronizes.
     */
    HashNode* newHt;
    uint32_t num_blocks = numBucketsReshape / BLOCK_SIZE;

    if (numBucketsReshape % BLOCK_SIZE)
    {
	    ++num_blocks;
    }

    int old_capacity = capacity;
    this->capacity = numBucketsReshape;

    cudaMalloc(&newHt, numBucketsReshape * sizeof(HashNode));

    if (newHt == 0) {
        return;
    }

    cudaMemset(newHt, 0, numBucketsReshape * sizeof(HashNode));

    /**
     * Execution: Triggers the parallel migration of all existing buckets.
     */
    kernel_reshape<<<num_blocks, BLOCK_SIZE>>>(ht, newHt, old_capacity, capacity);
    cudaDeviceSynchronize();

    cudaFree(ht);
    ht = newHt;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
    /**
     * Functional Utility: High-throughput batch insertion host interface.
     * Logic: Enforces load factor threshold, reshapes if necessary, and dispatches parallel kernel.
     */
    const int num_bytes = numKeys * sizeof(int);
    uint32_t num_blocks = numKeys / BLOCK_SIZE;

    int *device_keys;
    int *device_values;

    if (numKeys % BLOCK_SIZE)
    {
	    ++num_blocks;
    }

    /**
     * Optimization: Triggers expansion if the load factor exceeds 90%.
     */
    if ((float)(current_size + numKeys) / capacity >= 0.9) {
        reshape(int((current_size + numKeys) / 0.8));
    }
    
    cudaMalloc(&device_keys, num_bytes);
    cudaMalloc(&device_values, num_bytes);

    if (device_keys == 0 || device_values == 0) {
        return false;
    }

    cudaMemcpy(device_keys, keys, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_values, values, num_bytes, cudaMemcpyHostToDevice);

    // Execution: Parallel insertion of the entire batch.
    kernel_insert_batch<<<num_blocks, BLOCK_SIZE>>>(ht, device_keys, device_values, numKeys, capacity);
    cudaDeviceSynchronize();

    current_size += numKeys;

    cudaFree(device_keys);
    cudaFree(device_values);
    
    return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
    /**
     * Functional Utility: Batch retrieval of values associated with multiple keys.
     */
    int* device_keys;
    int* values;
    int* device_values;

    const int num_bytes = numKeys * sizeof(int);
    uint32_t num_blocks = numKeys / BLOCK_SIZE;

    if (numKeys % BLOCK_SIZE)
    {
	    ++num_blocks;
    }

    values = (int*) malloc(num_bytes * sizeof(int));

    cudaMalloc(&device_keys, num_bytes);
    cudaMalloc(&device_values, num_bytes);

    if (device_keys == 0  || values == 0) {
        return NULL;
    }

    cudaMemcpy(device_keys, keys, num_bytes, cudaMemcpyHostToDevice);

    // Execution: Parallel lookup.
    kernel_get_batch<<<num_blocks, BLOCK_SIZE>>>(ht, device_keys, device_values, capacity);

    cudaDeviceSynchronize();

    cudaMemcpy(values, device_values, num_bytes, cudaMemcpyDeviceToHost);

    cudaFree(device_keys);
    cudaFree(device_values);

	return values;
}


float GpuHashTable::loadFactor() {
    /**
     * Functional Utility: Computes the current occupancy ratio.
     */
    if (this->capacity == 0)
        return 0;

	return (float) (this->current_size / this->capacity);
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
#define BLOCK_SIZE 		1024

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
 * Data Storage: Prime numbers for universal hashing.
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
 * @struct HashNode
 * @brief Descriptor for a single bucket in the hash table.
 */
struct HashNode 
{
	uint32_t key;
	uint32_t value;
};

class GpuHashTable
{

	public:
		HashNode *ht;
	
        int capacity;
        int current_size;

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
