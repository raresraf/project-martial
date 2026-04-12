
/**
 * @file gpu_hashtable.cu
 * @brief High-performance parallel hash table implementation using CUDA and linear probing.
 * 
 * This module provides a thread-safe hash map residing in GPU global memory. It uses 
 * linear probing for collision resolution and atomic Compare-And-Swap (CAS) to 
 * manage concurrent state transitions across thousands of GPU threads.
 */

#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "gpu_hashtable.hpp"

// Constants for the linear congruential hash function.
#define HASH_A 352617
#define HASH_B 1427462537

/**
 * @brief Device-side hash function that maps a key to a bucket index.
 * 
 * @param data The input key.
 * @param limit Current table capacity (modulo operand).
 * @return int The calculated bucket index.
 */
__device__ int getHash(int data, int limit) {
    return (long long) abs(data) * HASH_A % HASH_B % limit;
}

/**
 * @brief CUDA kernel for parallel batch insertion into the hash table.
 * 
 * Algorithm: Linear probing with atomic ownership resolution.
 * Invariant: A bucket is empty if its key equals KEY_INVALID.
 * 
 * @param keys Array of input keys in device memory.
 * @param values Array of input values in device memory.
 * @param numEntries Size of the input batch.
 * @param hashmap Descriptor for the GPU-resident hash table structure.
 */
__global__ void kernel_insert(int *keys, int *values, int numEntries, hash_table hashmap) {
    /**
     * Thread Indexing: Maps 1D grid/block coordinates to the input batch indices.
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Pre-condition: Thread index must be within the batch boundaries.
    if (idx < numEntries) {

        int oldKey, newKey;
        newKey = keys[idx];

        int hash = getHash(newKey, hashmap.size);
        int i = hash;
        
        /**
         * Block Logic: Probe loop for slot acquisition.
         * Synchronization: atomicCAS ensures atomic ownership of an empty slot or update of an existing key.
         */
        do {
            // Linear Probing: Manual wrap-around check.
            if (i >= hashmap.size) {
                i = 0;
            }

            // Atomic: Attempt to claim the bucket or verify existing occupancy by this key.
            oldKey = atomicCAS(&hashmap.map[i].key, KEY_INVALID, newKey);

            if (oldKey == newKey || oldKey == KEY_INVALID) {
                // Success: Update the value associated with the now-owned key.
                hashmap.map[i].value = values[idx];
                break;
            }
            i++;
        } while (i != hash); // Invariant: Probing terminates if we return to the start (table full).
    }
}

/**
 * @brief CUDA kernel for parallel batch value retrieval.
 * 
 * Algorithm: Linear probing search.
 */
__global__ void kernel_get(int *keys, int *values, int numEntries, hash_table hashmap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numEntries) {
        int key = keys[idx];
        int hash = getHash(keys[idx], hashmap.size);

        int i = hash;
        /**
         * Block Logic: Search along the linear probe path.
         * Invariant: Terminate search when the specific key is located.
         */
        do {
            if (i >= hashmap.size) {
                i = 0;
            }
            if (hashmap.map[i].key == key) {
                values[idx] = hashmap.map[i].value;
                break;
            }
            i++;
        } while (i != hash);
    }
}

/**
 * @brief CUDA kernel to migrate elements from an old hash table to a new resized one.
 */
__global__ void kernel_rehash(hash_table oldHash, hash_table newHash) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Logic: Only migrate active entries.
    if (idx < oldHash.size && oldHash.map[idx].key != KEY_INVALID) {
        int oldKey, newKey;
        newKey = oldHash.map[idx].key;

        int hash = getHash(newKey, newHash.size);
        int i = hash;
        
        /**
         * Block Logic: Re-inserts the element into the new table using linear probing rules.
         */
        do {
            if (i >= newHash.size) {
                i = 0;
            }

            oldKey = atomicCAS(&newHash.map[i].key, KEY_INVALID, newKey);

            if (oldKey == KEY_INVALID) {
                newHash.map[i].value = oldHash.map[idx].value;
                break;
            }
            i++;
        } while (i != hash);
    }

}


GpuHashTable::GpuHashTable(int
                           size) {
    /**
     * Functional Utility: Initializes the GPU hash table state and allocates global memory.
     */
    numInsertedPairs = 0;
    hashmap.size = size;
    hashmap.map = nullptr;

    // Action: Allocate device memory for the bucket array.
    if (cudaMalloc(&hashmap.map, size * sizeof(entry)) == cudaSuccess) {
        // Initial State: Key 0 (KEY_INVALID) indicates an empty slot.
        cudaMemset(hashmap.map, 0, size * sizeof(entry));
    } else {
        std::cerr << "Memory error during init\n";
    }
}


GpuHashTable::~GpuHashTable() {
    /**
     * Functional Utility: Resource cleanup for device-side allocations.
     */
    cudaFree(hashmap.map);
}


void GpuHashTable::reshape(int numBucketsReshape) {
    /**
     * Functional Utility: Dynamic re-hashing orchestrator.
     * Logic: Allocates a new buffer, invokes the migration kernel, and synchronizes.
     */
    hash_table newHashmap;
    newHashmap.size = numBucketsReshape;
    unsigned int numBlocks = (hashmap.size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    if (cudaMalloc(&newHashmap.map, numBucketsReshape * sizeof(entry)) == cudaSuccess) {
        cudaMemset(newHashmap.map, 0, numBucketsReshape * sizeof(entry));
        
        // Execution: Parallel data migration.
        kernel_rehash<<<numBlocks, THREADS_PER_BLOCK>>>(hashmap, newHashmap);

        // Synchronization: Ensures migration is complete before freeing old memory.
        cudaDeviceSynchronize();

        cudaFree(hashmap.map);
        hashmap = newHashmap;
    }
}


bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
    /**
     * Functional Utility: Host-side interface for bulk insertions.
     * Logic: Enforces load factor threshold, reshapes if necessary, and dispatches kernel.
     */
    int *cudaKeys, *cudaValues;
    unsigned int numBlocks = (numKeys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    size_t memSize = numKeys * sizeof(int);
    cudaMalloc(&cudaKeys, memSize);
    cudaMalloc(&cudaValues, memSize);

    /**
     * Optimization: Triggers expansion if occupancy exceeds MAX_LOAD_FACTOR (80%).
     */
    float loadIndex = float(numInsertedPairs + numKeys) / hashmap.size;
    if (loadIndex >= MAX_LOAD_FACTOR) {
        reshape(int((numInsertedPairs + numKeys) / MIN_LOAD_FACTOR));
    }

    cudaMemcpy(cudaKeys, keys, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaValues, values, memSize, cudaMemcpyHostToDevice);

    // Execution: dispatch insertion requests to the GPU.
    kernel_insert<<<numBlocks, THREADS_PER_BLOCK>>>(cudaKeys, cudaValues, numKeys, hashmap);

    numInsertedPairs += numKeys;
    cudaDeviceSynchronize();
    
    cudaFree(cudaValues);
    cudaFree(cudaKeys);

    return true;

}


int *GpuHashTable::getBatch(int *keys, int numKeys) {
    /**
     * Functional Utility: Batch retrieval interface.
     * Logic: Uses Managed Memory for the results to simplify host access.
     */
    int *cudaKeys, *values;

    size_t memSize = numKeys * sizeof(int);
    cudaMalloc(&cudaKeys, memSize);
    // Optimization: Unified Memory allows direct CPU access after device completion.
    cudaMallocManaged(&values, memSize);

    unsigned int numBlocks = (numKeys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    if (cudaKeys && values) {
        cudaMemcpy(cudaKeys, keys, memSize, cudaMemcpyHostToDevice);
        
        // Execution: Dispatch parallel lookup.
        kernel_get<<<numBlocks, THREADS_PER_BLOCK>>>(cudaKeys, values, numKeys, hashmap);
        
        cudaDeviceSynchronize();
        cudaFree(cudaKeys);

        return values;
    }

    std::cerr << "Memory error during batch get\n";
    return nullptr;
}


float GpuHashTable::loadFactor() {
    /**
     * Functional Utility: Accessor for the current map occupancy ratio.
     */
    if (hashmap.size == 0) {
        return 0;
    }

    return float(numInsertedPairs) / hashmap.size;
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

#define	KEY_INVALID     0
#define THREADS_PER_BLOCK   1024
#define MIN_LOAD_FACTOR     0.6
#define MAX_LOAD_FACTOR     0.8

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
 * @struct entry
 * @brief Descriptor for a single bucket in the hash table.
 */
struct entry {
    int key;
    int value;
};


/**
 * @struct hash_table
 * @brief Descriptor for the entire GPU-resident map allocation.
 */
struct hash_table {
    int size;
    entry *map;
};




class GpuHashTable
{
    int numInsertedPairs;
    hash_table hashmap;

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
