
/**
 * @file gpu_hashtable.cu
 * @brief High-performance CUDA-based hash table using linear probing and atomic synchronization.
 * 
 * This module provides a parallel hash table implementation designed for GPU global memory.
 * It handles collisions via linear probing and ensures atomicity during concurrent insertions
 * using the atomicCAS primitive.
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
 * @brief Device-side hash function using a linear congruential mapping with large prime constants.
 * 
 * @param key The input key to be hashed.
 * @param capacity The total number of buckets available in the table.
 * @return int A bucket index within the range [0, capacity-1].
 */
__device__ int getHashCode(int key, int capacity) {
	return ((long long)abs(key) * 1646237llu) % 67965551447llu % capacity;
}

/**
 * @brief Parallel kernel for batch insertion of key-value pairs.
 * 
 * Algorithm: Linear probing with atomic ownership resolution.
 * Invariant: A slot is unoccupied if its key is 0.
 * 
 * @param keys Array of input keys.
 * @param values Array of corresponding values.
 * @param N Number of elements in the current batch.
 * @param hashtable Descriptor for the GPU-resident hash table.
 */
__global__ void kernel_insert(int *keys, int *values, int N, HashTable hashtable) {
	/**
	 * Thread Indexing: Maps 1D grid/block coordinates to the input batch.
	 */
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	// Pre-condition: Thread index must be within the batch size.
	if (i < N) {
		int idx = getHashCode(keys[i], hashtable.capacity);

		/**
		 * Block Logic: Primary probe path (from hash index to end of table).
		 * Synchronization: atomicCAS ensures only one thread claims an empty bucket.
		 */
		for (int k = idx; k < hashtable.capacity; k++) {
			int old_key = atomicCAS(&hashtable.nodes[k].key, 0, keys[i]);
			if (!old_key || old_key == keys[i]) {
				// Functional Utility: Set the value once the key slot is acquired or verified.
				hashtable.nodes[k].value = values[i];
				return;
			}
		}

		/**
		 * Block Logic: Secondary probe path (circular wrap-around from table start).
		 */
		for (int k = 0; k < idx; k++) {
			int old_key = atomicCAS(&hashtable.nodes[k].key, 0, keys[i]);
			if (!old_key || old_key == keys[i]) {
				hashtable.nodes[k].value = values[i];
				return;
			}
		}
	}
}

/**
 * @brief Parallel kernel to migrate elements from an old hash table to a new resized one.
 * 
 * @param old_hashtable Source hash table.
 * @param new_hashtable Target hash table with higher capacity.
 */
__global__ void kernel_reshape(HashTable old_hashtable, HashTable new_hashtable) {
	/**
	 * Thread Indexing: Maps threads to existing slots in the old table.
	 */
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < old_hashtable.capacity) {
		// Logic: Process only slots that contain active data.
		if (old_hashtable.nodes[i].key != 0) {
			int idx = getHashCode(old_hashtable.nodes[i].key, new_hashtable.capacity);

			/**
			 * Block Logic: Re-inserts the element into the new table using the same linear probing rules.
			 */
			for (int k = idx; k < new_hashtable.capacity; k++) {
				int old_key = atomicCAS(&new_hashtable.nodes[k].key, 0, old_hashtable.nodes[i].key);
	
				if (!old_key) {
					new_hashtable.nodes[k].value = old_hashtable.nodes[i].value;
					return;
				}
			}
	
			for (int k = 0; k < idx; k++) {
				int old_key = atomicCAS(&new_hashtable.nodes[k].key, 0, old_hashtable.nodes[i].key);
	
				if (!old_key) {
					new_hashtable.nodes[k].value = old_hashtable.nodes[i].value;
					return;
				}
			}
		}
	}
}

/**
 * @brief Parallel kernel for batch lookup of keys.
 * 
 * @param keys Array of keys to find.
 * @param values Output array for found values.
 * @param N Number of search operations.
 * @param hashtable Source hash table descriptor.
 */
__global__ void kernel_get(int *keys, int *values, int N, HashTable hashtable) {
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < N) {
		int idx = getHashCode(keys[i], hashtable.capacity);

		/**
		 * Block Logic: Search via linear probing path.
		 * Invariant: If the key exists, it must be on this probing path.
		 */
		for (int k = idx; k < hashtable.capacity; k++) {
			if (hashtable.nodes[k].key == keys[i]) {
				values[i] = hashtable.nodes[k].value;
				return;
			}
		}

		for (int k = 0; k < idx; k++) {
			if (hashtable.nodes[k].key == keys[i]) {
				values[i] = hashtable.nodes[k].value;
				return;
			}
		}
	}

}


GpuHashTable::GpuHashTable(int size) {
	/**
	 * Functional Utility: Initializes the GPU hash table state and allocates global memory.
	 */
	hashtable.size = 0;
	hashtable.capacity = size;

	const int num_bytes = size * sizeof(Node);

	cudaMalloc(&hashtable.nodes, num_bytes);
	if (!hashtable.nodes) {
		return;
	}

	// Invariant: Keys are initialized to 0 to represent empty buckets.
	cudaMemset(hashtable.nodes, 0, num_bytes);
}


GpuHashTable::~GpuHashTable() {
	/**
	 * Functional Utility: Releases GPU memory resources.
	 */
	cudaFree(hashtable.nodes);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	/**
	 * Functional Utility: Dynamic re-hashing orchestrator.
	 * Logic: Scales capacity by 1.25x (1/0.8), migrates existing data, and updates pointers.
	 */
	HashTable new_hashtable;
	new_hashtable.capacity = (int)(numBucketsReshape / 0.8f);
	new_hashtable.size = hashtable.size;

	const int num_bytes = new_hashtable.capacity * sizeof(Node);

	cudaMalloc(&new_hashtable.nodes, num_bytes);
	if (!new_hashtable.nodes) {
		return;
	}

	cudaMemset(new_hashtable.nodes, 0, num_bytes);

	unsigned int blocks_no = hashtable.capacity / 1024;
	if (hashtable.capacity % 1024) {
		++blocks_no;
	}

	/**
	 * Execution: Triggers the data migration from the old buffer to the new one.
	 */
	kernel_reshape<<<blocks_no, 1024>>>(hashtable, new_hashtable);

	// Synchronization: Ensures all elements are moved before the source buffer is freed.
	cudaDeviceSynchronize();

	cudaFree(hashtable.nodes);
	hashtable = new_hashtable;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	/**
	 * Functional Utility: Host-side interface for parallel insertions.
	 */
	const int num_bytes = numKeys * sizeof(int);

	int *device_keys = 0;
	int *device_values = 0;

	cudaMalloc(&device_keys, num_bytes);
	cudaMalloc(&device_values, num_bytes);

	if (!device_keys || !device_values) {
		return false;
	}

	/**
	 * Optimization: Triggers reshape if the resulting load factor would exceed 80%.
	 */
	float new_factor = (float)(hashtable.size + numKeys) / (float)hashtable.capacity;
	if (new_factor > 0.8f) {
		reshape(hashtable.size + numKeys);
	}

	cudaMemcpy(device_keys, keys, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, num_bytes, cudaMemcpyHostToDevice);

	unsigned int blocks_no = numKeys / 1024;
	if (numKeys % 1024) {
		++blocks_no;
	}

	// Execution: Dispatch concurrent insertion threads.
	kernel_insert<<<blocks_no, 1024>>> (device_keys, 
											device_values, 
											numKeys, 
											hashtable);

	cudaDeviceSynchronize();

	hashtable.size += numKeys;

	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	/**
	 * Functional Utility: Host-side interface for parallel retrieval.
	 */
	int *host_values = 0;
	int *device_keys = 0;
	int *device_values = 0;

	const int num_bytes = numKeys * sizeof(int);

	host_values = (int *) malloc(num_bytes);

	cudaMalloc(&device_keys, num_bytes);
	cudaMalloc(&device_values, num_bytes);

	if(!host_values || !device_keys || !device_values) {
		return NULL;
	}

	cudaMemcpy(device_keys, keys, num_bytes, cudaMemcpyHostToDevice);

	unsigned int blocks_no = numKeys / 1024;
	if (numKeys % 1024) {
		++blocks_no;
	}

	// Execution: Parallel lookup execution.
	kernel_get<<<blocks_no, 1024>>> (device_keys,
										device_values, 
										numKeys, 
										hashtable);

	cudaDeviceSynchronize();

	cudaMemcpy(host_values, device_values, num_bytes, cudaMemcpyDeviceToHost);

	cudaFree(device_keys);
	cudaFree(device_values);

	return host_values;
}


float GpuHashTable::loadFactor() {
	/**
	 * Functional Utility: Returns the ratio of used entries to total capacity.
	 */
	return (float)hashtable.size / (float)hashtable.capacity;
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
 * Data Storage: Precomputed primes for universal hash coefficients.
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
 * @struct Node
 * @brief Representation of a single key-value entry.
 */
struct Node {
	int key;
	int value;
};


/**
 * @struct HashTable
 * @brief Container for the GPU-resident bucket array.
 */
struct HashTable {
	int capacity;
	int size;
	Node *nodes;
};




class GpuHashTable
{
	HashTable hashtable;

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
