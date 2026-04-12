
/**
 * @file gpu_hashtable.cu
 * @brief High-performance parallel hash table implementation using CUDA and Unified Memory.
 * 
 * This module implements a thread-safe hash map residing in GPU memory, accessible via
 * unified memory addressing. It uses linear probing for collision resolution and 
 * atomic operations (atomicCAS, atomicExch) to manage concurrent state transitions.
 */

#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "gpu_hashtable.hpp"

// Global state descriptors for the hash table instance.
int mapSize;
int *currentNo;
size_t *copyList = NULL;
struct realMap *hashMap = NULL;

/**
 * @brief CUDA kernel for parallel batch retrieval of values associated with a set of keys.
 * 
 * Algorithm: Linear probing search.
 * 
 * @param keys Array of search keys in device memory.
 * @param numKeys Number of keys in the lookup batch.
 * @param myMap Pointer to the hash table in global/unified memory.
 * @param values Output array for retrieved values.
 * @param num Maximum capacity of the hash table.
 * @param list Pointer to hash function prime coefficients.
 */
__global__ void kernel_get(int* keys, int numKeys, struct realMap *myMap,
	int* values, int num, size_t *list) {

	/**
	 * Thread Indexing: Maps 1D grid/block indices to search key indices.
	 */
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	// Pre-condition: Thread index must be within the batch boundaries.
	if(i >= numKeys)
		return;

	int index = hash1(keys[i], num, list);

	/**
	 * Block Logic: Probe loop for key location.
	 * Invariant: Search continues along the linear probe path until the key is found.
	 */
	while(1) {
		if(myMap[index].key == keys[i]) {
			values[i] = myMap[index].value;
			break;
		} else {
			// Linear probing: Circular increment.
			index ++;
			if(index >= num)
				index = 0;
		}
	}
}

/**
 * @brief CUDA kernel for concurrent batch insertion of key-value pairs.
 * 
 * Algorithm: Linear probing with atomic ownership resolution.
 * 
 * @param keys Array of input keys.
 * @param numKeys Batch size.
 * @param values Array of corresponding values.
 * @param myMap Target hash map buffer.
 * @param num Current capacity.
 * @param list Prime coefficients for hashing.
 * @param current Global counter for unique elements added.
 */
__global__ void kernel_insert(int* keys, int numKeys, int* values,
	struct realMap *myMap, int num, size_t *list, int* current) {

	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i >= numKeys)
		return;

	int index = hash1(keys[i], num, list);

	/**
	 * Block Logic: Parallel slot acquisition and value assignment.
	 * Synchronization: Uses atomicCAS to claim a key slot and atomicExch for value updates.
	 */
	while(1) {
		// Optimization: Check for existing key first to support updates.
		if(myMap[index].key == keys[i]) {
			atomicExch(&myMap[index].value, values[i]);
			return;
		}
		
		// Atomic: Attempt to claim an empty slot (key 0).
		int result = atomicCAS(&myMap[index].key, 0, keys[i]);

		if(result == 0) {
			// Success: We claimed the slot. Now set the value atomically.
			atomicExch(&myMap[index].value, values[i]);
			if(current != NULL)
				atomicAdd(current, 1);
			break;
		}
		else {
			// Conflict: Slot occupied by another key, continue probing.
			index ++;
			if(index >= num)
				index = 0;
		}
	}
}


GpuHashTable::GpuHashTable(int size) {
	/**
	 * Functional Utility: Initializes the GPU hash table using Unified Memory.
	 * Allows the CPU to zero-initialize the data while the GPU executes kernels on it.
	 */
	cudaMallocManaged(&hashMap, size * sizeof(struct realMap));
	cudaDeviceSynchronize();

	DIE(hashMap == NULL, "allocation for hashMap failed");
	mapSize = size;

	// Optimization: Global counter in unified memory for easy synchronization between host and device.
	cudaMallocManaged(&currentNo, sizeof(int));
	cudaDeviceSynchronize();
	DIE(currentNo == NULL, "allocation for currentNo failed");

	// Initial State: Key 0 indicates an empty bucket.
	for(int i = 0; i < size; i ++) {
		hashMap[i].key = 0;
		hashMap[i].value = 0;
	}

	cudaMallocManaged(&copyList, count() * sizeof(size_t));
	cudaDeviceSynchronize();

	// Telemetry: Copy hash coefficients to device-accessible unified memory.
	cudaMemcpy(copyList, primeList, count() * sizeof(size_t), cudaMemcpyHostToDevice);
}


GpuHashTable::~GpuHashTable() {
	/**
	 * Functional Utility: Cleanup shared unified memory allocations.
	 */
	cudaFree(hashMap);
	cudaFree(currentNo);
	cudaFree(copyList);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	/**
	 * Functional Utility: Dynamic re-hashing orchestrator.
	 * Logic: 
	 * 1. Allocates a new larger unified memory map.
	 * 2. Extracts existing keys and values from the old map.
	 * 3. Re-inserts extracted pairs into the new map via parallel kernel.
	 */
	struct realMap *hashMapCopy = 0;
	cudaMallocManaged(&hashMapCopy, numBucketsReshape * sizeof(struct realMap));
	cudaDeviceSynchronize();

	int *keyCopy;
	cudaMallocManaged(&keyCopy, *currentNo * sizeof(int));
	cudaDeviceSynchronize();

	int *valuesCopy;
	cudaMallocManaged(&valuesCopy, *currentNo * sizeof(int));
	cudaDeviceSynchronize();

	for(int i = 0; i < numBucketsReshape; i ++) {
		hashMapCopy[i].key = 0;
		hashMapCopy[i].value = 0;
	}

	int x = 0;

	// Extraction: Collect all active entries into a compact batch for re-insertion.
	for(int i = 0; i < mapSize; i ++) {
		if(hashMap[i].key != 0) {
			keyCopy[x] = hashMap[i].key;
			valuesCopy[x] = hashMap[i].value;
			x ++;
		}
	}

	// Execution: Re-insert batch into the new map.
	kernel_insert<<< (x + 1023) / 1024, 1024 >>>(keyCopy, x, valuesCopy, hashMapCopy,
		numBucketsReshape, copyList, NULL);
	cudaDeviceSynchronize();

	cudaFree(hashMap);

	hashMap = hashMapCopy;
	mapSize = numBucketsReshape;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	/**
	 * Functional Utility: Host-side interface for parallel batch insertion.
	 */
	// Optimization: Proactive resizing if the operation would exceed current capacity.
	if(*currentNo + numKeys >= mapSize) {
		reshape(*currentNo + numKeys + (*currentNo + numKeys) * 5 / 100);
		mapSize = *currentNo + numKeys + (*currentNo + numKeys) * 5 / 100;
	}

	int *keyCopy;
	cudaMallocManaged(&keyCopy, numKeys * sizeof(int));
	cudaDeviceSynchronize();

	int *valuesCopy;
	cudaMallocManaged(&valuesCopy, numKeys * sizeof(int));
	cudaDeviceSynchronize();

	// Data preparation: Move host data to unified buffers.
	memcpy(keyCopy, keys, numKeys * sizeof(int));
	memcpy(valuesCopy, values, numKeys * sizeof(int));

	// Execution: Dispatch insertion kernel.
	kernel_insert<<< (numKeys + 1023) / 1024, 1024 >>>(keyCopy, numKeys, valuesCopy,
		hashMap, mapSize, copyList, currentNo);
	cudaDeviceSynchronize();

	return false;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	/**
	 * Functional Utility: Host-side interface for parallel batch retrieval.
	 */
	int* results = NULL;
	cudaMallocManaged(&results, numKeys * sizeof(int));
	cudaDeviceSynchronize();

	int *keyCopy;
	cudaMallocManaged(&keyCopy, numKeys * sizeof(int));
	cudaDeviceSynchronize();

	memcpy(keyCopy, keys, numKeys * sizeof(int));

	// Execution: Parallel search dispatch.
	kernel_get<<< (numKeys + 1023) / 1024, 1024 >>>(keyCopy, numKeys, hashMap,
		results, mapSize, copyList);
	cudaDeviceSynchronize();

	return results;
}


float GpuHashTable::loadFactor() {
	/**
	 * Functional Utility: Returns the ratio of occupied buckets to total capacity.
	 */
	return (float)(*currentNo) / (float)(mapSize);
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

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
 * Data Storage: Global prime list used for hashing.
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
 * @struct realMap
 * @brief Representation of a single hash table bucket.
 */
struct realMap {
	int key;
	int value;
};

/**
 * @brief Host/Device hash function implementation.
 */
__host__ __device__  int hash1(int data, int limit, size_t *primeList1) {
	return ((long)abs(data) * primeList1[64]) % primeList1[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

/**
 * @brief Utility to calculate the number of elements in primeList.
 */
int count() {
	int i = 0;

	while(primeList[i] != '\0') {
		i ++;
	}
	return i + 1;
}

class GpuHashTable
{
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
