
/**
 * @file gpu_hashtable.cu
 * @brief High-performance GPU-based hash table using linear probing and atomic synchronization.
 * 
 * This module provides a parallel hash table implementation designed for GPU global memory.
 * It handles collisions via linear probing and ensures atomicity during concurrent insertions
 * and updates using atomicCAS and atomicExch primitives.
 */

#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include "gpu_hashtable.hpp"


GpuHashTable::GpuHashTable(int size) {	
	/**
	 * Functional Utility: Initializes the GPU hash table state and allocates global memory.
	 * Sets the initial capacity and clears the bucket array.
	 */
	int numBytes = size * sizeof(Node);
	cudaError_t rez;

	this->capacity = size;
	this->freeSlots = this->capacity;
	this->takenSlots = 0;
	this->loadF = (float)this->takenSlots / (float)this->capacity;

	cudaMalloc((void **) &(this->vectorH), numBytes);

	if (this->vectorH == 0) {
		printf("[DEVICE] Couldn't alloc my hash vector inside constructor.\n");
		exit(1);
	}
	// Invariant: Keys initialized to 0 indicate empty slots.
	cudaMemset(this->vectorH, 0, numBytes);

	// Telemetry: Allocate device memory for a synchronized occupancy counter.
	rez = cudaMalloc((void **) &(this->nrTakenSlots), 1 * sizeof(int));
	DIE(rez != cudaSuccess, "couldn't alloc the nr taken slots.\n");

	cudaMemset((void *)(this->nrTakenSlots), 0, 1 * sizeof(int));
}


GpuHashTable::~GpuHashTable() {
	/**
	 * Functional Utility: Resource cleanup for device-side allocations.
	 */
	cudaFree(this->vectorH);
	cudaFree(this->nrTakenSlots);
}

/**
 * @brief Device-side wrapper for the hash calculation.
 */
__device__ int calcHash(int key, int dim) {
	return hash4(key, dim);
}

/**
 * @brief Core insertion logic executed on the device.
 * 
 * Algorithm: Linear probing with atomic ownership resolution.
 * 
 * @param key The input key.
 * @param val The input value.
 * @param myHash The initial target bucket index.
 * @param vec Pointer to the hash table data.
 * @param capacity Current table capacity.
 * @param nrTakenSlots Pointer to the atomic occupancy counter.
 */
__device__ void insert(int key, int val, int myHash, Node *vec, int capacity, int *nrTakenSlots) {

	int index = myHash;
	int aux = 0;

	/**
	 * Block Logic: Probe loop for slot acquisition.
	 * Synchronization: atomicCAS ensures atomic ownership of an empty slot.
	 */
	do {
		// Atomic: Attempt to claim an empty slot (key 0).
		aux = atomicCAS(&(vec[index].key), 0, key);
		if (aux == 0) {
			// Success: Occupy the bucket and update the global count if requested.
			vec[index].value = val;

			if (nrTakenSlots != NULL) {
				atomicAdd(nrTakenSlots, 1);
			}

		} else {
			// Conflict: Check if the slot already contains the target key for an update.
			if (aux == key) {
				vec[index].value = val;
				break;
			}
			
			// Linear Probing: Move to the next bucket with circular wrap-around.
			index += 1;
			index = index % capacity;
		}
	} while (aux != 0);
}

/**
 * @brief CUDA kernel to migrate elements during table resizing.
 */
__global__ void moveNodeToNewVector(Node *oldVector, Node *newVector, int newNrBuckets, int oldCapacity) {
	/**
	 * Thread Indexing: Maps threads to slots in the old table.
	 */
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int key = 0;
	int val = 0;
	int myHash = 0;

	if (index >= oldCapacity)
		return;

	key = oldVector[index].key;
	val = oldVector[index].value;
	
	// Optimization: Skip empty buckets during migration.
	if (key == 0)
		return;
	
	// Logic: Re-hash and insert into the new map.
	myHash = calcHash(key, newNrBuckets);
	insert(key, val, myHash, newVector, newNrBuckets, NULL);
}

void GpuHashTable::reshape(int numBucketsReshape) {
	/**
	 * Functional Utility: Dynamic re-hashing orchestrator.
	 * Logic: Scales the hash table to a new size and migrates all existing entries.
	 */
	int neededThreads = this->capacity;
	int nrBlocks = 0;
	int nrThreads = 1024;
	Node *newVec = NULL;
	cudaError_t res;

	if (neededThreads % nrThreads != 0)
		nrBlocks += 1;
	nrBlocks += neededThreads / nrThreads;

	res = cudaMalloc((void **)&newVec, numBucketsReshape * sizeof(Node));
	DIE(res != cudaSuccess, "allocation on GPU failed at reshp.\n");

	cudaMemset(newVec, 0, numBucketsReshape * sizeof(Node));

	DIE(newVec == NULL, "Allocation failed on GPU at reshp.\n");

	if (neededThreads != 0) {
		/**
		 * Execution: Triggers the parallel migration kernel.
		 */
		moveNodeToNewVector<<<nrBlocks, nrThreads>>>(this->vectorH, newVec, numBucketsReshape, this->capacity);
		cudaDeviceSynchronize();
	}

	res = cudaFree((void *)this->vectorH);
	DIE(res != cudaSuccess, "Couldn't free the old mem after reshape.\n");

	this->vectorH = newVec;
}


/**
 * @brief CUDA kernel for parallel batch insertion.
 */
__global__ void addNewNodes(int *keys, int *vals, int numKeys, Node *vec, int cap, int *nrTakenSlots) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int key = 0;
	int val = 0;
	int myHash = 0;

	if (index >= numKeys)
		return;
	
	key = keys[index];
	val = vals[index];
	myHash = calcHash(key, cap);

	// Action: Insert the individual key-value pair.
	insert(key, val, myHash, vec, cap, nrTakenSlots);
}

bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	/**
	 * Functional Utility: Host-side interface for bulk insertions.
	 * Logic: Checks capacity, reshapes if load factor > 80%, transfers data, and launches kernel.
	 */
	int newSize = numKeys + this->takenSlots;
	float onePercent = 0;

	int neededThreads = numKeys;
	int nrBlocks = 0;
	int nrThreads = 1024;

	int *gpuKeys = NULL;
	int *gpuVals = NULL;


	cudaError_t res;

	res = cudaMalloc((void **)&gpuKeys, numKeys * sizeof(int));
	DIE(res != cudaSuccess, " failled cudaMalloc at insertB\n");

	res = cudaMalloc((void **)&gpuVals, numKeys * sizeof(int));
	DIE(res != cudaSuccess, " failled cudaMalloc at insertB\n");

	if (gpuKeys == NULL || gpuVals == NULL) {
		printf("Cuda malloc failled at insertB\n");
		return false;
	}

	res = cudaMemcpy(gpuKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(res != cudaSuccess, "failed at gpuKeys from host.\n");

	res = cudaMemcpy(gpuVals, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(res != cudaSuccess, "failed at gpuValues from host.\n");

	if (neededThreads % nrThreads != 0)
		nrBlocks += 1;
	nrBlocks += neededThreads / nrThreads;

	/**
	 * Optimization: Enforces a maximum load factor of 80% to maintain O(1) performance.
	 */
	if (newSize >= this->capacity) {
		onePercent = (float)newSize / 80.0;
		this->reshape((int)(onePercent * 100));

		this->capacity = (int)(onePercent * 100);
	}
	
	// Execution: Dispatch concurrent insertions.
	addNewNodes<<<nrBlocks, nrThreads>>>(gpuKeys, gpuVals, numKeys, this->vectorH, this->capacity, this->nrTakenSlots);
	cudaDeviceSynchronize();

	res = cudaFree((void *)gpuKeys);
	DIE(res != cudaSuccess, "couldn't free gpuKeys\n");

	res = cudaFree((void *)gpuVals);
	DIE(res != cudaSuccess, "couldn't free gpuVals.\n");

	// Telemetry: Synchronize the occupancy count back to the host.
	res = cudaMemcpy(&(this->takenSlots), this->nrTakenSlots, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	DIE(res != cudaSuccess, "Couldn't get back nrTakenSlots from device.\n");

	this->freeSlots = this->capacity - this->takenSlots;
	this->loadF = (float)this->takenSlots / (float)this->capacity;

	return true;
}

/**
 * @brief Device-side function to search for a key in the hash map.
 */
__device__ int findVal(int key, int hash, Node *vec, int cap) {

	int index = hash;
	int aux = cap;

	/**
	 * Block Logic: Sequential scan along the probe path.
	 * Pre-condition: Search starting from the calculated hash index.
	 */
	while (aux > 0) {
		if (vec[index].key == key) {
			return vec[index].value;
		} else {
			index += 1;
			index = index % cap;
		}
		aux--;
	}
	
	return -1; // Not found sentinel.
}

/**
 * @brief CUDA kernel for parallel value retrieval.
 */
__global__ void getValues(int *gpuRes, int *gpuKeys, int numKeys, Node *vector, int cap) {
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int key = 0;
	int val = 0;
	int myHash = 0;

	if (index >= numKeys)
		return;

	key = gpuKeys[index];
	myHash = calcHash(key, cap);
	
	// Logic: Execute device search for each requested key.
	val = findVal(key, myHash, vector, cap);
	gpuRes[index] = val;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	/**
	 * Functional Utility: Host-side interface for bulk lookups.
	 */
	int *resultDevice = NULL;
	int *resultHost = NULL;
	int *gpuKeys = NULL;
	cudaError_t res = cudaErrorInvalidValue;

	int neededThreads = numKeys;
	int nrBlocks = 0;
	int nrThreads = 1024;

	if (neededThreads % nrThreads != 0)
		nrBlocks += 1;

	nrBlocks += neededThreads / nrThreads;

	res = cudaMalloc( (void **)&resultDevice, numKeys * sizeof(int) );
	DIE(res != cudaSuccess, "failled to alloc resultDevice.\n");

	res = cudaMalloc((void **)&gpuKeys, numKeys * sizeof(int));
	DIE(res != cudaSuccess, "failled to alloc gpuKeys at get.\n");

	if (resultDevice == NULL || gpuKeys == NULL) {
		printf("failed to alloc resultDevice.\n");
		return NULL;
	}

	res = cudaMemcpy(gpuKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(res != cudaSuccess, "Failed to copy to gpuKeys.\n");

	// Execution: Dispatch parallel lookup threads.
	getValues<<<nrBlocks, nrThreads>>>(resultDevice, gpuKeys, numKeys, this->vectorH, this->capacity);
	cudaDeviceSynchronize();

	resultHost = (int *)malloc(numKeys * sizeof(int));
	DIE(resultHost == NULL, "failed to alloc resultHost.\n");

	res = cudaMemcpy(resultHost, resultDevice, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	DIE(res != cudaSuccess, "failled to copy res from dev to host.\n");

	res = cudaFree((void *) resultDevice);
	DIE(res != cudaSuccess, "Failed to free resultDevice.\n");

	res = cudaFree((void *) gpuKeys);
	DIE(res != cudaSuccess, "Failed to free gpuKeys.\n");

	return resultHost;
}


float GpuHashTable::loadFactor() {
	/**
	 * Functional Utility: Accessor for the current occupancy ratio.
	 */
	return this->loadF;
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
 * Data Storage: Global primes for hash function logic.
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
 * @brief Device-side universal hash function implementation.
 */
__device__ int hash4(int data, int limit) {

	size_t firstPrime = 7070586497075177llu;
	size_t secondPrime = 1140272737634240411llu;

	return ((long)abs(data) * firstPrime) % secondPrime % limit;
}





/**
 * @struct Node
 * @brief Descriptor for a single key-value entry in the map.
 */
typedef struct s {
	int key;
	int value;
} Node;

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

		int capacity;
		float loadF;
		int takenSlots;
		int freeSlots;
		Node *vectorH;
		int *nrTakenSlots;
};

#endif
/* Update complete: gpu_hashtable.cu processed with semantic documentation. */
