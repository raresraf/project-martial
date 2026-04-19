/**
 * @file gpu_hashtable.cu
 * @brief High-occupancy CUDA Hash Table with XOR-based hashing.
 * 
 * This module implements a parallel hash table using a custom XOR-shift hash 
 * function and linear probing for collision resolution. It employs a proactive 
 * resizing strategy based on configurable load factor thresholds and utilizes 
 * atomic Compare-And-Swap (CAS) to manage concurrent state transitions.
 * 
 * Algorithm: Open addressing with linear probing and XOR-shift hashing.
 * Memory Model: Global memory (cudaMalloc) for entry arrays and Managed Memory for occupancy.
 * Domain: HPC, Parallel Data Structures.
 */

#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include "gpu_hashtable.hpp"
#define THREADS_BLOCK 1024
#define MIN_LOAD_FACTOR 75
#define MAX_LOAD_FACTOR 95



/**
 * @brief Device-side hash function using XOR-shift bitwise operations.
 */
__device__ int getHash(int a, int limit)
{
	a ^= (a << 13);
	a ^= (a >> 17);
	a ^= (a << 5);
	return (unsigned int) a % limit;
}




/**
 * @brief CUDA kernel for parallel data migration.
 * 
 * functional Utility: Transfers and re-hashes existing mappings into a new, 
 * larger memory buffer.
 */
__global__ void kernel_rehash(hashBucket* hashTable, hashBucket* newHash, int size, int oldSize) {
	
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= oldSize)
		return;
	hashBucket buck = hashTable[index];
	int key = buck.bucket.key;
	
	if (key == 0)
		return;
	int value = buck.bucket.value;
	int hash = getHash(key, size);
	
	// Block Logic: Linear re-insertion search.
	for (int j = 0; j < size; j++) {
			int lastKey = atomicCAS(&newHash[hash].bucket.key, 0, key);
			if (lastKey == 0 || lastKey == key) {
				newHash[hash].bucket.value = value;
				return;
			}
			hash = (hash + 1) % size;
	}
}



/**
 * @brief CUDA kernel for parallel entry retrieval.
 */
__global__ void kernel_getBatch(hashBucket* hashTable, int* key, int* value, int noKeys, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= noKeys)
		return;
	if (key[index] <= 0) {
		value[index] = -2;
		return;
	}
	int trial = 0;
	int hash = getHash(key[index], size);
	
	/**
	 * Block Logic: Linear search traversal.
	 */
	for (int j = 0; j < size; j++) {
		hashBucket buck = hashTable[hash];
		if (buck.bucket.key == key[index] && buck.bucket.value != 0) {
			
			value[index] = buck.bucket.value;
			return;
		}
		


		hash = (hash + 1) % size;
	}
}

/**
 * @brief CUDA kernel for parallel batch insertion.
 * 
 * Functional Utility: Claims slots via atomicCAS and tracks unique insertions 
 * via a shared occupancy counter.
 * 
 * @param addToThis Managed memory counter for successful insertions.
 */
__global__ void kernel_inserBatch(hashBucket* hashTable, int* key, int* value, int noKeys, int size, int* addToThis) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index >= noKeys)
		return;
	int hash = getHash(key[index], size);
	int lastKey;
	
	/**
	 * Block Logic: Atomic insertion probe loop.
	 */
	while (true) {
		
		lastKey = atomicCAS(&hashTable[hash].bucket.key, 0, key[index]);
		if (lastKey == 0 || lastKey == key[index]) {
			
			if (lastKey == 0)
				// Logic: Increment occupancy if a new slot was initialized.
				atomicAdd(addToThis, 1);
			hashTable[hash].bucket.value = value[index];
			
			return;
		}
		hash = (hash + 1) % size;
	}
}


/**
 * @brief Constructor: Initializes table storage and managed size counter.
 */
GpuHashTable::GpuHashTable(int size) {
	capacity = size * 100 / MIN_LOAD_FACTOR;
	// Memory Hierarchy: Global memory allocation for entries.
	cudaMalloc(&hashTable, capacity * sizeof(hashBucket));
	// Memory Hierarchy: Managed memory (Unified) for host-accessible occupancy tracking.
	cudaMallocManaged((void**)&this->size, sizeof(int));
	*(this->size) = 0;

}


/**
 * @brief Destructor: Frees allocated device resources.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashTable);
	cudaFree(size);
}


/**
 * @brief Resizes the hash table using parallel re-hashing.
 */
void GpuHashTable::reshape(int numBucketsReshape) {


	hashBucket* newHash;
	int newCapacity = numBucketsReshape * 100 / MIN_LOAD_FACTOR;

	if (newCapacity capacity)
		return;

	cudaMalloc((void**)&newHash, newCapacity * sizeof(hashBucket));

	unsigned int blocks = newCapacity / THREADS_BLOCK;

	if (newCapacity % THREADS_BLOCK != 0)
		blocks++;

	// Synchronization: Ensures all mappings are migrated before cleaning up old memory.
	kernel_rehash >> (this->hashTable, newHash, newCapacity, this->capacity);


	cudaDeviceSynchronize();

	this->capacity = newCapacity;

	cudaFree(hashTable);

	hashTable = newHash;

}


/**
 * @brief Performs batch parallel insertion.
 * 
 * Logic: Transfers batch to GPU and manages expansion if load factor > 95%.
 */
bool GpuHashTable::insertBatch(int* keys, int* values, int numKeys) {

	// Adaptive Scaling: Monitors density to maintain performance.
	float loadF = 1.0f * (*(this->size) + numKeys) / capacity * 100;

	if (loadF > MAX_LOAD_FACTOR) {
		reshape(*(this->size) + numKeys);
	}

	int* deviceKeys = 0;
	int* deviceValues = 0;


	cudaMalloc((void**)&deviceKeys, numKeys * sizeof(int));
	cudaMalloc((void**)&deviceValues, numKeys * sizeof(int));


	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);


	unsigned int blocks = numKeys / THREADS_BLOCK;

	if (numKeys % THREADS_BLOCK != 0)
		blocks++;
	kernel_inserBatch >> (hashTable, deviceKeys, deviceValues, numKeys, capacity, size);


	cudaDeviceSynchronize();

	cudaFree(deviceValues);
	cudaFree(deviceKeys);

	return true;
}


/**
 * @brief Performs batch parallel retrieval using Unified Memory for results.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int* deviceKeys, * values;

	cudaMalloc((void**)&deviceKeys, numKeys * sizeof(int));



	cudaMallocManaged((void**)&values, numKeys * sizeof(int));


	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	unsigned int blocks = numKeys / THREADS_BLOCK;

	if (numKeys % THREADS_BLOCK != 0)
		blocks++;

	kernel_getBatch>> (hashTable, deviceKeys, values, numKeys, capacity);


	cudaDeviceSynchronize();

	cudaFree(deviceKeys);
	return values;
}


/**
 * @brief Returns the current table density.
 */
float GpuHashTable::loadFactor() {
	return (float)((float)(*(this->size))/ ((float)this->capacity));	
}


// ... hash functions and structures ...

/**
 * @struct hashElement
 * @brief Atomic storage unit for a key-value mapping on the GPU.
 */
typedef struct hashE {
	unsigned int key;
	unsigned int value;
} hashElement;

/**
 * @struct hashBucket
 * @brief Wrapper for individual hash table slots.
 */
typedef struct hashB {
	hashElement bucket;
} hashBucket;

/**
 * @class GpuHashTable
 * @brief Host controller for Managing the GPU hash mapping lifecycle.
 */
class GpuHashTable
{
public:
	hashBucket* hashTable;  // Device entry array.
	int* size;              // Managed pointer to occupancy count.
	int capacity;           // Current buffer capacity.
	GpuHashTable(int size);
	void reshape(int sizeReshape);

	bool insertBatch(int* keys, int* values, int numKeys);
	int* getBatch(int* key, int numItems);

	float loadFactor();
	void occupancy();
	void print(string info);

	~GpuHashTable();
};


