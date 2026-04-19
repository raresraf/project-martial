/**
 * @file gpu_hashtable.cu
 * @brief Occupancy-optimized GPU Hash Table with automated thread configuration.
 * 
 * This module implements a high-performance hash table that leverages CUDA's occupancy 
 * API to dynamically determine optimal block sizes. It features a robust single-hash 
 * linear probing algorithm with atomic operations and host-device synchronized 
 * key tracking.
 * 
 * Algorithm: Open addressing with linear probing and occupancy optimization.
 * Memory Model: Global memory (cudaMalloc) for map storage and device-side counters.
 * Domain: HPC, Parallel Data Structures.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief Device-side hash function utilizing prime constants.
 */
__device__ int createHash(int data, int limit) {
	return ((long long) abs(data) * PARAM_1_HASH) % PARAM_2_HASH % limit;
}


/**
 * @brief CUDA kernel for table initialization.
 * 
 * Functional Utility: Zeroes out the key fields in the hash map to prepare 
 * for concurrent insertions.
 */
__global__ void initHashT(HashTable hashTable)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= hashTable.capacity)
		return;

	hashTable.map[idx].key = KEY_INVALID;
}


/**
 * @brief Helper device function for atomic insertion of a single mapping.
 * 
 * Logic: Uses linear probing with atomicCAS to find an empty or matching slot.
 * @return 1 if a new slot was claimed, 0 if it was an update or failed.
 */
__device__ int insertOneItem(HashTable hashTable, int newKey, int newValue)
{
	int n = 0;
	int existingKey;
	int startPos = createHash(newKey, hashTable.capacity);

	
	/**
	 * Block Logic: Linear probe search.
	 * Invariant: Traverses the table until an atomic reservation succeeds.
	 */
	for (int i = 0; i < hashTable.capacity; ++i) {
    	
    	existingKey = atomicCAS(&hashTable.map[startPos].key, KEY_INVALID, newKey);
    	if (existingKey == KEY_INVALID || existingKey == newKey) {
    		
    		hashTable.map[startPos].value = newValue;
    		if (existingKey == KEY_INVALID) {
    			
    			n = 1;
    		}
    		break;
    	}
    	
		if (startPos == hashTable.capacity - 1)
			startPos = 0;
		else
			startPos++;
	}
	return n;
}


/**
 * @brief CUDA kernel for parallel batch insertion.
 */
__global__ void insert(HashTable hashTable, int *deviceKeys, int *deviceValues,
											 int numKeys, int *deviceExistKeys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numKeys)
		return;

	
    int ret = insertOneItem(hashTable, deviceKeys[idx], deviceValues[idx]);
    // Logic: Atomically tracks global occupancy increment.
    atomicAdd(&(*deviceExistKeys), ret);
}


/**
 * @brief CUDA kernel for parallel entry lookup.
 */
__global__ void get(HashTable hashTable, int *deviceKeys, int *deviceValues, int numKeys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numKeys)
		return;
    
    int searchedKey = deviceKeys[idx];
    int startPos = createHash(searchedKey, hashTable.capacity);

    
    /**
     * Block Logic: Linear search traversal.
     */
    for (int i = 0; i < hashTable.capacity; ++i) {
    	if (hashTable.map[startPos].key == searchedKey) {
    		deviceValues[idx] = hashTable.map[startPos].value;
    		break;
    	}
		if (startPos == hashTable.capacity - 1)
			startPos = 0;
		else
			startPos++;
	}
}


/**
 * @brief CUDA kernel for parallel migration during expansion.
 */
__global__ void reshapeHT(HashTable hashTable, HashTable newHashTable)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= hashTable.capacity)
		return;

	// Logic: Re-inserts existing valid mappings into the new table structure.
	if (hashTable.map[idx].key != KEY_INVALID)
		insertOneItem(newHashTable, hashTable.map[idx].key, hashTable.map[idx].value);
}


/**
 * @brief Constructor: Initializes table and calculates optimal launch configurations.
 */
GpuHashTable::GpuHashTable(int size) {
	cudaError_t err;
	int minGridSize, blocksize, gridSize;

	hashTable.capacity = size;
	
	hostExistKeys = (int*)calloc(1,sizeof(int));
	DIE(hostExistKeys == NULL, "calloc");

	
	// Memory Hierarchy: Global memory allocation for the occupancy counter.
	err = cudaMalloc(&deviceExistKeys, sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMemset(deviceExistKeys, 0, sizeof(int));
	DIE(err != cudaSuccess, "cudaMemset");
	
	// Memory Hierarchy: Global memory allocation for hash items.
	err = cudaMalloc(&(hashTable.map), size * sizeof(HashItem));
	DIE(err != cudaSuccess, "cudaMalloc");

	// Optimization: Uses CUDA Occupancy API to maximize device throughput.
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blocksize, initHashT, 0, 0);
	gridSize = hashTable.capacity / blocksize;
	if (hashTable.capacity % blocksize)
		gridSize++;

	
	initHashT>>(hashTable);
	
	err = cudaDeviceSynchronize();
	DIE(err != cudaSuccess, "cudaDeviceSynchronize");
}


/**
 * @brief Destructor: Reclaims all allocated GPU and host resources.
 */
GpuHashTable::~GpuHashTable() {
	cudaError_t err;

	free(hostExistKeys);

	err = cudaFree(deviceExistKeys);
	DIE(err != cudaSuccess, "cudaFree");

	err = cudaFree(hashTable.map);
	DIE(err != cudaSuccess, "cudaFree");
}


/**
 * @brief Resizes the hash table using parallel re-hashing.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t err;
	HashTable newHashTable;
	int minGridSize, blocksize, gridSize;

	newHashTable.capacity = numBucketsReshape;

	
	err = cudaMalloc(&newHashTable.map, numBucketsReshape * sizeof(HashItem));
	DIE(err != cudaSuccess, "cudaFree");

	// Logic: Initializes new memory before migration.
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blocksize, initHashT, 0, 0);
	gridSize = hashTable.capacity / blocksize;
	if (hashTable.capacity % blocksize)
		gridSize++;

	
	initHashT>>(newHashTable);
	
	err = cudaDeviceSynchronize();
	DIE(err != cudaSuccess, "cudaDeviceSynchronize");

	// Logic: Parallel re-insertion of current keys into the expanded table.
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blocksize, reshapeHT, 0, 0);
	gridSize = hashTable.capacity / blocksize;
	if (hashTable.capacity % blocksize)
		gridSize++;
	
	
	reshapeHT>>(hashTable, newHashTable);

	err = cudaDeviceSynchronize();
	DIE(err != cudaSuccess, "cudaDeviceSynchronize");

	err = cudaFree(hashTable.map);
	DIE(err != cudaSuccess, "cudaDeviceSynchronize");

	hashTable = newHashTable;
}


/**
 * @brief Performs host-initiated batch insertion.
 * 
 * Logic: Transfers data to GPU and manages adaptive scaling if load > 90%.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	cudaError_t err;
	int *deviceKeys; 
	int *deviceValues;
	int minGridSize, blocksize, gridSize;

	
	err = cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMalloc(&deviceValues, numKeys * sizeof(int));


	DIE(err != cudaSuccess, "cudaMalloc");

	
	err = cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	err = cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	
	// Adaptive Scaling: Ensures performance by maintaining low collision density.
	float auxLF = ((float)(*hostExistKeys + numKeys) / hashTable.capacity);
	if (auxLF > 0.9f) {
		
		int newCapacity = (int)(((float)*hostExistKeys + numKeys) / 0.85f);
		reshape(newCapacity);
	}

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blocksize, insert, 0, 0);
	gridSize = hashTable.capacity / blocksize;
	if (hashTable.capacity % blocksize)
		gridSize++;

	
	insert>>(hashTable, deviceKeys, deviceValues, numKeys, deviceExistKeys);

	err = cudaDeviceSynchronize();
	DIE(err != cudaSuccess, "cudaDeviceSynchronize");

	// Logic: Synchronizes device occupancy count back to the host.
	err = cudaMemcpy(hostExistKeys, deviceExistKeys, sizeof(int), cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, "cudaMemcpy");
	
	
	err = cudaFree(deviceKeys);
	DIE(err != cudaSuccess, "cudaFree");

	err = cudaFree(deviceValues);
	DIE(err != cudaSuccess, "cudaFree");

	return true;
}


/**
 * @brief Performs host-initiated batch retrieval.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	cudaError_t err;

	int *values;
	int *deviceKeys; 
	int *deviceValues;
	int minGridSize, blocksize, gridSize;

	values = (int *)calloc(numKeys,sizeof(int));
	DIE(values == NULL , "malloc");

	
	err = cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");



	err = cudaMalloc(&deviceValues, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blocksize, get, 0, 0);
	gridSize = hashTable.capacity / blocksize;
	if (hashTable.capacity % blocksize)
		gridSize++;

	
	get>>(hashTable, deviceKeys, deviceValues, numKeys);
	
	err = cudaDeviceSynchronize();


	DIE(err != cudaSuccess, "cudaDeviceSynchronize");

	
	err = cudaMemcpy(values, deviceValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, "cudaMemcpy");

	
	err = cudaFree(deviceKeys);
	DIE(err != cudaSuccess, "cudaFree");

	err = cudaFree(deviceValues);
	DIE(err != cudaSuccess, "cudaFree");

	return values;
}


/**
 * @brief Returns the current table density.
 */
float GpuHashTable::loadFactor() {
	return ((float)*hostExistKeys / hashTable.capacity); 
}


// ... hash functions and primeList documentation ...

/**
 * @brief Modular hash indexers.
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
 * @struct HashItem
 * @brief Representation of an individual mapping on the GPU.
 */
typedef struct
{
	int key;
	int value;
}HashItem;

/**
 * @struct HashTable
 * @brief Internal metadata for the GPU resident hash map.
 */
typedef struct
{
	int capacity;   // Size of the map buffer.
	HashItem *map;  // Pointer to the device entry array.
}HashTable;



/**
 * @class GpuHashTable
 * @brief Host controller for Managing the GPU hash mapping lifecycle.
 */
class GpuHashTable
{
	int *hostExistKeys;     // Host-side occupancy counter.
	int *deviceExistKeys;   // Device-side occupancy counter.
	HashTable hashTable;    // Metadata for the device structure.

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
