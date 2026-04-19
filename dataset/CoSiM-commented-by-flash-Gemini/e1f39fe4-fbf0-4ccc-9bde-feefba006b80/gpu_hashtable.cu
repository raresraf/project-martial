
/**
 * @file gpu_hashtable.cu
 * @brief Dual-tier GPU Hash Table with atomic collision resolution.
 * 
 * This module implements a parallel hash table using two distinct memory tiers 
 * (node arrays) to resolve collisions. It employs hardware-accelerated atomic 
 * Compare-And-Swap (CAS) operations to ensure thread-safe updates and robust 
 * circular linear probing across both storage tiers.
 * 
 * Algorithm: Open addressing with dual-tier circular linear probing.
 * Memory Model: Global memory (cudaMalloc) for dual entry buffers.
 * Domain: HPC, Parallel Data Structures.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define BLOCKSIZE 256


/**
 * @brief Constructor: Initializes dual-tier device storage.
 * 
 * @param size Capacity for each of the two node tiers.
 */
GpuHashTable::GpuHashTable(int size) {
	// Memory Hierarchy: Global memory allocation for primary and secondary tiers.
	cudaMalloc((void**)&my_ht.nodes1, size * sizeof(Node));
	cudaMalloc((void**)&my_ht.nodes2, size * sizeof(Node));
	my_ht.items = 0;
	cudaMemset(my_ht.nodes1, 0, size * sizeof(Node));
	cudaMemset(my_ht.nodes2, 0, size * sizeof(Node));
	my_ht.size = size;
}


/**
 * @brief Destructor: Releases both tiers of device memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(my_ht.nodes1);
	cudaFree(my_ht.nodes2);
	my_ht.items = 0;
	my_ht.size = 0;
}


/**
 * @brief CUDA kernel for parallel data migration.
 * 
 * Functional Utility: Re-hashes existing elements into a single destination tier 
 * using linear probing.
 */
__global__ void kernel_resize_HashTableElems(Node *input, int size, Node *output)
{


	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= size)
		return;

	int rKey = input[i].key;
	int rValue = input[i].value;
	if (rKey <= 0 || rValue <= 0)
		return;
	int where = hash1(rKey, size);

	/**
	 * Block Logic: Linear re-insertion search.
	 * Invariant: Probes until an empty slot is successfully claimed via atomicCAS.
	 */
	while (1) {
		if(atomicCAS(&(output[where].key), 0, rKey) == 0)
			break;
		where = (where + 1) % size;
	}
	output[where].value = rValue;
}


/**
 * @brief Dynmically resizes both memory tiers of the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	Node *result1, *result2;	
	cudaMalloc((void**)&result1, numBucketsReshape * sizeof(Node));
	if (!result1)
		return;

	cudaMalloc((void**)&result2, numBucketsReshape * sizeof(Node));
	if (!result2)
		return;

	unsigned int block_no = my_ht.size / BLOCKSIZE;
	if (my_ht.size % BLOCKSIZE)
		block_no++;
	my_ht.size = numBucketsReshape;
	
	
	// Synchronization: Ensures re-hashing of tier 1 is complete.
	kernel_resize_HashTableElems>>(
		my_ht.nodes1, my_ht.size, result1);
	cudaDeviceSynchronize();
	
	// Synchronization: Ensures re-hashing of tier 2 is complete.
	kernel_resize_HashTableElems>>(
		my_ht.nodes2, my_ht.size, result2);
	cudaDeviceSynchronize();
	
	cudaFree(my_ht.nodes1);
	cudaFree(my_ht.nodes2);
	my_ht.nodes1 = result1;
	my_ht.nodes2 = result2;
}


/**
 * @brief CUDA kernel for parallel entry insertion into dual tiers.
 * 
 * functional Utility: Attempts insertion into tier 1, then tier 2. If both 
 * are occupied, it performs a circular probe until an empty or matching 
 * key slot is found.
 */
__global__ void kernel_insert_HastTableElems(HashTable h, int *keys, int *values, int numKeys)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numKeys)
		return;
	if(keys[i] <= 0 || values[i] <= 0)
		return;
	int where = hash1(keys[i], h.size);
	int key = keys[i]; 

	/**
	 * Block Logic: Atomic multi-tier insertion sequence.
	 * Logic: Sequentially attempts atomic reservation across both memory tiers 
	 * followed by circular linear probing for overflow resolution.
	 */
	while (1){
		// Tier 1 logic: Empty claim or existing update.
		if (atomicCAS(&(h.nodes1[where].key), 0, key) == 0){
			h.nodes1[where].value =  values[i];
			return;
		}
		if (atomicCAS(&(h.nodes1[where].key), key, key) == key){
			h.nodes1[where].value =  values[i];
			return;
		}

		// Tier 2 logic: Empty claim or existing update.
		if (atomicCAS(&(h.nodes2[where].key), 0, key) == 0){
			h.nodes2[where].value =  values[i];
			return;
		}
		if (atomicCAS(&(h.nodes2[where].key), key, key) == key){
			h.nodes2[where].value =  values[i];
			return;
		}	
		// Circular probe step.
		where = (where + 1) % (h.size);
	}
}




/**
 * @brief Performs batch parallel insertion from host memory.
 * 
 * Logic: Transfers batches to the GPU and manages automatic scaling if 
 * load factor > 90%.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *gpuKeys, *gpuValues;
	cudaMalloc((void**)&gpuKeys, numKeys * sizeof(int));
	cudaMalloc((void**)&gpuValues, numKeys * sizeof(int));
	cudaMemcpy(gpuKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	my_ht.items += numKeys;
	// Adaptive Scaling: Expands dual tiers to maintain low collision probability.
	if ( ((float)((numKeys + my_ht.items) / my_ht.size)) >= 0.9f)
		reshape((int)(my_ht.size / 0.8f));
	
	int block_no = numKeys / BLOCKSIZE;
	if (numKeys % BLOCKSIZE)
		block_no++;
	
	kernel_insert_HastTableElems>>(
		my_ht, gpuKeys, gpuValues, numKeys);
	cudaDeviceSynchronize();
	
	cudaFree(gpuKeys);
	cudaFree(gpuValues);
	return false;
}


/**
 * @brief CUDA kernel for parallel entry retrieval from dual tiers.
 */
__global__ void kernel_get_HashTableElems(HashTable h, int *keys, int *values, int numKeys)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numKeys)
		return;

	int where = hash1(keys[i], h.size);
	int key = keys[i];
	if (key == 0)
		return;
	
	// Block Logic: Multi-tier search pass.
	while (1) {
		if (atomicCAS(&(h.nodes1[where].key), key, key) == key){
			values[i] = h.nodes1[where].value;
			break;
		}
		if (atomicCAS(&(h.nodes2[where].key), key, key) == key){
			values[i] = h.nodes2[where].value;
			break;
		}
		where = (where + 1) % (h.size);	
	}
}


/**
 * @brief Performs batch parallel retrieval of values.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *result;
	int *gpuKeys, *gpuValues;
	result = (int*)calloc(numKeys, sizeof(int));
	if (!result)
		return NULL;
	cudaMalloc((void**)&gpuValues, numKeys * sizeof(int));
	if (!gpuValues)
		return NULL;
	cudaMalloc((void**)&gpuKeys, numKeys * sizeof(int));
	if (!gpuKeys)
		return NULL;
	cudaMemcpy(gpuKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int block_no = numKeys / BLOCKSIZE;
	if (numKeys % BLOCKSIZE)
		block_no++;
	
	kernel_get_HashTableElems>>(
		my_ht, gpuKeys, gpuValues, numKeys);
	cudaDeviceSynchronize();

	cudaMemcpy(result, gpuValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(gpuKeys);
	cudaFree(gpuValues);

	return result;
}


/**
 * @brief Returns the current occupancy density.
 */
float GpuHashTable::loadFactor() {
		return 1.0f * (float) my_ht.items / my_ht.size; 
}


// ... primeList documentation ...

/**
 * @brief Modular hash functions for device indexing.
 */
__device__ int hash1(int data, int limit) {
	return ((long)abs(data) * 3llu) % 9122181901073924329llu % limit;
}
__device__ int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
__device__ int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

/**
 * @struct Node
 * @brief Atomic storage unit for a key-value mapping on the GPU.
 */
typedef struct {
	int key;
	int value;
} Node;

/**
 * @struct HashTable
 * @brief Metadata and storage pointers for the dual-tier device hash map.
 */
typedef struct {
	Node *nodes1;   // Primary node array.


	Node *nodes2;   // Secondary overflow node array.
	int size;       // Capacity of each array.
	int items;      // Count of inserted units.
} HashTable;




/**
 * @class GpuHashTable
 * @brief Host-side orchestrator for dual-tier GPU hash storage.
 */
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

		HashTable my_ht; // Managed device state.
		~GpuHashTable();
};

