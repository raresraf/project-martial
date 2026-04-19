/**
 * @file gpu_hashtable.cu
 * @brief Thread-safe GPU Hash Table with atomic collision handling and scaling.
 * 
 * This module implements a parallel hash table for high-throughput query and 
 * update operations. It utilizes atomic Compare-And-Swap (CAS) to claim slots 
 * and atomic additions to track occupancy across massive thread pools. The 
 * implementation includes an integrated re-hashing kernel to handle dynamic 
 * table expansion while maintaining data integrity.
 * 
 * Algorithm: Open addressing with linear probing and atomic counter reconciliation.
 * Memory Model: Global memory (cudaMalloc) for node arrays and occupancy metrics.
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
 * @brief Constructor: Initializes table and occupancy counter.
 */
GpuHashTable::GpuHashTable(int size) {
	// Memory Hierarchy: Global memory allocation for entries and size counter.
	cudaMalloc((void **)&map, size*sizeof(GpuHashNode));
	limit = size;
	curr_size = 0;
	cudaMalloc((void **)&dev_size, sizeof(int));
}


/**
 * @brief Destructor: Releases device-resident entries.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(map);
}

/**
 * @brief Device-side hash function using a large prime multiplier.
 */
__device__ int hash1(int data, int limit) {
	return ((long)abs(data) * 1061961721llu) % 14480561146010017169llu;
}


/**
 * @brief CUDA kernel for parallel data migration.
 * 
 * functional Utility: Transfers valid key-value pairs into a new memory buffer 
 * using parallel re-hashing.
 */
__global__ void kernel_reshape_map(GpuHashNode *hmap, GpuHashNode *new_hmap, int new_lim, int lim) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= lim) {
		return;
	}

	GpuHashNode node = hmap[idx];
	if(node.val == 0 || node.key == 0) {
		return;
	}

	int key = node.key;
	int val = node.val;
	
	unsigned int hash_code = hash1(key, new_lim);
	unsigned int tmp_code = hash_code;
	hash_code %= new_lim;
	
	/**
	 * Block Logic: Linear re-insertion search.
	 * Invariant: Probes the new buffer until an empty slot is claimed via atomicCAS.
	 */
	for(int i = 1; i <= new_lim; i++) {

		if(atomicCAS(&(new_hmap[hash_code].key), 0, key) == 0) {
			atomicExch(&(new_hmap[hash_code].val), val);
			return;
		}

		if(new_hmap[hash_code].key == key && new_hmap[hash_code].val == val) {
			return;
		}

		hash_code = (tmp_code + i%new_lim) % new_lim;
	}
}


/**
 * @brief Resizes the hash table using parallel re-hashing.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	if(limit < numBucketsReshape) {
		GpuHashNode *new_map;
		cudaMalloc((void **)&new_map, numBucketsReshape*sizeof(GpuHashNode));

		int numBlocks = limit / 512 + 1;
		// Synchronization: Migrates data before updating the primary pointer.
		kernel_reshape_map>>(map, new_map, numBucketsReshape, limit);
		cudaDeviceSynchronize();

		GpuHashNode *tmp_map = map;
		map = new_map;
		limit = numBucketsReshape;


		cudaFree(tmp_map);
	}

}

/**
 * @brief CUDA kernel for parallel entry insertion.
 * 
 * functional Utility: Performs atomic reservations. Tracks unique insertions 
 * via atomicAdd for global occupancy reconciliation.
 */
__global__ void kernel_insertBatch(GpuHashNode *hmap, int *keys, int *values, int limit, int numKeys, int *size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= numKeys) {
		return;
	}

	int key = keys[idx];
	int val = values[idx];

	if(key == 0 || val == 0) {
		return;
	}
	
	unsigned int hash_code = hash1(key, limit);
	unsigned int tmp_code = hash_code;
	hash_code %= limit;
	
	/**
	 * Block Logic: Atomic probe loop.
	 */
	for(int i = 1; i <= limit; i++) {
		if(hmap[hash_code].key == key && hmap[hash_code].val == val) {
			return;
		}

		if(atomicCAS(&(hmap[hash_code].key), 0, key) == 0) {
			atomicExch(&(hmap[hash_code].val), val);
			// Logic: Atomically increment occupancy if a new slot was initialized.
			atomicAdd((unsigned int *)size, 1);
			return;
		}

		
		if(hmap[hash_code].key == key) {
			// Logic: Key matches; performing thread-safe update.
			atomicExch(&(hmap[hash_code].val), val);
			return;
		}

		hash_code = (tmp_code + i%limit) % limit;
	}

}

/**
 * @brief Performs host-initiated batch parallel insertion.
 * 
 * Logic: Monitors density and triggers expansion if load factor > 85%.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	
	
	
	// Adaptive Scaling: Ensures performance by maintaining density targets.
	GpuHashTable::reshape((((double)curr_size + (double)numKeys))/0.85);

	int numBlocks = numKeys / 512 + 1;
	int *dev_keys, *dev_values;
	cudaMalloc((void **)&dev_keys, numKeys*sizeof(int));
	cudaMalloc((void **)&dev_values, numKeys*sizeof(int));
	cudaMemcpy(dev_keys, keys, numKeys*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_values, values, numKeys*sizeof(int), cudaMemcpyHostToDevice);

	
	kernel_insertBatch>>(map, dev_keys, dev_values, limit, numKeys, dev_size);
	cudaDeviceSynchronize();
	
	// Logic: Synchronizes current occupancy back to host.
	cudaMemcpy(&curr_size, dev_size, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_keys);
	cudaFree(dev_values);
	return true;
}



__global__ void kernel_getBatch(GpuHashNode *hmap, int *keys, int *values, int lim, int numKeys) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= numKeys) {
		return;
	}
	int key = keys[idx];

	unsigned int hash_code = hash1(key, lim);
	unsigned int tmp_code = hash_code;
	hash_code %= lim;
	
	for(int i = 1; i <= lim; i++) {
		
		if(hmap[hash_code].key == key) {
			values[idx] = hmap[hash_code].val;
			return;
		}
		hash_code = (tmp_code + i%lim) % lim;
	}
}

/**
 * @brief Performs host-initiated batch retrieval.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *result = (int*)malloc(numKeys * sizeof(int));
	int *dev_keys, *dev_values;
	cudaMalloc((void **)&dev_keys, numKeys*sizeof(int));
	cudaMalloc((void **)&dev_values, numKeys*sizeof(int));
	cudaMemcpy(dev_keys, keys, numKeys*sizeof(int), cudaMemcpyHostToDevice);

	int numBlocks = numKeys / 512 + 1;
	kernel_getBatch>>(map, dev_keys, dev_values, limit, numKeys);
	cudaDeviceSynchronize();

	cudaMemcpy(result, dev_values, numKeys*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_values);
	cudaFree(dev_keys);
	return result;
}


/**
 * @brief Returns the current table density ratio.
 */
float GpuHashTable::loadFactor() {
	
	return (curr_size*1.0f)/limit; 
}


// ... primeList and hash3 ...

/**
 * @struct GpuHashNode
 * @brief Representation of an individual mapping on the device.
 */
typedef struct GpuHashNode {
	int val = 0;
	int key = 0;
}GpuHashNode;

/**
 * @class GpuHashTable
 * @brief Host controller for Managing the GPU hash mapping lifecycle.
 */
class GpuHashTable
{

	public:
		GpuHashNode *map;   // Pointer to device entry array.
		int limit;          // Maximum capacity of the buffer.
		int curr_size;      // Host-side occupancy tracker.
		int *dev_size;      // Device-side atomic counter for occupancy.
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		void reshape_and_rehash(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		
		float loadFactor();
		void occupancy();
		void print(string info);
	
		~GpuHashTable();
};

#endif

