/**
 * @file gpu_hashtable.cu
 * @brief GPGPU Hash Table with linear probing and duplicate tracking.
 * 
 * This implementation features a single-hash linear probing algorithm with manual 
 * circularity handling. It manages global memory entries and attempts to track 
 * redundant key insertions via a dedicated device-side counter.
 * 
 * Algorithm: Open addressing with two-loop linear probing.
 * Memory Model: Global memory (cudaMalloc) for entry buffers.
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
 * @brief Device-side hash function using a multiplicative constant.
 */
__device__ int myHash(int data, int limit) {
	return ((long)abs(data) * 2654435761llu) % 4294967296llu % limit;
}


/**
 * @brief CUDA kernel for parallel entry insertion.
 * 
 * Functional Utility: Claims empty slots using atomicCAS. Employs a manual 
 * circular probe strategy (two distinct loops) to resolve collisions.
 * 
 * @param duplicate Pointer to an integer for tracking redundant insertions.
 */
__global__ void kernel_insert(int *keys, int *values, int limitBound, hashtable hashmap, int *duplicate) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= limitBound) {
		return;
	}

	int extractKey = keys[i], size = hashmap.size, oldK, check = 0;
	int hash = myHash(extractKey, size);

	/**
	 * Block Logic: Forward probe sequence.
	 * Logic: Searches from the initial hash index to the end of the memory buffer.
	 */
	for (int k = hash; k < size; ++k) {


		oldK = atomicCAS(&hashmap.list[k].key, KEY_INVALID, extractKey);
		if (oldK == KEY_INVALID || oldK == extractKey) {
			hashmap.list[k].value  = values[i];
			check = 1;
			if (oldK == extractKey)
				*duplicate++; // Note: Incrementing the pointer itself (potential logic bug in original).
			break;
		}
	}

	/**
	 * Block Logic: Wrap-around probe sequence.
	 * Logic: Searches from the start of the buffer to the original hash index.
	 */
	if (check == 0) {
		for (int k = 0; k < hash; ++k) {
			oldK = atomicCAS(&hashmap.list[k].key, KEY_INVALID, extractKey);
			if (oldKey == KEY_INVALID || oldKey == extractKey) {
				hashmap.list[k].value  = values[i];
				if (oldK == extractKey)
					*duplicate++;
				return;
			}
		}
	}
 
	return;
}


/**
 * @brief CUDA kernel for parallel value retrieval.
 * 
 * Functional Utility: Implements a two-pass linear search to find values 
 * mapped to specific keys in the concurrent hash table.
 */
__global__ void kernel_get(int *keys, int *values, int limitBound, hashtable hashmap) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= limitBound) {
		return;
	}

	int extractKey = keys[i], size = hashmap.size, check = 0;
	int hash = myHash(extractKey, size);

	/**
	 * Block Logic: Forward search pass.
	 */
	for (int k = hash; k < size; ++k) {
		if (hashmap.list[k].key == extractKey) {
			values[i] = hashmap.list[k].value;
			check = 1;
			return;
		}
	}

	/**
	 * Block Logic: Backward wrap-around search pass.
	 */
	if (check == 0) {
		for (int k = 0; k < hash; ++k) {
			if (hashmap.list[k].key == extractKey) {
				values[i] = hashmap.list[k].value;
				return;
			}
		}
	}

}


/**
 * @brief CUDA kernel for data migration during table expansion.
 */
__global__ void kernel_reshape(hashtable hashmap, hashtable newHashmap) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= hashmap.size) {
		return;
	}

	// Logic: Re-inserts valid entries into the new structure.
	if (hashmap.list[i].key != KEY_INVALID) {
		int extractKey = hashmap.list[i].key, oldK, isInserted = 0, size = newHashmap.size;
		int hash = myHash(extractKey, size);

		for (int k = hash; k < size && !isInserted; ++k) {


			oldK = atomicCAS(&newHashmap.list[k].key, KEY_INVALID, extractKey);
			if (oldK == KEY_INVALID) {
				newHashmap.list[k].value  = hashmap.list[i].value;
				isInserted = 1;
				break;
			}
		}

		if (isInserted == 0) {
			for (int k = 0; k < hash && !isInserted; ++k) {
				oldK = atomicCAS(&newHashmap.list[k].key, KEY_INVALID, extractKey);
				if (oldK == KEY_INVALID) {
					newHashmap.list[k].value  = hashmap.list[i].value;
					isInserted = 1;
					break;
				}
			}
		}
	}

}


/**
 * @brief Constructor: Initializes the hash map on the GPU.
 */
GpuHashTable::GpuHashTable(int size) {
	length = 0;
	hashmap.size = size;
	hashmap.list = nullptr;
	
	// Memory Hierarchy: Global memory allocation for the entry pool.
	if (cudaMalloc(&hashmap.list, size * sizeof(entry)) != cudaSuccess) {
		std::cerr << "Memory allocation error\n";
		return;
	}
	cudaMemset(hashmap.list, 0, size * sizeof(entry));  
}


/**
 * @brief Destructor: Releases device resources.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashmap.list);
}


/**
 * @brief Rebuilds the hash table with a new capacity.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hashtable newHashmap;
	newHashmap.size = numBucketsReshape;
	if (cudaMalloc(&newHashmap.list, numBucketsReshape * sizeof(entry)) != cudaSuccess) {
		std::cerr << "Memory allocation error\n";
		return;
	}
	cudaMemset(newHashmap.list, 0, numBucketsReshape * sizeof(entry));

	int numBlocks = ((hashmap.size % 1024 == 0) ? (hashmap.size / 1024) : (hashmap.size / 1024 + 1));
	
	// Synchronization: Ensures all mappings are re-hashed before memory cleanup.
	kernel_reshape>>(hashmap, newHashmap);

	cudaDeviceSynchronize();
	cudaFree(hashmap.list);
	hashmap = newHashmap;
}


/**
 * @brief Performs host-initiated batch parallel insertion.
 * 
 * Logic: Transfers batch to GPU and manages adaptive scaling if load > 100%. 
 * Reconciles actual insertion count by tracking updates to existing keys.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *deviceKeys, *deviceValues, *duplicate;
	int dupl;
	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	cudaMalloc(&deviceValues, numKeys * sizeof(int));
	cudaMalloc(&duplicate, sizeof(int));
	if (!deviceKeys || !deviceValues) {
		std::cerr << "Memory allocation error\n";
		return false;
	}
	
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemset(duplicate, 0, sizeof(int));

	 // Adaptive Scaling: Resizes if the incoming batch might overflow the table.
	 if (float(length + numKeys) >= hashmap.size) {                                             
                reshape(int(((length + numKeys) / 0.83)));                                                                                                                                                                  }
	int numBlocks = ((numKeys % 1024 == 0) ? (numKeys / 1024) : (numKeys / 1024 + 1));
	
	kernel_insert>>(deviceKeys, deviceValues, numKeys, hashmap, duplicate);
	
	// Logic: Synchronizes update count back to host for occupancy reconciliation.
	cudaMemcpy(&dupl, duplicate, sizeof(int), cudaMemcpyDeviceToHost);
	int dup = dupl;

	cudaDeviceSynchronize();
	length += numKeys;
	length -= dup;

	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	return true;
}


/**
 * @brief Performs host-initiated batch parallel retrieval.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {


	int *deviceKeys, *values;
	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	// Memory Hierarchy: Managed memory (Unified) for host access to results.
	cudaMallocManaged(&values, numKeys * sizeof(int));

	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int numBlocks = ((numKeys % 1024 == 0) ? (numKeys / 1024) : (numKeys / 1024 + 1));
	kernel_get>>(deviceKeys, values, numKeys, hashmap);

	cudaDeviceSynchronize();

	cudaFree(deviceKeys);
	return values;
}


/**
 * @brief Returns current occupancy density.
 */
float GpuHashTable::loadFactor() {
	
	if (hashmap.size == 0) {
		return 0.f;
	}
	return (float(length) / hashmap.size);
}


// ... hash functions and structures ...

/**
 * @struct entry
 * @brief Atomic storage unit for a key-value mapping on the GPU.
 */
typedef struct entry {
	int key;
	int value;
} entry;


/**
 * @struct hashtable
 * @brief Internal metadata and storage pointer for the device resident map.
 */
typedef struct hashtable {
	int size;       // Entry pool capacity.
	entry *list;    // Pointer to device mapping array.
} hashtable;


/**
 * @class GpuHashTable
 * @brief Host controller for Managing the GPU hash mapping lifecycle.
 */
class GpuHashTable
{	
	int length;         // Host-side occupancy tracker.
	hashtable hashmap;  // Metadata for the device structure.
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

