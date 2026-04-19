/**
 * @file gpu_hashtable.cu
 * @brief Dual-bucket GPU Hash Table with atomic synchronization and circular probing.
 * 
 * This module implements a parallel hash table utilizing a dual-bucket storage 
 * architecture to minimize collision density. It employs hardware-accelerated 
 * atomic operations (CAS, Exch) for lock-free concurrency and a robust 
 * circular linear probing algorithm across both memory tiers.
 * 
 * Algorithm: Open addressing with dual-bucket circular linear probing.
 * Memory Model: Global memory (cudaMalloc) for primary storage buffers.
 * Domain: HPC, Parallel Data Structures.
 */

#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief Device-side hash function using a 64-bit modular scheme.
 */
__device__ int myHash(int data, int limit) {
	return ((long) abs(data) * 20906033) % 5351951779 % limit;
}

/**
 * @brief Constructor: Initializes table metadata and dual device-side buffers.
 */
GpuHashTable::GpuHashTable(int size) {
	pairsInserted = 0;
	bucketSize = size;
	bucket1 = nullptr;
	bucket2 = nullptr;
	// Memory Hierarchy: Global memory allocation for primary bucket.
	if (cudaMalloc(&bucket1, size * sizeof(hash_entry)) != cudaSuccess) {
		return;
	}
	cudaMemset(bucket1, 0, size * sizeof(hash_entry));
	// Memory Hierarchy: Global memory allocation for secondary bucket.
	if (cudaMalloc(&bucket2, size * sizeof(hash_entry)) != cudaSuccess) {
		return;
	}
	cudaMemset(bucket2, 0, size * sizeof(hash_entry));
}


/**
 * @brief Destructor: Releases both tiers of device memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(bucket1);
	cudaFree(bucket2);
}

/**
 * @brief CUDA kernel for parallel entry insertion across dual buckets.
 * 
 * functional Utility: Attempts insertion into bucket 1 then bucket 2 using CAS. 
 * Resolves remaining collisions via a multi-pass circular linear probe.
 */
__global__ void kernel_insert(int *keys, int *values, int numKeys, hash_entry* bucket1, hash_entry* bucket2, int bucketSize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= bucketSize) return;
	int keyOld, keyNew;
	keyNew = keys[idx];
	int hash = myHash(keyNew, bucketSize);
	
	/**
	 * Block Logic: Primary insertion probe pass.
	 * Logic: Sequentially attempts atomic reservation in both buckets at the hashed index.
	 */
	for (int i = hash; i < bucketSize; i++) {


		keyOld = atomicCAS(&bucket1[i].key, KEY_INVALID, keyNew);
		if (keyOld == KEY_INVALID || keyOld == keyNew) {
			bucket1[i].value = values[idx];
			return;
		}
		keyOld = atomicCAS(&bucket2[i].key, KEY_INVALID, keyNew);
		if (keyOld == KEY_INVALID || keyOld == keyNew) {
			bucket2[i].value = values[idx];
			return;
		}
	}
	
	/**
	 * Block Logic: Wrap-around insertion probe pass.
	 */
	for (int i = 0; i < hash; i++) {
		keyOld = atomicCAS(&bucket1[i].key, KEY_INVALID, keyNew);
		if (keyOld == KEY_INVALID || keyOld == keyNew) {
			bucket1[i].value = values[idx];
			return;
		}
		keyOld = atomicCAS(&bucket2[i].key, KEY_INVALID, keyNew);
		if (keyOld == KEY_INVALID || keyOld == keyNew) {
			bucket2[i].value = values[idx];
			return;
		}
	}
}

/**
 * @brief CUDA kernel for parallel entry lookup in dual tiers.
 */
__global__ void kernel_get(int *keys, int *values, int numItems, hash_entry* bucket1, hash_entry* bucket2, int bucketSize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= bucketSize) return;
	int crtKey = keys[idx];
	int hash = myHash(crtKey, bucketSize);
	
	// Block Logic: Multi-tier search traversal.
	for (int i = hash; i < bucketSize; i++) {
		if (bucket1[i].key == crtKey) {
			values[idx] = bucket1[i].value;
			return;
		}
		if (bucket2[i].key == crtKey) {
			values[idx] = bucket2[i].value;
			return;
		}
	}
	for (int i = 0; i < hash; i++) {
		if (bucket1[i].key == crtKey) {
			values[idx] = bucket1[i].value;
			return;
		}
		if (bucket2[i].key == crtKey) {
			values[idx] = bucket2[i].value;
			return;
		}
	}
}

/**
 * @brief CUDA kernel for re-hashing data during capacity migration.
 * 
 * Functional Utility: Transfers and re-inserts mappings from old dual-buckets 
 * into new, larger destination buffers.
 */
__global__ void kernel_rehash(hash_entry* oldBucket1, hash_entry* oldBucket2, int oldBucketSize,
hash_entry* newBucket1, hash_entry* newBucket2, int newBucketSize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= oldBucketSize) return;
	
	// Logic: Process first bucket elements.
	if (oldBucket1[idx].key != KEY_INVALID) {
		int keyNew, keyOld, hash;
		keyNew = oldBucket1[idx].key;
		hash = myHash(keyNew, newBucketSize);
		bool completed = false;
		for (int i = hash; i < newBucketSize; i++) {
			keyOld = atomicCAS(&newBucket1[idx].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket1[idx].value = oldBucket1[idx].value;
				completed = true;
				break;
			}
			keyOld = atomicCAS(&newBucket2[idx].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket2[idx].value = oldBucket2[idx].value;
				completed = true;
				break;
			}
		}
		if (completed == false) {
			for (int i = 0; i < hash; i++) {
				keyOld = atomicCAS(&newBucket1[idx].key, KEY_INVALID, keyNew);
				if (keyOld == KEY_INVALID) {
					newBucket1[idx].value = oldBucket1[idx].value;
					break;
				}
				keyOld = atomicCAS(&newBucket2[idx].key, KEY_INVALID, keyNew);
				if (keyOld == KEY_INVALID) {
					newBucket2[idx].value = oldBucket2[idx].value;
					break;
				}
			}
		}
	}
	
	// Logic: Process second bucket elements.
	if (oldBucket2[idx].key != KEY_INVALID) {
		int keyNew, keyOld, hash;
		keyNew = oldBucket2[idx].key;
		hash = myHash(keyNew, newBucketSize);
		bool completed = false;
		for (int i = hash; i < newBucketSize; i++) {
			keyOld = atomicCAS(&newBucket1[idx].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket1[idx].value = oldBucket1[idx].value;
				completed = true;
				break;
			}
			keyOld = atomicCAS(&newBucket2[idx].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket2[idx].value = oldBucket2[idx].value;
				completed = true;
				break;
			}
		}
		if (completed == false) {
			for (int i = 0; i < hash; i++) {
			keyOld = atomicCAS(&newBucket1[idx].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket1[idx].value = oldBucket1[idx].value;
				break;
			}
			keyOld = atomicCAS(&newBucket2[idx].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket2[idx].value = oldBucket2[idx].value;
				break;
			}
		}
		}
	}
}

/**
 * @brief Resizes the hash table using re-allocation and re-hashing.
 */
void GpuHashTable::reshape(int sizeReshape) {
	hash_entry* newBucket1;
	hash_entry* newBucket2;
	if (cudaMalloc(&newBucket1, sizeReshape * sizeof(hash_entry)) != cudaSuccess) {
		return;
	}
	cudaMemset(newBucket1, 0, sizeReshape * sizeof(hash_entry));
	if (cudaMalloc(&newBucket2, sizeReshape * sizeof(hash_entry)) != cudaSuccess) {
		return;
	}
	cudaMemset(newBucket2, 0, sizeReshape * sizeof(hash_entry));
	unsigned int numBlocks = bucketSize / THREADS_PER_BLOCK;
	if (bucketSize % THREADS_PER_BLOCK != 0) numBlocks++;

	// Synchronization: Parallel migration to expanded memory pools.
	kernel_rehash>>(bucket1, bucket2, bucketSize, newBucket1, newBucket2, sizeReshape);
	cudaDeviceSynchronize();
	cudaFree(bucket1);
	cudaFree(bucket2);


	bucket1 = newBucket1;
	bucket2 = newBucket2;
	bucketSize = sizeReshape;
}


/**
 * @brief Performs host-initiated batch parallel insertion.
 * 
 * Logic: Transfers batch to GPU and manages adaptive scaling if load > 75%.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *keysFromDevice, *valuesFromDevice;
	unsigned int numBlocks;
	if (cudaMalloc(&keysFromDevice, numKeys * sizeof(int)) != cudaSuccess) {
		return false;
	}
	if (cudaMalloc(&valuesFromDevice, numKeys * sizeof(int)) != cudaSuccess) {
		return false;
	}

	// Adaptive Scaling: Monitors table saturation to maintain O(1) performance.
	if ((pairsInserted + numKeys) * 1.0 / bucketSize >= MAX_LOAD)
		reshape(int((pairsInserted + numKeys) / MIN_LOAD));

	cudaMemcpy(keysFromDevice, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(valuesFromDevice, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK != 0) numBlocks++;


	insert_batch>>(keysFromDevice, valuesFromDevice, numKeys, bucket1, bucket2, bucketSize);
	cudaDeviceSynchronize();

	// Logic: Updates global element count.
	pairsInserted += numKeys;
	cudaFree(keysFromDevice);
	cudaFree(valuesFromDevice);
	return true;
}


/**
 * @brief Performs host-initiated batch parallel retrieval.
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
	int *keysFromDevice, *valuesFromDevice;
	unsigned int numBlocks;
	if (cudaMalloc(&keysFromDevice, numKeys * sizeof(int)) != cudaSuccess) {
		std::cerr<< "Memory for keys failed to allocate\n";
		return nullptr;
	}
	if (cudaMalloc(&valuesFromDevice, numKeys * sizeof(int)) != cudaSuccess) {
		std::cerr<< "Memory for values failed to allocate\n";
		return  nullptr;
	}
	cudaMemcpy(keysFromDevice, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK != 0) numBlocks++;

	kernel_get>>(keysFromDevice, valuesFromDevice, numKeys, bucket1, bucket2, bucketSize);
	cudaDeviceSynchronize();
	cudaFree(keysFromDevice);
	return valuesFromDevice;
}


/**
 * @brief Returns the current occupancy density.
 */
float GpuHashTable::loadFactor() {
	if (bucketSize == 0) return 0.0;
	return pairsInserted * 1.0 / bucketSize;
}


// ... hash functions and structures ...

/**
 * @struct hash_entry
 * @brief Atomic unit of storage in the hash map.
 */
struct hash_entry {
	int key;
	int value;
};

/**
 * @class GpuHashTable
 * @brief Host controller for Managing the dual-bucket GPU hash map.
 */
class GpuHashTable {
	int pairsInserted;      // Count of elements inserted.
	hash_entry* bucket1;    // Pointer to primary device buffer.
	hash_entry* bucket2;    // Pointer to secondary device buffer.
	int bucketSize;         // Current capacity per bucket.

public:
	GpuHashTable(int size);

	void reshape(int sizeReshape);

	bool insertBatch(int *keys, int *values, int numKeys);

	int *getBatch(int *key, int numItems);

	float loadFactor();

	void occupancy();

	void print(std::string info);

	~GpuHashTable();
};


