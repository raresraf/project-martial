
/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table using open addressing with linear probing.
 * @details This file implements a hash table on the GPU using a correct linear
 * probing algorithm for collision resolution and a proper parallel re-hashing
 * strategy. It also attempts to count duplicate key insertions. The implementation
 * leverages Unified Memory (`cudaMallocManaged`).
 *
 * @warning RACE CONDITION: The `insert_kernel` contains a race condition on the
 * non-atomic value update for existing keys.
 * @warning LOGIC BUG: The `get_kernel` incorrectly terminates its search if it
 * encounters an empty slot, which will fail to find keys that are located after
 * a deleted element in a probe chain.
 * @warning POINTER ARITHMETIC BUG: The duplicate counter in `insert_kernel`
 * (`*duplicate++`) incorrectly increments the pointer instead of the value it points to.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief Computes a hash value for a given key on the GPU.
 */
__device__ int myHash(int data, int limit) {
	return ((long)abs(data) * 2654435761llu) % 4294967296llu % limit;
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs.
 */
__global__ void kernel_insert(int *keys, int *values, int limitBound, hashtable hashmap, int *duplicate) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= limitBound) {
		return;
	}

	int extractKey = keys[i], size = hashmap.size, oldK, check = 0;
	int hash = myHash(extractKey, size);
	
	// --- Linear Probing with Full Wrap-Around ---
	// Probes from the initial hash index to the end of the table.
	for (int k = hash; k < size; ++k) {
		// Atomically attempt to claim a slot if empty or update if key matches.
		oldK = atomicCAS(&hashmap.list[k].key, KEY_INVALID, extractKey);
		if (oldK == KEY_INVALID || oldK == extractKey) {
			// RACE CONDITION: This value update is not atomic.
			hashmap.list[k].value  = values[i];
			check = 1;
			if (oldK == extractKey) {
				// BUG: This increments the pointer `duplicate`, not the value it points to.
				// Should be `atomicAdd(duplicate, 1)` or `(*duplicate)++` (if single-threaded access).
				*duplicate++;
			}
			break;
		}
	}
	
	// If no slot found, wrap around and probe from the beginning.
	if (check == 0) {
		for (int k = 0; k < hash; ++k) {
			oldK = atomicCAS(&hashmap.list[k].key, KEY_INVALID, extractKey);
			if (oldK == KEY_INVALID || oldK == extractKey) {
				hashmap.list[k].value  = values[i];
				if (oldK == extractKey) {
					// BUG: Same pointer arithmetic error as above.
					*duplicate++;
				}
				return;
			}
		}
	}
 
	return;
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @warning LOGIC BUG: The search incorrectly stops if it finds an empty slot (`key == 0`).
 * This will fail to find keys that are located after a deleted element in a probe chain.
 */
__global__ void kernel_get(int *keys, int *values, int limitBound, hashtable hashmap) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= limitBound) {
		return;
	}

	int extractKey = keys[i], size = hashmap.size, check = 0;
	int hash = myHash(extractKey, size);
	
	// --- Search with Flawed Linear Probing ---
	for (int k = hash; k < size; ++k) {
		// A non-atomic read is sufficient for a get operation.
		if (hashmap.list[k].key == extractKey) {
			values[i] = hashmap.list[k].value;
			check = 1;
			return; // Should be `break;` to handle multiple matches if needed, but return is ok.
		}
	}

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
 * @brief CUDA kernel to resize the hash table by re-hashing all elements.
 */
__global__ void kernel_reshape(hashtable hashmap, hashtable newHashmap) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= hashmap.size) {
		return;
	}

	// Skip empty slots in the old table.
	if (hashmap.list[i].key != KEY_INVALID) {
		int extractKey = hashmap.list[i].key, oldK, isInserted = 0, size = newHashmap.size;
		int hash = myHash(extractKey, size);

		// --- Re-insertion with correct Linear Probing ---
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
 * @brief Host-side constructor using Unified Memory.
 */
GpuHashTable::GpuHashTable(int size) {
	length = 0;
	hashmap.size = size;
	hashmap.list = nullptr;
	if (cudaMallocManaged(&hashmap.list, size * sizeof(entry)) != cudaSuccess) {
		std::cerr << "Memory allocation error
";
		return;
	}
	cudaMemset(hashmap.list, 0, size * sizeof(entry));  
}

/**
 * @brief Host-side destructor.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashmap.list);
}

/**
 * @brief Resizes the hash table to a new, larger capacity.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hashtable newHashmap;
	newHashmap.size = numBucketsReshape;
	if (cudaMalloc(&newHashmap.list, numBucketsReshape * sizeof(entry)) != cudaSuccess) {
		std::cerr << "Memory allocation error
";
		return;
	}
	cudaMemset(newHashmap.list, 0, numBucketsReshape * sizeof(entry));

	int numBlocks = (hashmap.size + 1023) / 1024;
	kernel_reshape>>(hashmap, newHashmap);
	cudaDeviceSynchronize();

	cudaFree(hashmap.list);
	hashmap = newHashmap;
}

/**
 * @brief Inserts a batch of key-value pairs from the host.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *deviceKeys, *deviceValues, *duplicate;
	int dupl;
	
	// 1. Allocate temporary device buffers.
	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	cudaMalloc(&deviceValues, numKeys * sizeof(int));
	cudaMalloc(&duplicate, sizeof(int)); // For counting duplicates
	if (!deviceKeys || !deviceValues) {
		std::cerr << "Memory allocation error
";
		return false;
	}
	
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemset(duplicate, 0, sizeof(int));

	// 2. Resize if load factor exceeds threshold.
	if (float(length + numKeys) / hashmap.size >= 0.95f) { // More robust check
        reshape(int((length + numKeys) / 0.83f));
    }
    
    // 3. Launch insertion kernel.
	int numBlocks = (numKeys + 1023) / 1024;
	kernel_insert>>(deviceKeys, deviceValues, numKeys, hashmap, duplicate);
	
	// 4. Copy duplicate count back and update total size.
	cudaMemcpy(&dupl, duplicate, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize(); // Ensure kernel is done before reading 'dupl'
	
	// The 'dup' value is unreliable due to the pointer bug in the kernel.
	int dup = dupl; 
	length += numKeys;
	length -= dup;

	// 5. Free temporary buffers.
	cudaFree(deviceKeys);
	cudaFree(deviceValues);
	cudaFree(duplicate);

	return true;
}

/**
 * @brief Retrieves values for a batch of keys from the host.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *deviceKeys, *values;
	
	// Allocate Unified Memory for the results.
	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	cudaMallocManaged(&values, numKeys * sizeof(int));

	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int numBlocks = (numKeys + 1023) / 1024;
	kernel_get>>(deviceKeys, values, numKeys, hashmap);
	cudaDeviceSynchronize();

	cudaFree(deviceKeys);
	
	// Returns a Unified Memory pointer, accessible by the host.
	return values;
}

/**
 * @brief Calculates the current load factor.
 */
float GpuHashTable::loadFactor() {
	if (hashmap.size == 0) {
		return 0.f;
	}
	return (float(length) / hashmap.size);
}

// --- Boilerplate test harness ---
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_
// ... (rest of boilerplate)
#endif
