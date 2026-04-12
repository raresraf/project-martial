
/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table using open addressing with linear probing.
 * @details This file provides a hash table implementation intended for the GPU.
 * It uses a simple open addressing scheme with linear probing for collision
 * resolution.
 *
 * @warning This implementation is NOT THREAD-SAFE for insertions. The `insertBatchDevice`
 * kernel lacks atomic operations, leading to severe race conditions where multiple
 * threads can overwrite the same bucket, causing data loss.
 *
 * @warning The `reshape` function is FUNDAMENTALLY FLAWED. It copies memory instead of
 * re-hashing elements, which breaks the logic of the hash table and will cause
 * lookups to fail after a resize.
 */
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief Host-side constructor. Allocates memory for the hash table on the GPU.
 * @param size The initial capacity (number of buckets) of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	this->capacity = size;
	this.size = 0;

	// Allocate a single contiguous block of memory on the GPU.
	cudaMalloc(&this->hashTableDevice, this->capacity * sizeof(Pair));
	// Initialize all keys and values to 0. A key of 0 signifies an empty bucket.
	cudaMemset(this->hashTableDevice, 0, this->capacity * sizeof(Pair));
}

/**
 * @brief Host-side destructor. Frees the GPU memory used by the hash table.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(this->hashTableDevice);
}

/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The new capacity of the hash table.
 *
 * @warning CRITICAL BUG: This function does not re-hash the existing elements. It
 * simply copies the old data into a new, larger memory block. This breaks the hash
 * table's invariant, as the positions of existing elements are not updated for the
 * new capacity, leading to lookup failures. A correct implementation would launch a
 * kernel to re-insert every element into the new table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	Pair *newHashTableDevice = 0;
	// 1. Allocate a new table with the larger capacity.
	cudaMalloc(&newHashTableDevice, numBucketsReshape * sizeof(Pair));
	cudaMemset(newHashTableDevice, 0, numBucketsReshape * sizeof(Pair));

	// 2. BUG: Copies the old table data directly without re-hashing.
	cudaMemcpy(newHashTableDevice,
		   this->hashTableDevice,
		   this->capacity * sizeof(Pair),
		   cudaMemcpyDeviceToDevice);

	// 3. Update capacity and swap pointers.
	this->capacity = numBucketsReshape;
	cudaFree(this->hashTableDevice);
	this->hashTableDevice = newHashTableDevice;
}

/**
 * @brief CUDA kernel for inserting a batch of key-value pairs.
 * @param hashTableDevice Device pointer to the hash table's data array.
 * @param capacity The total capacity of the hash table.
 * @param keys Device pointer to an array of keys to insert.
 * @param values Device pointer to an array of corresponding values.
 * @param numKeys The number of pairs to insert.
 *
 * @warning CRITICAL RACE CONDITION: This kernel does not use atomic operations
 * for writing. If two threads hash to the same bucket or find the same empty
 * slot via linear probing, they will overwrite each other's writes, leading to
 * data loss. This makes the function unsafe for parallel execution.
 */
__global__ void insertBatchDevice(Pair *hashTableDevice, int capacity, int *keys, int *values, int numKeys) {
	
	// Determine the unique index for this thread.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	
	if (i < numKeys) {
		int key = keys[i];
		int value = values[i];
		int bucket = hash2(key, capacity);

		// --- Unsafe Insertion Attempt ---
		// The following check-then-write sequence is not atomic.
		if (hashTableDevice[bucket].key == 0 || hashTableDevice[bucket].key == key) {
			// RACE CONDITION: Multiple threads could pass this check simultaneously for an
			// empty bucket and all will write here. Only the last write will persist.
			hashTableDevice[bucket].key = key;
			hashTableDevice[bucket].value = value;
		} else {
			// --- Unsafe Linear Probing ---
			int found = 0;
			// Probe from the collision point to the end of the table.
			for (int j = 0; j < capacity && found == 0; ++j) {
				// RACE CONDITION: Same issue as above. Multiple threads can find the
				// same empty slot 'j' and will cause a write collision.
				if (hashTableDevice[j].key == 0 || hashTableDevice[j].key == key) {
					hashTableDevice[j].key = key;
					hashTableDevice[j].value = value;
					found = 1;
				}
			}

			// Wrap around and probe from the beginning of the table.
			for (int j = 0; j < bucket && found == 0; ++j) {
				// RACE CONDITION: The same race condition applies here.
				if (hashTableDevice[j].key == 0 || hashTableDevice[j].key == key) {
					hashTableDevice[j].key = key;
					hashTableDevice[j].value = value;
					found = 1;
				}
			}
		}
	}
}

/**
 * @brief Host-side function to insert a batch of key-value pairs.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	
	// 1. Reshape the table if capacity is insufficient.
	while (numKeys > this->capacity - this->size) {
		this->reshape(2 * this->capacity);
	}

	// 2. Allocate temporary device memory for the batch.
	int *keys_device = 0;
	int *values_device = 0;
	cudaMalloc(&keys_device, numKeys * sizeof(int));
	cudaMalloc(&values_device, numKeys * sizeof(int));

	// 3. Copy data from host to device.
	cudaMemcpy(keys_device,
		   keys,
		   numKeys * sizeof(int),
		   cudaMemcpyHostToDevice);
	cudaMemcpy(values_device,
		   values,
		   numKeys * sizeof(int),
		   cudaMemcpyHostToDevice);

	// 4. Calculate kernel launch parameters.
	const size_t block_size = 1024;
	size_t blocks_no = (numKeys + block_size - 1) / block_size;

	// 5. Launch the (non-thread-safe) insertion kernel.
	insertBatchDevice>>(this->hashTableDevice, this->capacity, keys_device, values_device, numKeys);
	cudaDeviceSynchronize();

	this->size += numKeys;

	// 6. Free temporary device memory.
	cudaFree(keys_device);
	cudaFree(values_device);

	return true;
}

/**
 * @brief CUDA kernel for retrieving values for a batch of keys.
 * @note This operation is read-only and does not use atomics, which is generally
 * acceptable if no concurrent writes are occurring. However, its correctness
 * depends on the (flawed) insertion logic.
 */
__global__ void getBatchDevice(Pair *hashTableDevice, int capacity, int *keys, int *values, int numKeys) {
  
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  
	if (i < numKeys) {
		int key = keys[i];
		int bucket = hash2(key, capacity);

		// --- Search with Linear Probing ---
		// Check the initial bucket first.
		if (hashTableDevice[bucket].key == key) {
			values[i] = hashTableDevice[bucket].value;
		} else {
			int found = 0;
			// Probe from the next bucket to the end of the table.
			for (int j = bucket + 1; j < capacity && found == 0; ++j) {
				if (hashTableDevice[j].key == key) {
					found = 1;
					values[i] = hashTableDevice[j].value;
				}
			}
			// If not found, wrap around and probe from the beginning.
			for (int j = 0; j < bucket && found == 0; ++j) {
				if (hashTableDevice[j].key == key) {
					found = 1;
					values[i] = hashTableDevice[j].value;
				}
			}
		}
	}
}


/**
 * @brief Host-side function to retrieve values for a batch of keys.
 */
int* GpuHashTable::getBatch(int *keys, int numKeys) {
	int *values = (int *)malloc(numKeys * sizeof(int));

	// 1. Allocate temporary device buffers.
	int *keys_device = 0;
	int *values_device = 0;
	cudaMalloc(&keys_device, numKeys * sizeof(int));
	cudaMalloc(&values_device, numKeys * sizeof(int));

	// 2. Copy keys from host to device.
	cudaMemcpy(keys_device,
		   keys,
		   numKeys * sizeof(int),
		   cudaMemcpyHostToDevice);

	// 3. Configure and launch the 'get' kernel.
	const size_t block_size = 1024;
	size_t blocks_no = (numKeys + block_size -1) / block_size;
	getBatchDevice>>(this->hashTableDevice, this->capacity, keys_device, values_device, numKeys);
	cudaDeviceSynchronize();

	// 4. Copy results from device back to host.
	cudaMemcpy(values,
		   values_device,
		   numKeys * sizeof(int),
		   cudaMemcpyDeviceToHost);

	return values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
	return (float)this->size / (float) this->capacity;
}


// --- Test harness and boilerplate ---
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
