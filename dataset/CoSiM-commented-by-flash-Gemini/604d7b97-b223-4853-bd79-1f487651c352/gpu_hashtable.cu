
/**
 * @604d7b97-b223-4853-bd79-1f487651c352/gpu_hashtable.cu
 * @brief Vector-oriented parallel GPU Hash Table with Managed Memory counters.
 * This implementation provides a high-throughput hash table for CUDA devices. 
 * It handles collisions via linear probing and uses atomic operations (atomicCAS) 
 * for consistent bucket state transitions. The element count is maintained in 
 * Managed Memory for efficient Host-Device synchronization during batch 
 * insertions and load-factor-based resizing.
 * 
 * Algorithm: Open addressing with linear probing and large-prime multiplicative hashing.
 * Domain: HPC, Parallel Computing, CUDA.
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>

#include "gpu_hashtable.hpp"

const size_t HASH1 = 10703903591llu;
const size_t HASH2 = 21921594616111llu;

/**
 * Functional Utility: Device-side prime multiplicative hash function.
 */
__device__ int hash4(int data, int limit) {
	return ((long long)abs(data) * HASH1) % HASH2 % limit;
}

/**
 * Block Logic: Resizing kernel for migrating entry vectors.
 * Thread Indexing: Each thread maps to one slot in the old hash table.
 * Logic: Transfers non-empty elements from old_vec into new_vec using 
 * recalculated indices and linear probing with wraparound.
 */
__global__ void kernel_copyVec(struct hash_pair *old_vec, struct hash_pair *new_vec, int old_n, int new_n) {
	
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	
	if (i < old_n) {
		int newKey = old_vec[i].key;
		if (newKey == 0) return; // Skip empty slots.

		int hash = hash4(newKey, new_n);
		int start = hash;
		int end = new_n;

		int flag = 0;	
		/**
		 * Block Logic: Destination linear probing (forward).
		 */
		for (int k = start; k < end; k++) {
			int oldKey = atomicCAS(&new_vec[k].key, KEY_INVALID, newKey);
			if (oldKey == KEY_INVALID) {
				new_vec[k].value = old_vec[i].value;
				flag = 1;
				break;
			}
		}
		/**
		 * Block Logic: Destination linear probing (wraparound).
		 */
		if (flag == 0) {
			for (int k = 0; k < start; k++) {
				int oldKey = atomicCAS(&new_vec[k].key, KEY_INVALID, newKey);
				if (oldKey == KEY_INVALID) {
					new_vec[k].value = old_vec[i].value;
					break;
				}
			}
		}
  	}
}

/**
 * Block Logic: Parallel atomic batch insertion kernel.
 * Logic: Maps threads to input keys and performs linear probing with 
 * atomicCAS. If a new bucket is claimed, it atomically updates the 
 * global 'capacity' (element count) pointer.
 */
__global__ void kernel_insertVec(int *keys, int *values, int numKeys, struct hash_pair *hashtable, int full_capacity, int *capacity) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < numKeys) {
		int newKey = keys[i];
		int hash = hash4(newKey, full_capacity);
		int start = hash;
		int end = full_capacity;

		int flag = 0;
		for(int k = start; k < end; k++) {
			int oldKey = atomicCAS(&hashtable[k].key, KEY_INVALID, newKey);

			if (oldKey == KEY_INVALID) {
				
				hashtable[k].value = values[i];
				flag = 1;
				
				atomicAdd(&(*capacity), 1);
				break;
			}
			if (oldKey == newKey) {
				
				hashtable[k].value = values[i];
				flag = 1;
				break;
			}
			
		}
		if (flag == 0) {
			
			for (int k = 0; k < start; k++) {
				int oldKey = atomicCAS(&hashtable[k].key, KEY_INVALID, newKey);

				if (oldKey == KEY_INVALID) {
					hashtable[k].value = values[i];
					atomicAdd(&(*capacity), 1);
					break;
				}
				if (oldKey == newKey) {
					hashtable[k].value = values[i];
					break;
				}
			}
		}
  	}
}

/**
 * Block Logic: Parallel batch lookup kernel.
 * Logic: Searches for keys in the vector starting from their hashed indices.
 */
__global__ void kernel_getVec(int *keys, int *values, int numKeys, struct hash_pair *hashtable, int full_capacity) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < numKeys) {
		int key = keys[i];
		int hash = hash4(key, full_capacity);
		int start = hash;
		int end = full_capacity;

		int flag = 0;
		for(int k = start; k < end; k++) {
			if (hashtable[k].key == key) {
				
				values[i] = hashtable[k].value;
				flag = 1;
				break;
			}
		}
		if (flag == 0) {
			for (int k = 0; k < start; k++) {
				if (hashtable[k].key == key) {
					values[i] = hashtable[k].value;
					break;
				}
			}
		}
  	}
}

/**
 * Functional Utility: Initializes the GPU hash table with a managed capacity counter.
 */
GpuHashTable::GpuHashTable(int size) {
	cudaError_t err;

	
	err = cudaMallocManaged(&capacity, sizeof(int));
	DIE(err != cudaSuccess, "cudaMallocManaged");

	*capacity = 0;

	full_capacity = size;

	
	err = cudaMalloc(&hashtable, sizeof(struct hash_pair) * full_capacity);
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMemset(hashtable, 0, full_capacity * sizeof(struct hash_pair));
	DIE(err != cudaSuccess, "cudaMemset");
}

/**
 * Functional Utility: Reclaims GPU and Managed memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaError_t err;

	
	err = cudaFree(hashtable);
	DIE(err != cudaSuccess, "cudaFree hashtable");

	err = cudaFree(capacity);
	DIE(err != cudaSuccess, "cudaFree capacity");
}

/**
 * Functional Utility: Increases table capacity and migrates entry vector.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t err;
	struct hash_pair *copy_hashtable;

	
	err = cudaMalloc(&copy_hashtable, sizeof(struct hash_pair) * numBucketsReshape);
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMemset(copy_hashtable, 0, numBucketsReshape * sizeof(hash_pair));
	DIE(err != cudaSuccess, "cudaMalloc");

	
	int blocks_no = full_capacity / BLOCKSIZE;
	if (full_capacity % BLOCKSIZE)
  		++blocks_no;

  	
	kernel_copyVec>>(hashtable, copy_hashtable, full_capacity, numBucketsReshape);
	err = cudaDeviceSynchronize();
	DIE(err != cudaSuccess, "cudaDeviceSynchronize");

	err = cudaFree(hashtable);
	DIE(err != cudaSuccess, "cudaDeviceSynchronize");

	full_capacity = numBucketsReshape;
	hashtable = copy_hashtable;
}

/**
 * Functional Utility: Orchestrates high-concurrency batch insertions with dynamic resizing.
 * Logic: Checks if the new batch will exceed 90% load factor. If so, expands the table 
 * to target an 80% load factor.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	cudaError_t err;
	int *deviceKeys, *deviceValues;

	err = cudaMalloc(&deviceKeys, (size_t) (numKeys * sizeof(int)));
	DIE(err != cudaSuccess, "cudaMalloc");



	err = cudaMalloc(&deviceValues, (size_t) (numKeys * sizeof(int)));
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	err = cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	
	float new_factor = ( (float) (*capacity + numKeys) / full_capacity );
	if (new_factor > 0.9f)
		reshape((int)(((float) *capacity + numKeys) / 0.8f));

	int blocks_no = numKeys / BLOCKSIZE;
	if (numKeys % BLOCKSIZE)
		++blocks_no;

	kernel_insertVec>>(deviceKeys, deviceValues, numKeys, hashtable, full_capacity, capacity);
	err = cudaDeviceSynchronize();
	DIE(err != cudaSuccess, "cudaDeviceSynchronize");



	err = cudaFree(deviceKeys);
	DIE(err != cudaSuccess, "cudaFree");

	err = cudaFree(deviceValues);
	DIE(err != cudaSuccess, "cudaFree");

	return true;
}

/**
 * Functional Utility: Retrieves values for a batch of keys in parallel.
 * Logic: Executes lookup kernel and returns result array in host memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	cudaError_t err;


	int *deviceKeys, *deviceValues, *hostValues;

	err = cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMalloc(&deviceValues, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");

	hostValues = (int*) calloc(numKeys, sizeof(int));

	err = cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	int blocks_no = numKeys / BLOCKSIZE;
	if (numKeys % BLOCKSIZE)
		++blocks_no;

	kernel_getVec>>(deviceKeys, deviceValues, numKeys, hashtable, full_capacity);
	cudaDeviceSynchronize();

	err = cudaMemcpy(hostValues, deviceValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, "cudaMemcpy");

	err = cudaFree(deviceKeys);
	DIE(err != cudaSuccess, "cudaFree");

	err = cudaFree(deviceValues);
	DIE(err != cudaSuccess, "cudaFree");
	
	return hostValues;
}

/**
 * Functional Utility: Returns the ratio of occupied slots to total capacity.
 */
float GpuHashTable::loadFactor() {
	float factor = ((float) *capacity / full_capacity);
	return factor;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
