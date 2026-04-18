
/**
 * @59d4e7d8-ac7c-4515-89b1-7802eba13e29/gpu_hashtable.cu
 * @brief CUDA implementation of a Hash Table with linear probing and dynamic re-hashing.
 * This implementation distributes keys across a device-side bucket array. It uses 
 * atomic operations (atomicCAS) to handle parallel insertions and features a 
 * two-pass linear probing strategy (forward and wraparound). Resizing is handled 
 * by a dedicated kernel that migrates data to a new larger table.
 * 
 * Algorithm: Open addressing with linear probing and prime multiplicative hashing.
 * Domain: HPC, Parallel Data Structures, CUDA.
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>

#include "gpu_hashtable.hpp"

/**
 * Functional Utility: Device-side prime multiplicative hash function.
 */
__device__ int myhashKernel(int data, int limit) {
	return ((long long)abs(data) * PRIMENO) % PRIMENO2 % limit;
}

/**
 * Block Logic: Parallel atomic insertion kernel.
 * Thread Indexing: Each thread maps to one key-value pair in the input batch.
 * Logic: Implements linear probing with wraparound. Uses atomicCAS to 
 * ensure exclusive bucket acquisition or update existing keys.
 */
__global__ void insert_kerFunc(int *keys, int *values, int limit, hashtable myTable) {
	int newIndex;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int old, i, j;
	int low, high;

	if (index < limit) {
		newIndex = myhashKernel(keys[index], myTable.size);
		low = newIndex;
		high = myTable.size;
		
		/**
		 * Block Logic: Two-pass probing (forward then wraparound).
		 */
		for(j = 0; j < 2; j++) {
			for (i = low; i < high; i++) {
				old = atomicCAS(&myTable.list[i].key, 0, keys[index]);
				if (old == keys[index] || old == 0) {
					myTable.list[i].value = values[index];
					return;
				}
			}
			low = 0;
			high = newIndex;
		}

	}
}

/**
 * Block Logic: Parallel batch lookup kernel.
 * Logic: Searches for matching keys starting from the hash index using linear probing.
 */
__global__ void get_kerFunc(hashtable myTable, int *keys, int *values, int limit) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j;
	int newIndex;
	int low, high;

	if (index < limit) {
		newIndex = myhashKernel(keys[index], myTable.size);
		low = newIndex;
		high = myTable.size;
		for(j = 0; j < 2; j++) {
			for (i = low; i < high; i++) { 
				if (myTable.list[i].key == keys[index]) {
					values[index] = myTable.list[i].value;
					return;
				}
			}
			low = 0;
			high = newIndex;	
		}
	}
}

/**
 * Block Logic: Entry migration kernel for reshaping.
 * Logic: Re-hashes non-empty elements from the source table into the 
 * destination table at their new positions.
 */
__global__ void reshapeKerFunc(hashtable oldTable, hashtable newTable, int size) {
	int newIndex, i, j;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int low, high;

	if (index >= oldTable.size)
		return;

	if (oldTable.list[index].key == 0) return;

	newIndex = myhashKernel(oldTable.list[index].key, size);
	low = newIndex;
	high = size;
	for(j = 0; j < 2; j++) {
		for (i = low; i < high; i++) {
			if (atomicCAS(&newTable.list[i].key, 0,
				oldTable.list[index].key) == 0) {
				newTable.list[i].value = oldTable.list[index].value;
				return;
			}
		}
		low = 0;
		high = newIndex;
	}
}

/**
 * Functional Utility: Initializes the GPU hash table object.
 * Logic: Allocates device memory for buckets and clears them.
 */
GpuHashTable::GpuHashTable(int size) {
	myTable.size = size;
	myTable.slotsTaken = 0;

	DIE(cudaMalloc(&myTable.list, size * sizeof(node)) != cudaSuccess, "BOO COULDNT ALLOC MYTABLE LIST INIT\n"); 
	cudaMemset(myTable.list, 0, size * sizeof(node));
}

/**
 * Functional Utility: Reclaims device memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(myTable.list);
}

/**
 * Functional Utility: Increases table capacity and migrates data.
 * Logic: Allocates a new larger bucket array, executes the reshape kernel, 
 * and synchronizes the device before updating internal state.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hashtable newTable;
	int noBlocks;
	newTable.size = numBucketsReshape;
	newTable.slotsTaken = myTable.slotsTaken;

	DIE(cudaMalloc(&newTable.list, numBucketsReshape * sizeof(node)) != cudaSuccess, "BOO COULDNT ALLOC MYTABLE LIST INIT\n"); 	
	cudaMemset(newTable.list, 0, numBucketsReshape * sizeof(node));



	noBlocks = (myTable.size % BLOCK_SIZE != 0) ?
		(myTable.size / BLOCK_SIZE + 1) : (myTable.size / BLOCK_SIZE);
	reshapeKerFunc>>(myTable, newTable, numBucketsReshape);
	
	cudaDeviceSynchronize();
	cudaFree(myTable.list);
	myTable = newTable;
}

/**
 * Functional Utility: Orchestrates high-throughput parallel insertions.
 * Logic: Checks capacity and triggers reshape if load exceeds 95%. 
 * Transfers batch data to GPU and launches the insertion kernel.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *device_keys, *device_values;
	int noBlocks;
	myTable.slotsTaken += numKeys;

	size_t memSize = numKeys * sizeof(int);
	DIE(cudaMalloc(&device_keys, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE KEYS\n");
	DIE(cudaMalloc(&device_values, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE VALUES\n");

	if ((float(myTable.slotsTaken) / myTable.size) >= 0.95)
		reshape(int((myTable.slotsTaken) / 0.8));

	cudaMemcpy(device_keys, keys, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, memSize, cudaMemcpyHostToDevice);



	noBlocks = (numKeys % BLOCK_SIZE != 0) ?
		(numKeys / BLOCK_SIZE + 1) : (numKeys / BLOCK_SIZE);
	insert_kerFunc>>(device_keys, device_values, numKeys, myTable);

	cudaDeviceSynchronize();

	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}

/**
 * Functional Utility: Retrieves a batch of values for given keys in parallel.
 * Memory Hierarchy: Uses Managed Memory (Unified Memory) for results to 
 * simplify Host-Device data transfer.
 */
 int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_values;
    int *device_keys;
    int noBlocks;

	DIE(cudaMallocManaged(&device_values, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE VALUES\n");
	DIE(cudaMalloc(&device_keys, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE KEYS\n");

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	


	noBlocks = (numKeys % BLOCK_SIZE != 0) ?
		(numKeys / BLOCK_SIZE + 1) : (numKeys / BLOCK_SIZE);

	get_kerFunc >> (myTable, device_keys, device_values, numKeys);
	
	cudaDeviceSynchronize();
	
	cudaFree(device_keys);
	return device_values;
}

/**
 * Functional Utility: Returns the ratio of occupied buckets to total capacity.
 */
 float GpuHashTable::loadFactor() {
	return (myTable.size > 0) ? float(myTable.slotsTaken) / myTable.size : 0;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
