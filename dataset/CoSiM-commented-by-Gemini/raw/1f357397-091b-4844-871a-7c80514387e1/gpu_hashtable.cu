/**
 * @file gpu_hashtable.cu
 * @brief A self-contained, robust implementation of a GPU-accelerated hash table.
 * @details This file provides a complete and well-designed hash table using CUDA.
 * It uses an open-addressing, linear-probing collision strategy. Key features
 * include a fully device-side reshape kernel, correct and race-free atomic
 * operations for insertions/updates, and robust occupancy tracking using a
 * device-side atomic counter. The probing logic is clean and efficient.
 */
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_hashtable.hpp"

#define IN
#define OUT
#define FREE_SPOT -1
#define MAX_OK_LOAD 0.9f
#define HASH_MUL 653267llu
#define HASH_MOD 2740199326961llu
#define EFF_FACT 1.1

/**
 * @brief Computes a hash value for a given key on the device.
 */
__device__ unsigned long long hash_test(int key, unsigned long long mod) {
	return (1ull * key * HASH_MUL) % HASH_MOD % mod;
}

/**
 * @brief CUDA kernel to insert or update a batch of key-value pairs.
 * @param inserted A device counter, atomically incremented for each new insertion.
 * @details Each thread handles one key-value pair, using linear probing and a
 * race-free `atomicCAS` loop to find a slot. It correctly distinguishes between
 * new insertions and updates to existing keys.
 */
__global__ void kernel_insert(int *keys, int* values, int numberOfPairs,
	Entry* data, int data_size, int *inserted) {
	
	int index =  blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long h;
	int old_key;
	int range[2][2];
	
	if (index >= numberOfPairs)
		return;
	
	h = hash_test(keys[index], data_size);

	// Pre-computation: Define ranges for the two-part linear probe to handle wrap-around.
	range[0][0] = h;
	range[0][1] = data_size;
	range[1][0] = 0;
	range[1][1] = h;
	
	// Block Logic: Linear probe through the two ranges.
	for (int r = 0; r < 2; r++ ) {
		for (int alt = range[r][0]; alt < range[r][1]; ++alt) {
			old_key = atomicCAS(&data[alt].key, FREE_SPOT, keys[index]);

			// Invariant: If slot was empty, we claimed it. Increment count.
			if (old_key == FREE_SPOT) {
				atomicAdd(inserted, 1);
				data[alt].value = values[index];
				return;
			// Invariant: If slot already held our key, it's an update.
			} else if (old_key == keys[index]){
				data[alt].value = values[index];
				return;
			} else {
				// Collision: The slot was occupied by another key, continue probing.
				continue;
			}
		}
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 */
__global__ void kernel_get(IN int *keys, OUT int* values,
	IN int numberOfPairs, IN Entry* data, IN int data_size) {

	int index =  blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long h;
	int range[2][2];

	if (index >= numberOfPairs)
		return;

	h = hash_test(keys[index], data_size);
	range[0][0] = h;
	range[0][1] = data_size;
	range[1][0] = 0;
	range[1][1] = h;

	// Block Logic: Linear probe search across two ranges to handle wrap-around.
	for (int r = 0; r < 2; r++ ) {
		for (int alt = range[r][0]; alt < range[r][1]; ++alt) {
			// Pre-condition: Check for a key match.
			if (data[alt].key == keys[index]) {
				values[index] = data[alt].value;
				return;
			}
		}
	}
}

/**
 * @brief CUDA kernel to initialize all hash table entries to an empty state.
 */
__global__ void kernel_init_data(Entry *data, int data_size) {
	int index =  blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= data_size)
	   return;

	data[index].key = -1;
	data[index].value = -1;
}

/**
 * @brief CUDA kernel to rehash all elements from an old table to a new one.
 */
__global__ void kernel_rehash_transfer(Entry *old_data,
	 int old_size, Entry *new_data, int new_size) {

	int index =  blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long h;
	int range[2][2];
	int old_key;

	if (index >= old_size) 
	   return;

	if (old_data[index].key == FREE_SPOT)
		return;

	h = hash_test(old_data[index].key, new_size);
	range[0][0] = h;
	range[0][1] = new_size;
	range[1][0] = 0;
	range[1][1] = h;

	// Block Logic: Linear probing to find an empty slot in the new table.
	for (int r = 0; r < 2; r++ ) {
		for (int alt = range[r][0]; alt < range[r][1]; ++alt) {
			old_key = atomicCAS(&new_data[alt].key, FREE_SPOT, old_data[index].key);
			if (old_key == FREE_SPOT) {
				new_data[alt].value = old_data[index].value;
				return;
			}
		}
	}
}


GpuHashTable::GpuHashTable(int size) {
	cudaError_t err;
	int block_no = (size + this->block_size - 1) / this->block_size;

	// Memory Hierarchy: Allocate the main table storage on the device.
	err = cudaMalloc((void **) &this->data, size * sizeof(Entry));
	DIE(err != cudaSuccess, "cudaMalloc");

	// Functional Utility: Launch a kernel to initialize memory. `cudaMemset` could also be used.
	kernel_init_data<<<block_no, this->block_size>>>(this->data, size);
	cudaDeviceSynchronize();

	this->limit = size * 1ull;
	this->filled = 0;
}


GpuHashTable::~GpuHashTable() {
	if (this->data)
		cudaFree(this->data);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	numBucketsReshape *= EFF_FACT; // Apply an efficiency factor to overallocate.
	Entry *new_data;
	cudaError_t err;
	int block_no = (numBucketsReshape + this->block_size - 1) / this->block_size;

	err = cudaMalloc((void **) &new_data, numBucketsReshape * sizeof(Entry));
	DIE(err != cudaSuccess, "cudaMalloc");

	kernel_init_data<<<block_no, this->block_size>>>(new_data, numBucketsReshape);
	cudaDeviceSynchronize();
	
	int rehash_block_no = (this->limit + this->block_size - 1) / this->block_size;
	// Launch the fully device-side reshape kernel.
	kernel_rehash_transfer<<<rehash_block_no, this->block_size>>>(data, limit, new_data, numBucketsReshape);
	cudaDeviceSynchronize();

	err = cudaFree(this->data);
	DIE(err != cudaSuccess, "cudaFree");

	this->data = new_data;
	this->limit = numBucketsReshape;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int block_no;
	int *device_keys;
	int *device_values;
	cudaError_t err;

	// Pre-condition: Check if the load factor will exceed the maximum threshold and resize if so.
	if ((this->filled + numKeys) > MAX_OK_LOAD * this->limit) {
		reshape((this->filled + numKeys));
	}

	// Functional Utility: Use a device-side atomic counter to accurately track new insertions.
	int tally, *dev_tally;
	cudaMalloc((void **)&dev_tally, sizeof(int));
	tally = 0;
	cudaMemcpy(dev_tally, &tally, sizeof(int), cudaMemcpyHostToDevice);

	err = cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");
	err = cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	block_no = (numKeys + this->block_size - 1) / this->block_size;

	// Launch insertion kernel, passing the atomic counter.
	kernel_insert<<<block_no, this->block_size>>>(
		device_keys, device_values, numKeys, this->data, this->limit, dev_tally
	);
	
	cudaDeviceSynchronize();
	cudaMemcpy(&tally, dev_tally, sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(device_keys);
	cudaFree(device_values);

	// Update host-side counter with the precise number of new elements.
	this->filled += tally;
	cudaFree(dev_tally);
	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_values = NULL;
	int *device_keys = NULL;
	int *host_values = NULL;
	int block_no;
	cudaError_t err;

	err = cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");
	err = cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");

	host_values = (int *)malloc(numKeys * sizeof(int));
	
	block_no = (numKeys + this->block_size - 1) / this->block_size;
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	kernel_get<<<block_no, this->block_size>>>(device_keys, device_values, numKeys,
		 this->data, this->limit);
	cudaDeviceSynchronize();

	cudaMemcpy(host_values, device_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(device_values);
	cudaFree(device_keys);

	// The caller is responsible for freeing the returned host memory.
	return host_values;
}


float GpuHashTable::loadFactor() {
	float lf = 0.0f;
	if (limit > 0.0f) // Check against float to avoid warning, though limit is ull.
		lf = (float)((float)filled / (float)limit);
	return lf; 
}

unsigned long long GpuHashTable::occupancy() {
	return filled;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

#include 
#include 

using namespace std;

#define KEY_INVALID 0

#define DIE(assertion, call_description)                                       
	do {                                                                   
		if (assertion) {                                               
			fprintf(stderr, "(%s, %d): ", __FILE__, __LINE__);     
			perror(call_description);                              
			exit(errno);                                           
		}                                                              
	} while (0)

const size_t primeList[] = {2llu,
			    3llu,
...
			    18446744073709551557llu};




int hash1(int data, int limit)
{
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit)
{
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit)
{
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

class Entry
{
      public:
	int key;
	int value;
};




class GpuHashTable
{
      public:
	GpuHashTable(int size);
	void reshape(int sizeReshape);

	bool insertBatch(int *keys, int *values, int numKeys);
	int *getBatch(int *key, int numItems);

	float loadFactor();
	unsigned long long occupancy();
	void print(string info);

	~GpuHashTable();

	unsigned long long filled;
	unsigned long long limit;
	const int block_size = 1024;

	Entry *data;
};

#endif
