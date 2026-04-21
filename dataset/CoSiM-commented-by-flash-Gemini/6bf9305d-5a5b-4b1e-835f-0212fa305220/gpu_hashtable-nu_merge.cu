/**
 * @6bf9305d-5a5b-4b1e-835f-0212fa305220/gpu_hashtable-nu_merge.cu
 * @brief High-performance GPU-based hash table utilizing linear probing for collision resolution.
 * Functional Utility: Implements a thread-safe, massively parallel hash table using 
 * CUDA atomic operations. Supports batch insertion, retrieval, and dynamic reshaping 
 * to maintain an optimal load factor.
 * Domain: HPC Data Structures, GPGPU Algorithms.
 */

#include "gpu_hashtable.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Hash algorithm designed for uniform distribution across GPU buckets.
 * Logic: Uses large prime modular arithmetic to minimize clustering in linear probing.
 */
__host__ __device__ int hash_function(int data, int limit) {
	return ((long)abs(data) * 718326812383316683llu) % 8699590588571llu % limit;
}

/**
 * @brief Initializes GPU memory for the hash table state.
 * Memory Strategy: Uses Managed Memory (Unified Memory) for simplified host/device access 
 * to table meta-data and buckets.
 */
GpuHashTable::GpuHashTable(int size) {
	this->HTcontor = 0;
	this->HTmarime = size;

	
	cudaMallocManaged((void **) &this->device_valori, this->HTmarime * sizeof(int));
	cudaMallocManaged((void **) &this->device_chei, this->HTmarime * sizeof(int));
	cudaMemset(this->device_chei, KEY_INVALID, this->HTmarime);
}

GpuHashTable::~GpuHashTable() {
	
	cudaFree(this->device_valori);
	cudaFree(this->device_chei);
}

/**
 * @brief Dynamically resizes the hash table to accommodate more elements.
 * Logic: Performs a 'stop-the-world' migration where existing valid entries are 
 * collected on the host and re-inserted into a newly allocated, larger GPU buffer.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int **ht_aux = (int **)malloc(sizeof(int *) * 2);
	ht_aux[0] = (int *)malloc(this->HTcontor * sizeof(int));
	ht_aux[1] = (int *)malloc(this->HTcontor * sizeof(int));

	int idx = 0, numKeys = this->HTcontor;

	int i = 0;
	while (i < HTmarime){
		if (this->device_chei[i] != KEY_INVALID) {
			ht_aux[0][idx] = this->device_chei[i];
			ht_aux[1][idx] = this->device_valori[i];
			idx++;
		}
		i++;
	}

	this->HTcontor = 0;
	this->HTmarime = numBucketsReshape * 1.06f;

	cudaFree(this->device_chei);
	cudaFree(this->device_valori);

	cudaMallocManaged((void **) &this->device_valori, this->HTmarime * sizeof(int));
	cudaMallocManaged((void **) &this->device_chei, this->HTmarime * sizeof(int));
	cudaMemset(this->device_chei, KEY_INVALID, this->HTmarime);

	insertBatch(ht_aux[0], ht_aux[1], numKeys);

	for (int j = 0; j < 2; j++)
		free(ht_aux[j]);
}

/**
 * @kernel kernel_insert
 * @brief Concurrent insertion kernel with linear probing.
 * Thread Indexing: Linear 1D grid mapping work-items to input keys.
 * Synchronization: Uses `atomicCAS` (Compare-And-Swap) for thread-safe bucket reservation 
 * and `atomicExch` for value commitment.
 */
__global__ void kernel_insert(int *keys, int *values, int numKeys, int *htchei, int *htvalori, int HTmarime, int *HTcontor) {
	int h;
	int h_aux, aux, check = 0;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < numKeys) {
		int add_cheie = keys[idx], valueToAdd = values[idx];
		h = hash_function(add_cheie, HTmarime);
		
		/**
		 * Block Logic: Atomic linear probing loop.
		 * Invariant: Threads loop through buckets until an empty slot or existing key match is found.
		 */
		aux = atomicCAS(&htchei[h], KEY_INVALID, add_cheie);

		if (aux == KEY_INVALID || aux == add_cheie)
			check = 1;

		if (check == 1) {
			atomicAdd(HTcontor, 1);
			atomicExch(&htvalori[h], valueToAdd);
			return;
		}
		
		h_aux = h;
		h = (h + 1) % HTmarime;
		
		while (h != h_aux) {
			aux = atomicCAS(&htchei[h], KEY_INVALID, add_cheie);
			if (aux == KEY_INVALID || aux == add_cheie)
				check = 1;
			if (check == 1) {
				atomicAdd(HTcontor, 1);
				atomicExch(&htvalori[h], valueToAdd);
				return;
			}
			h = (h + 1) % HTmarime;
		}
	}
}

/**
 * @brief Orchestrates a batch insertion task on the GPU.
 * Functional Utility: Handles host-to-device memory transfers and grid configuration.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *device_chei = NULL;
	int *device_valori = NULL;
	int *device_HTcontor = NULL;

	cudaMalloc((void **) &device_chei, numKeys * sizeof(int));
	cudaMalloc((void **) &device_valori, numKeys * sizeof(int));
	cudaMallocManaged((void **) &device_HTcontor, sizeof(int));

	if (numKeys != 0) {
		if (this->HTcontor + numKeys > this->HTmarime)
			reshape(this->HTmarime + numKeys);

		cudaMemcpy(device_chei, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(device_valori, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(device_HTcontor, &this->HTcontor, sizeof(int), cudaMemcpyHostToDevice);
		
		int blockSize = 256;
		int gridSize = (numKeys + blockSize - 1) / blockSize;
		kernel_insert<<<gridSize, blockSize>>>(device_chei,device_valori,
			numKeys,this->device_chei,this->device_valori,this->HTmarime,device_HTcontor);
		cudaDeviceSynchronize();
		cudaMemcpy(&this->HTcontor, device_HTcontor, sizeof(int), cudaMemcpyDeviceToHost);
	}
	
	cudaFree(device_chei);
	cudaFree(device_valori);
	cudaFree(device_HTcontor);
	return true;
}

/**
 * @kernel kernel_get
 * @brief Concurrent retrieval kernel.
 * Logic: Linear probing search to find a key's associated value.
 */
__global__ void kernel_get(int *keys, int *values, int numKeys, int *hashTableKeys, int *hashTableValues, int HTmarime) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int new_key, h, h_aux;
	if (idx < numKeys) {
		new_key = keys[idx];
		h = hash_function(new_key, HTmarime);
		h_aux = h;

		if (hashTableKeys[h] == new_key) {
			values[idx] = hashTableValues[h];
			return;
		}
		h = (h + 1) % HTmarime;

		while (h != h_aux) {
			if (hashTableKeys[h] == new_key) {
				values[idx] = hashTableValues[h];
				return;
			}
			h = (h + 1) % HTmarime;
		}
	}
}

int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_chei = NULL;
	int *device_valori = NULL;

	cudaMalloc((void **) &device_chei, numKeys * sizeof(int));
	cudaMalloc((void **) &device_valori, numKeys * sizeof(int));
	cudaMemcpy(device_chei, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int *valori = (int *)calloc(numKeys, sizeof(int));
	cudaMemcpy(device_valori, valori, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int gridSize = (numKeys + blockSize - 1) / blockSize;
	kernel_get<<<gridSize, blockSize>>>(device_chei,device_valori,numKeys,
		this->device_chei,this->device_valori,this->HTmarime);
	cudaDeviceSynchronize();
	cudaMemcpy(valori, device_valori, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_chei);
	cudaFree(device_valori);

	return valori;
}

float GpuHashTable::loadFactor() {
	return HTcontor / (float)HTmarime;
}
