
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"




__global__ void m_insert(int* keys, int* values, int numKeys,
	int *h_keys, int* h_values, int limit)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int index;
	int old;

	
	if (i < numKeys && keys[i] != 0) {
		
		index = hash_func(keys[i], limit);

		while (true) {
			
			if (h_keys[index] == keys[i]) {
				h_values[index] = values[i];
				return;
			}

			
			old = atomicCAS(&h_keys[index], 0, keys[i]);
			if (old == 0) {
				h_values[index] = values[i];
				return;
			} else {
				index = (index + 1) % limit;
			}
		}
	}
}


__global__ void m_has(int* keys, int numKeys,
	int* h_keys, int limit, int* num)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int index;

	if (i < numKeys) {
		
		index = hash_func(keys[i], limit);

		while (h_keys[index]) {
			
			if (h_keys[index] == keys[i]) {
				atomicAdd(num, 1);
				return;
			}
			index = (index + 1) % limit;
		}
	}
}




__global__ void m_get(int* keys, int numKeys,
	int* h_keys, int* h_values, int limit)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int index;

	if (i < numKeys) {
		
		index = hash_func(keys[i], limit);

		while (true) {
			
			if (h_keys[index] == keys[i]) {
				keys[i] = h_values[index];
				return;
			} else {
				index = (index + 1) % limit;
			}
		}
	}
}


GpuHashTable::GpuHashTable(int size) {
	this->size = size;
	this->curr_size = 0;
	this->resize = 0;
	cudaMalloc((void **)&this->keys, size * sizeof(int));
	cudaMalloc((void **)&this->values, size * sizeof(int));
	cudaMallocManaged((void **)&this->num, sizeof(int));
}


GpuHashTable::~GpuHashTable() {
	cudaFree(this->keys);
	cudaFree(this->values);
	cudaFree(this->num);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	int *aux_keys = this->keys;
	int *aux_values = this->values;
	int old_size = this->size;

	
	if (numBucketsReshape curr_size)
		return;

	
	cudaMalloc((void **)&this->keys, numBucketsReshape * sizeof(int));
	cudaMalloc((void **)&this->values, numBucketsReshape * sizeof(int));
	this->size = numBucketsReshape;

	
	if (this->curr_size) {
		this->resize = 1;
		
		insertBatch(aux_keys, aux_values, old_size);
		this->resize = 0;
	}

	cudaFree(aux_keys);
	cudaFree(aux_values);
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *aux_keys, *aux_values;
	int num_blocks = (numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE;

	
	if (!this->resize) {
		cudaMalloc((void **)&aux_keys, numKeys * sizeof(int));
		cudaMalloc((void **)&aux_values, numKeys * sizeof(int));

		cudaMemcpy(aux_keys, keys,
			numKeys * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(aux_values, values,
			numKeys * sizeof(int), cudaMemcpyHostToDevice);
	} else {
		
		aux_keys = keys;
		aux_values = values;
	}

	
	if (!this->resize) {
		*this->num = 0;
		m_has>>(aux_keys,
			numKeys, this->keys, this->size, this->num);
		cudaDeviceSynchronize();
		this->curr_size += (numKeys - *this->num);
	}

	
	if (this->loadFactor() >= 0.01f * LOADFACTOR) {
		this->reshape((this->curr_size * 100) / LOADFACTOR


			+ 0.001f * this->curr_size);
	}

	
	m_insert>>(aux_keys,
		aux_values, numKeys, this->keys, this->values, this->size);
	cudaDeviceSynchronize();

	
	if (!this->resize) {
		cudaFree(aux_keys);
		cudaFree(aux_values);
	}
	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *batch;
	int *aux_key_values;
	int num_blocks = (numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE;

	
	batch = (int *)malloc(numKeys * sizeof(int));
	if (batch == NULL)
		return NULL;

	
	cudaMalloc((void **)&aux_key_values, numKeys * sizeof(int));
	cudaMemcpy(aux_key_values, keys,
		numKeys * sizeof(int), cudaMemcpyHostToDevice);

	
	m_get>>(aux_key_values,
		numKeys, this->keys, this->values, this->size);
	cudaDeviceSynchronize();

	cudaMemcpy(batch, aux_key_values,
		numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(aux_key_values);
	return batch;
}


float GpuHashTable::loadFactor() {
	return (1.0f * this->curr_size) / (1.0f * this->size);
}

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0
#define BLOCK_SIZE		256
#define LOADFACTOR		85

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)


__device__ int hash_func(int data, int limit) {
	return ((long)abs(data) * 1009llu) % 8495693897llu % limit;
}

class GpuHashTable
{
	private:
		int size, curr_size;
		int *keys, *values, *num;
		char resize;

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
