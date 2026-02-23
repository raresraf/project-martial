#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


__device__ int getHashCode(int key, int capacity) {


	return ((long long)abs(key) * 1646237llu) % 67965551447llu % capacity;
}

__global__ void kernel_insert(int *keys, int *values, int N, HashTable hashtable) {

	
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	
	if (i < N) {

		
		int idx = getHashCode(keys[i], hashtable.capacity);

		

		
		for (int k = idx; k < hashtable.capacity; k++) {
			int old_key = atomicCAS(&hashtable.nodes[k].key, 0, keys[i]);
			if (!old_key || old_key == keys[i]) {
				hashtable.nodes[k].value = values[i];
				return;
			}
		}

		
		for (int k = 0; k < idx; k++) {
			int old_key = atomicCAS(&hashtable.nodes[k].key, 0, keys[i]);
			if (!old_key || old_key == keys[i]) {
				hashtable.nodes[k].value = values[i];
				return;
			}
		}
	}
}

__global__ void kernel_reshape(HashTable old_hashtable, HashTable new_hashtable) {

	
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	
	if (i < old_hashtable.capacity) {

		
		if (old_hashtable.nodes[i].key != 0) {

			
			int idx = getHashCode(old_hashtable.nodes[i].key, new_hashtable.capacity);

			
			 
			

			
			for (int k = idx; k < new_hashtable.capacity; k++) {
				int old_key = atomicCAS(&new_hashtable.nodes[k].key, 0, old_hashtable.nodes[i].key);
	
				if (!old_key) {
					new_hashtable.nodes[k].value = old_hashtable.nodes[i].value;
					return;
				}
			}
	
			
			for (int k = 0; k < idx; k++) {
				int old_key = atomicCAS(&new_hashtable.nodes[k].key, 0, old_hashtable.nodes[i].key);
	
				if (!old_key) {
					new_hashtable.nodes[k].value = old_hashtable.nodes[i].value;
					return;
				}
			}
		}
	}
}

__global__ void kernel_get(int *keys, int *values, int N, HashTable hashtable) {
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	
	if (i < N) {

		
		int idx = getHashCode(keys[i], hashtable.capacity);

		

		
		for (int k = idx; k < hashtable.capacity; k++) {
			if (hashtable.nodes[k].key == keys[i]) {
				values[i] = hashtable.nodes[k].value;
				return;
			}
		}

		
		for (int k = 0; k < idx; k++) {
			if (hashtable.nodes[k].key == keys[i]) {
				values[i] = hashtable.nodes[k].value;
				return;
			}
		}
	}


}


GpuHashTable::GpuHashTable(int size) {

	
	hashtable.size = 0;
	
	
	hashtable.capacity = size;

	const int num_bytes = size * sizeof(Node);

	
	cudaMalloc(&hashtable.nodes, num_bytes);
	if (!hashtable.nodes) {
		return;
	}

	
	cudaMemset(hashtable.nodes, 0, num_bytes);
}


GpuHashTable::~GpuHashTable() {
	
	cudaFree(hashtable.nodes);
}


void GpuHashTable::reshape(int numBucketsReshape) {

	HashTable new_hashtable;
	
		
	new_hashtable.capacity = (int)(numBucketsReshape / 0.8f);

	
	new_hashtable.size = hashtable.size;

	const int num_bytes = new_hashtable.capacity * sizeof(Node);

	
	cudaMalloc(&new_hashtable.nodes, num_bytes);
	if (!new_hashtable.nodes) {
		return;
	}

	
	cudaMemset(new_hashtable.nodes, 0, num_bytes);

	

	unsigned int blocks_no = hashtable.capacity / 1024;

	
	if (hashtable.capacity % 1024) {
		++blocks_no;
	}

	
	kernel_reshape >> (hashtable, new_hashtable);

	cudaDeviceSynchronize();

	
	cudaFree(hashtable.nodes);

	
	hashtable = new_hashtable;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {

	const int num_bytes = numKeys * sizeof(int);

	int *device_keys = 0;
	int *device_values = 0;

	
	cudaMalloc(&device_keys, num_bytes);
	cudaMalloc(&device_values, num_bytes);

	if (!device_keys || !device_values) {
		
		return false;
	}

	
	float new_factor = (float)(hashtable.size + numKeys) / (float)hashtable.capacity;
	if (new_factor > 0.8f) {
		reshape(hashtable.size + numKeys);
	}

	
	cudaMemcpy(device_keys, keys, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, num_bytes, cudaMemcpyHostToDevice);

	
	
	unsigned int blocks_no = numKeys / 1024;

	
	if (numKeys % 1024) {
		++blocks_no;
	}

	


	kernel_insert>> (device_keys, 
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																									.json