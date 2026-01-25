
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"
#define LOAD_FACTOR 0.8f




__global__ void insertHash(unsigned long long *hashmap, int cap, int *keys, int *values, int numKeys, unsigned int *num_inserted) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i < numKeys)
	{	


		unsigned long long key;
		
		key = keys[i];
		key = key << 32;
		key += values[i];
		int index = hash1(keys[i], cap);
		
		
		
		while(true)
		{
			atomicCAS(


				(unsigned long long *) &hashmap[index], 
				(unsigned long long) 0, 
				key);

			if(hashmap[index] != key)
			{
				unsigned long long k = hashmap[index] >> 32;
				if(k == key >> 32)
				{
					atomicCAS(&hashmap[index], hashmap[index], key);
					return;
				}	
				index = (index + 1) % cap;
			} else 
			{
				atomicInc(num_inserted, 4294967295);
				return;
			}			

		}
	}	
}


__global__ void getHash(unsigned long long *hash, int cap, int *keys, int *values, int numKeys) {
	
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i < numKeys)
	{
		int index = hash1(keys[i], cap);
		int start = index;
		
		
		while(true)
		{
			if(hash[index] >> 32 == keys[i])
			{
				unsigned long long temp = hash[index];
				temp = temp << 32;
				temp = temp >> 32;
				values[i] =(int) temp;
				return;
			} else if( hash[index] >> 32 == 0)
			{
				values[i] = 0;
				return;
			} else 
			{
				index = (index + 1) % cap;
				if(start == index)
				{
					values[i] = 0;
					return;
				}	
			}
		}
	}
	
}
 

GpuHashTable::GpuHashTable(int size) {
	cudaMalloc((void**) &hashmap, size * sizeof(unsigned long long));
	if(!hashmap)
	{
		perror("hashmap alloc");
		return;
	}

	


	cudaMemset(hashmap, 0, size * sizeof(unsigned long long));
	tot_size = size;
	num_elements = 0;
}


GpuHashTable::~GpuHashTable() {
	cudaFree(hashmap);
	hashmap = NULL;
	tot_size = 0;
	num_elements = 0;
}


void GpuHashTable::reshape(int numBucketsReshape) {
	
	int threads_num_block = 1024, num_blocks = tot_size / threads_num_block + 1;
	int *k_h = 0, *v_h = 0, *k_d = 0, *v_d = 0;
	unsigned long long *temp = 0;

	if(cudaMalloc((void**) &temp, numBucketsReshape * sizeof(unsigned long long)) != cudaSuccess)
	{
		perror("temp alloc");
		return;
	}

	
	cudaMemset(temp, 0, numBucketsReshape * sizeof(unsigned long long));

	
	if(num_elements == 0)
	{
		cudaFree(hashmap);
		hashmap = 0;
		hashmap = temp;
		tot_size = numBucketsReshape;
		return;
	}

	int num_el = 0;

	k_h = (int *) malloc (num_elements * sizeof(int));
	v_h = (int *) malloc (num_elements * sizeof(int));
	if( !k_h || !v_h)
	{
		perror("k_h v_h");
		return;
	}

	unsigned long long *hash_host = 0;
	hash_host = (unsigned long long *) malloc ( tot_size * sizeof(unsigned long long));
	cudaMemcpy(hash_host, hashmap, tot_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	if(!hash_host)
	{
		perror("hash_host alloc");
		return;
	}

	unsigned long long el;
	for(int i = 0; i < tot_size; i++)
	{


		el = hash_host[i];
		if(el >> 32 != 0)
		{
			k_h[num_el] = el >> 32;
			el = el << 32;
			el = el >> 32;
			v_h[num_el] = el;
			num_el++;
		}
	}
	



	if(cudaMalloc((void **) &k_d, num_el * sizeof(int)) != cudaSuccess)
		perror("cudaMalloc");

	if(cudaMalloc((void **) &v_d, num_el * sizeof(int)) != cudaSuccess)
		perror("cudaMalloc");
	
	cudaMemcpy(k_d, k_h, num_el * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(v_d, v_h, num_el * sizeof(int), cudaMemcpyHostToDevice);

	unsigned int *num_inserted = 0;


	cudaMalloc((void **) &num_inserted, sizeof(unsigned int));
	cudaMemset(num_inserted, 0, sizeof(unsigned int));

	insertHash>>(temp, numBucketsReshape, k_d, v_d, num_el, num_inserted);

	cudaDeviceSynchronize();
		
	hashmap = temp;
	tot_size = numBucketsReshape;

	cudaFree(k_d);
	k_d = 0;
	cudaFree(v_d);	
	v_d = 0;
	cudaFree(num_inserted);
	num_inserted = 0;
	free(k_h);
	k_h = 0;
	free(v_h);
	v_h = 0;
	temp = 0;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *k_d = 0, *v_d = 0;
	unsigned int *num_inserted_d = 0, *num_inserted_h = 0;
	int acc = 0, threads_num_block = 1024;	
	int num_blocks = numKeys / threads_num_block + 1;

	cudaMalloc((void**) &k_d, numKeys * sizeof(int));
	if(!k_d)
	{
		perror("key alloc");
		return false;
	}

	cudaMalloc((void**) &v_d, numKeys * sizeof(int));
	if(!v_d)
	{


		printf("value alloc");
		return false;
	}	

	cudaMemcpy(k_d, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(v_d, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);	

	for(int i = 0; i < numKeys; i++)
		if(keys[i] > 0 && values[i] > 0)
		acc++;		

	int new_cap;
	if( (acc + num_elements) / tot_size > LOAD_FACTOR) {
		new_cap = (acc + num_elements) / LOAD_FACTOR;
		reshape((int) 1.5 * new_cap);
	}


	num_inserted_h = (unsigned int *) malloc (sizeof(unsigned int));
	memset(num_inserted_h, 0, sizeof(unsigned int));



	cudaMalloc((void **) &num_inserted_d, sizeof(unsigned int));
	cudaMemcpy(num_inserted_d, num_inserted_h, sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	insertHash>>(hashmap, tot_size, k_d, v_d, numKeys, num_inserted_d);
	cudaDeviceSynchronize();

	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
		perror(cudaGetErrorString(error));

	cudaMemcpy(num_inserted_h, num_inserted_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);	
	num_elements += (*num_inserted_h);

	cudaFree(k_d);
	k_d = 0;
	cudaFree(v_d);
	v_d = 0;
	cudaFree(num_inserted_d);
	num_inserted_d = 0;
	free(num_inserted_h);
	num_inserted_h = 0;

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {


	int *k_d, *v_d, *values;
	int threads_num_block = 1024;
	int num_blocks = numKeys / threads_num_block + 1;

	if(cudaMalloc((void **) &k_d, numKeys * sizeof(int)) != cudaSuccess)
		perror("cudaMalloc");
	
	if(cudaMalloc((void **) &v_d, numKeys * sizeof(int)) != cudaSuccess)
		perror("cudaMalloc");

	cudaMemcpy(k_d, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	getHash>>(hashmap,tot_size,k_d, v_d, numKeys);

	values = (int *) malloc (numKeys * sizeof(int));
	cudaMemcpy(values, v_d, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(k_d);
	cudaFree(v_d);
	k_d = 0;
	v_d = 0;

	return values;
}


float GpuHashTable::loadFactor() {
	return num_elements / (tot_size*1.0f); 
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
