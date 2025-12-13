
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define dim sizeof(int)


 
GpuHashTable::GpuHashTable(int size) {
	mod = 0;
	totMem = 0;
	usedMem = (int *)malloc(dim);
	memset(usedMem, 0, 1);

	cudaMalloc((void **)&(hashMapKeys), mod * dim);
	cudaMemset((void **)&(hashMapKeys), 0, mod * dim);
		
	cudaMalloc((void **)&(hashMapValues), mod * dim);
	cudaMemset((void **)&(hashMapValues), 0, mod * dim);
	
	cudaMalloc((void **)&(usedMemDev), dim);
	cudaMemset((void **)&(usedMemDev), 0, dim);
	
}


GpuHashTable::~GpuHashTable() {
}


__global__ void insertHelp(int *hashMapKeys, int *hashMapValues,
	int *keysDev, int *valuesDev, int numKeys, int mod, int *usedMemDev, int *hashes) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;	

	if (i < numKeys) {
		int j;
		bool f = false;
		
		for (j = 0 ; j < mod - hashes[i]; j++) {
			
			
			if (atomicCAS(&hashMapKeys[j + hashes[i]], 0, keysDev[i]) == 0) {
				
				atomicAdd(usedMemDev, 1);
				hashMapValues[j + hashes[i]] = valuesDev[i];
                f = true;
                break;
			} else if (atomicCAS(&hashMapKeys[j + hashes[i]], 0, keysDev[i]) == keysDev[i]) {
				
				hashMapValues[j + hashes[i]] = valuesDev[i];
				f = true;
				break;
			}	
		}
		
		if (!f) {
			for (j = -hashes[i] ; j < 0 ; j++) {
				if (atomicCAS(&hashMapKeys[j + hashes[i]], 0, keysDev[i]) == 0) {
					atomicAdd(usedMemDev, 1);
					hashMapValues[j + hashes[i]] = valuesDev[i];
		            f = true;
		            break;
				} else if (atomicCAS(&hashMapKeys[j + hashes[i]], 0, keysDev[i]) == keysDev[i]) {
					hashMapValues[j + hashes[i]] = valuesDev[i];
					f = true;
					break;
				}	
			}
		}
	}
} 


 
 
 int getNonZeroKeysValues(int oldmod, int* nonZeroKeys, int *nonZeroValues, int *allKeys, int *allValues)
 {
	int nr = 0, i;
	for (i = 0 ; i < oldmod; i++) {
		if (allKeys[i] > 0) {
			nonZeroKeys[nr] = allKeys[i];
			nonZeroValues[nr++] = allValues[i];
		}
	}
	return nr;
 }




void GpuHashTable::reshape(int numBucketsReshape) {
	int oldmod = mod, nr;
	mod = numBucketsReshape;
	totMem = mod;
	
	int *nonZeroKeys, *nonZeroKeysCUDA, *allKeys;	
	int *nonZeroValues, *nonZeroValuesCUDA, *allValues;
	
	if (usedMem[0] > 0) {

		nonZeroKeys = (int *)malloc(oldmod * dim);
		allKeys = (int *)malloc(oldmod * dim);
		cudaMalloc((void**)&(nonZeroKeysCUDA), oldmod * dim);
		
		nonZeroValues = (int *)malloc(oldmod * dim);
		allValues = (int *)malloc(oldmod * dim);	
		cudaMalloc((void**)&(nonZeroValuesCUDA), oldmod * dim);
		
		cudaMemcpy(allKeys, hashMapKeys, oldmod * dim, cudaMemcpyDeviceToHost);
		cudaMemcpy(allValues, hashMapValues, oldmod * dim, cudaMemcpyDeviceToHost);
		nr = getNonZeroKeysValues(oldmod, nonZeroKeys, nonZeroValues, allKeys, allValues);
		
		cudaMemcpy(nonZeroKeysCUDA, nonZeroKeys, oldmod * dim,
			cudaMemcpyHostToDevice);
		cudaMemcpy(nonZeroValuesCUDA, nonZeroValues, oldmod * dim,
			cudaMemcpyHostToDevice);
	} 

	
	cudaMalloc((void **)&(hashMapKeys), mod * dim);
    cudaMemset((void **)&(hashMapKeys), 0, mod * dim);
    
    cudaMalloc((void **)&(hashMapValues), mod * dim);
    cudaMemset((void **)&(hashMapValues), 0, mod * dim);

	if (usedMem[0] > 0) {
		usedMem[0] = 0;
		cudaMemcpy(usedMemDev, usedMem, dim, cudaMemcpyHostToDevice);
    
    	int *hashes = (int *)calloc(nr, sizeof(int));
	
		int i;
		for (i = 0 ; i < nr ; i++) {
			hashes[i] = myHash(nonZeroKeys[i], mod);
		}
		
		int *hashesCUDA;
		cudaMalloc((void **)&hashesCUDA, nr * dim);
		cudaMemcpy(hashesCUDA, hashes, nr * dim, cudaMemcpyHostToDevice);
    
    	
		insertHelp>>(hashMapKeys, hashMapValues,
			nonZeroKeysCUDA, nonZeroValuesCUDA, nr, mod, usedMemDev, hashesCUDA);
		
		cudaDeviceSynchronize();
		cudaMemcpy(usedMem, usedMemDev, dim, cudaMemcpyDeviceToHost);
	}
}



bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	if (usedMem[0] + numKeys + (usedMem[0] + numKeys) * 0.05f > totMem) {
		reshape(usedMem[0] + numKeys + (usedMem[0] + numKeys) * 0.15f);
	}

	int *hashes = (int *)calloc(numKeys, sizeof(int));
	
	
	int i;
	for (i = 0 ; i < numKeys ; i++) {
		hashes[i] = myHash(keys[i], mod);
	}
	
	int *hashesCUDA;
	cudaMalloc((void **)&hashesCUDA, numKeys * dim);
	cudaMemcpy(hashesCUDA, hashes, numKeys * dim, cudaMemcpyHostToDevice);

	int *keysCUDA = 0;
	cudaMalloc((void **)&keysCUDA, numKeys * dim);
	cudaMemcpy(keysCUDA, keys, numKeys * dim, cudaMemcpyHostToDevice);
	
	int *valuesCUDA = 0;
	cudaMalloc((void **)&valuesCUDA, numKeys * dim);
	cudaMemcpy(valuesCUDA, values, numKeys * dim, cudaMemcpyHostToDevice);

	insertHelp>>(hashMapKeys, hashMapValues,
		keysCUDA, valuesCUDA, numKeys, mod, usedMemDev, hashesCUDA);
	
	cudaMemcpy(usedMem, usedMemDev, dim, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	return true;
}




__global__ void helperGetBatch(int *hashMapKeys, int *hashMapValues, int *retCUDA,
	int *keysDev, int numKeys, int mod, int *hashes) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < numKeys) {
		int j;
		bool f = false;
		
		for (j = 0 ; j < mod - hashes[i]; j++) {
			if (keysDev[i] == hashMapKeys[j + hashes[i]]) {
				retCUDA[i] = hashMapValues[j + hashes[i]];
				f = true;
				break;
			}	
		}
		
		if (!f) {
			for (j = -hashes[i] ; j < 0 ; j++) {
				if (keysDev[i] == hashMapKeys[j + hashes[i]]) {
					retCUDA[i] = hashMapValues[j + hashes[i]];
					f = true;
					break;
				}	
			}
		}
	}
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *ret = (int *)malloc(numKeys * dim);
	
	int *hashes = (int *)calloc(numKeys, sizeof(int));
	
	int i;
	for (i = 0 ; i < numKeys ; i++) {
		hashes[i] = myHash(keys[i], mod);
	}
	
	int *hashesCUDA;
	cudaMalloc((void **)&hashesCUDA, numKeys * dim);
	cudaMemcpy(hashesCUDA, hashes, numKeys * dim, cudaMemcpyHostToDevice);
	
	int *keysDev = 0;
	cudaMalloc((void **)&keysDev, numKeys * dim);
	cudaMemcpy(keysDev, keys, numKeys * dim, cudaMemcpyHostToDevice);
	
	int *retCUDA;
	cudaMalloc((void **)&retCUDA, numKeys * dim);



	helperGetBatch>>(hashMapKeys, hashMapValues,
		retCUDA, keysDev, numKeys, mod, hashesCUDA);
	cudaMemcpy(ret, retCUDA, numKeys * dim, cudaMemcpyDeviceToHost);


	cudaDeviceSynchronize();

	return ret;
}


float GpuHashTable::loadFactor() {
	return ((float)usedMem[0])/((float)totMem);
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

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)
	
const size_t primeList[] =
{
	5746614499066534157llu, 7240280573005008577llu, 9122181901073924329llu,
	11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu
};



int myHash(int data, int limit) {
	return ((long)abs(data) * primeList[1]) % primeList[3] % limit;
}
	



class GpuHashTable
{
	public:
		int *hashMapKeys;
		int *hashMapValues;

		int totMem;
		int *usedMem;
		int *usedMemDev;
		int mod;

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

