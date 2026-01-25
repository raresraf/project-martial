
#include 
#include 
#include 
#include 
#include 
#include 
#include "gpu_hashtable.hpp"


__device__ long long getHash(int x, int N) {
        if (x < 0) x = -x;
        x = ((x >> 16) ^ x) * HASH_NO;
    	x = ((x >> 16) ^ x) * HASH_NO;
    	x = (x >> 16) ^ x;
		x = x % N;
    	return x;
}


__global__ void reshape_hashT(hashTable h1, long siz1, hashTable h2, long siz2) {
	int vall, idx = blockIdx.x * blockDim.x + threadIdx.x, key1, key2;
	bool ok = false;
	if ((h1.pairs[idx].key == KEY_INVALID) || (siz1 <= idx))
		return;
	key2 = h1.pairs[idx].key;
	vall = getHash(key2, h2.size);
	for (int i = vall; i < siz2; i++) {


		key1 = atomicCAS(&h2.pairs[i].key, KEY_INVALID, key2);
		if (key1 == KEY_INVALID) {
			h2.pairs[i].value = h1.pairs[idx].value;
			ok = true;
			break;
		} 
	}
	if (!ok) {
		for (int i = 0; i <vall; i++) {


			key1 = atomicCAS(&h2.pairs[i].key, KEY_INVALID, key2);
			if (key1 == KEY_INVALID) {
				h2.pairs[i].value = h1.pairs[idx].value;
				break;
			}

		}
	}
}


__global__ void insert_hash(int k, int *keys, int *values, hashTable h, long siz) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x, key, vall;
	
	if (k <= idx) return;
	vall = getHash(keys[idx], siz);
	for (int i = vall; i < siz; i++) {
		key = atomicCAS(&h.pairs[i].key, KEY_INVALID, keys[idx]);
		if (key == KEY_INVALID || key == keys[idx]) {
			h.pairs[i].value = values[idx];
			return;
		} 
	}
	for (int i = 0; i < vall; i++) {
		key = atomicCAS(&h.pairs[i].key, KEY_INVALID, keys[idx]);
		if (key == KEY_INVALID || key == keys[idx]) {
			h.pairs[i].value = values[idx];
			return;
		}
	}
}


__global__ void get_hash(int k, int *keys, int *values, hashTable h, long siz) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x, vall;
	if (k<=idx) return;
	vall = getHash(keys[idx], siz);
	for (int i = vall; i < siz; i++) {
		if (h.pairs[i].key == keys[idx]) {
			values[idx] = h.pairs[i].value;
			return;
		} 
	}
	for (int i = 0; i < vall; i++) {
		if (h.pairs[i].key == keys[idx]) {
			values[idx] = h.pairs[i].value;
			return;
		}
	}
}


GpuHashTable::GpuHashTable(int size) {
	hashT.size = size;
	cntPairs = 0;
	hashT.pairs = nullptr;
	cudaMalloc(&hashT.pairs, size * sizeof(pair));
	cudaMemset(hashT.pairs, 0, size * sizeof(pair));
}


GpuHashTable::~GpuHashTable() {
	cudaFree(hashT.pairs);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	int k = hashT.size / THREADS_NO;  
	if (!(hashT.size % THREADS_NO == 0)) k = k + 1;
	hashTable newH;
	newH.size = numBucketsReshape;

	cudaMalloc(&newH.pairs, numBucketsReshape * sizeof(pair));
	cudaMemset(newH.pairs, 0, numBucketsReshape * sizeof(pair));
	reshape_hashT>>(hashT, hashT.size, newH, newH.size);

	cudaDeviceSynchronize();


	cudaFree(hashT.pairs);
	hashT = newH;
}


bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *aKeys, *aValues, k = numKeys / THREADS_NO, nr = cntPairs + numKeys;
	if (numKeys % THREADS_NO != 0) k++;   
	cudaMalloc(&aKeys, numKeys * sizeof(int));
	cudaMalloc(&aValues, numKeys * sizeof(int));

	if (nr / hashT.size >= LOADFACTOR_MAX) reshape((int) (nr / LOADFACTOR_MIN));

	cudaMemcpy(aKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(aValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	insert_hash>>(numKeys, aKeys, aValues, hashT, hashT.size);

	cudaDeviceSynchronize();
	cudaFree(aKeys);
	cudaFree(aValues);
	cntPairs += numKeys;
	return true;
}


int *GpuHashTable::getBatch(int *keys, int numKeys) {
	int *aKeys, *valls, k = numKeys / THREADS_NO;
	if (!(numKeys % THREADS_NO == 0)) k++;
	cudaMalloc(&aKeys, numKeys * sizeof(int));
	cudaMallocManaged(&valls, numKeys * sizeof(int));

	cudaMemcpy(aKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	get_hash>>(numKeys, aKeys, valls, hashT, hashT.size);
	cudaDeviceSynchronize();
	cudaFree(aKeys);
	return valls;
}


float GpuHashTable::loadFactor() {
	if (hashT.size == 0) return 0;
	return (float(cntPairs) / hashT.size);
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
#define THREADS_NO 1000
#define KEY_INVALID 0
#define HASH_NO 0x45d9f3b
#define LOADFACTOR_MIN 0.8
#define LOADFACTOR_MAX 0.94

#define DIE(assertion, call_description) \
    do {    \
        if (assertion) {    \
        fprintf(stderr, "(%s, %d): ",    \
        __FILE__, __LINE__);    \
        perror(call_description);    \
        exit(errno);    \
    }    \
} while (0)


struct pair {
	int key, value;
};

struct hashTable {
	pair *pairs;
	long size;
};




class GpuHashTable {
	public:
		long cntPairs;
		hashTable hashT;
		GpuHashTable(int size);
		void reshape(int sizeReshape);

		bool insertBatch(int *keys, int *values, int numKeys);
		int *getBatch(int *key, int numItems);
		float loadFactor();

		void occupancy();
		void print(std::string info);
		~GpuHashTable();
};

#endif
