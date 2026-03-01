/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated, open-addressing hash table using CUDA.
 *
 * @details This hash table supports parallel batch insertion and retrieval of
 * key-value pairs. It uses a linear probing strategy for collision resolution,
 * with thread-safe slot acquisition handled by atomic Compare-And-Swap (atomicCAS)
 * operations. The table can also be dynamically resized on the GPU to maintain
 * an optimal load factor.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include <string>
#include "gpu_hashtable.hpp"

/**
 * @brief __device__ function to compute the hash of a key.
 * @param x The input key.
 * @param N The size of the hash table, used for the modulo operation.
 * @return The initial bucket index for the given key.
 * @details Uses a series of XOR-shift and multiply operations ("integer finalizer")
 * to improve bit distribution before mapping to the table size.
 */
__device__ long long getHash(int x, int N) {
        if (x < 0) x = -x;
        x = ((x >> 16) ^ x) * HASH_NO;
    	x = ((x >> 16) ^ x) * HASH_NO;
    	x = (x >> 16) ^ x;
		x = x % N;
    	return x;
}

/**
 * @brief __global__ kernel to rehash all elements from an old table to a new, larger one.
 * @param h1 The source hash table.
 * @param siz1 The size of the source table.
 * @param h2 The destination hash table.
 * @param siz2 The size of the destination table.
 * @details Each thread handles one entry from the old table. It reads the key-value
 * pair, re-computes its hash for the new table size, and uses linear probing with
 * atomicCAS to insert it into the new table.
 */
__global__ void reshape_hashT(hashTable h1, long siz1, hashTable h2, long siz2) {
	int vall, idx = blockIdx.x * blockDim.x + threadIdx.x, key1, key2;
	bool ok = false;

	// Each thread processes one slot from the old table. Ignore empty or out-of-bounds slots.
	if ((h1.pairs[idx].key == KEY_INVALID) || (siz1 <= idx))
		return;
	
	key2 = h1.pairs[idx].key;
	vall = getHash(key2, h2.size);

	// Linear probing from the initial hash value to the end of the new table.
	for (int i = vall; i < siz2; i++) {
		// Atomically try to claim an empty slot for the key.
		key1 = atomicCAS(&h2.pairs[i].key, KEY_INVALID, key2);
		if (key1 == KEY_INVALID) {
			h2.pairs[i].value = h1.pairs[idx].value;
			ok = true;
			break;
		} 
	}
	// If not inserted, wrap around and probe from the beginning of the table.
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

/**
 * @brief __global__ kernel for parallel insertion of a batch of key-value pairs.
 * @param k The number of keys to insert.
 * @param keys A device pointer to the keys array.
 * @param values A device pointer to the values array.
 * @param h The hash table to insert into.
 * @param siz The size of the hash table.
 * @details Each thread handles one key-value pair. It uses linear probing and
 * atomicCAS to find an empty slot or update an existing key. This implements
 * an "upsert" functionality.
 */
__global__ void insert_hash(int k, int *keys, int *values, hashTable h, long siz) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x, key, vall;
	
	if (k <= idx) return; // Grid-stride loop check; each thread handles one key.

	vall = getHash(keys[idx], siz);

	// Linear probing from the initial hash value to the end of the table.
	for (int i = vall; i < siz; i++) {
		// Atomically try to claim an empty slot or find the existing key.
		key = atomicCAS(&h.pairs[i].key, KEY_INVALID, keys[idx]);
		// If the slot was empty OR we found our key, update the value and return.
		if (key == KEY_INVALID || key == keys[idx]) {
			h.pairs[i].value = values[idx];
			return;
		} 
	}
	// If not inserted, wrap around and probe from the beginning of the table.
	for (int i = 0; i < vall; i++) {
		key = atomicCAS(&h.pairs[i].key, KEY_INVALID, keys[idx]);
		if (key == KEY_INVALID || key == keys[idx]) {
			h.pairs[i].value = values[idx];
			return;
		}
	}
}

/**
 * @brief __global__ kernel for parallel retrieval of values for a batch of keys.
 * @param k The number of keys to look up.
 * @param keys A device pointer to the keys array.
 * @param values A device pointer to the output values array.
 * @param h The hash table to search in.
 * @param siz The size of the hash table.
 * @details Each thread handles one key lookup. It uses linear probing to find the
 * key. If found, it writes the corresponding value to the output array. This is a
 * read-only operation on the hash table.
 */
__global__ void get_hash(int k, int *keys, int *values, hashTable h, long siz) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x, vall;
	if (k<=idx) return; // Grid-stride loop check.

	vall = getHash(keys[idx], siz);

	// Linear probing from the initial hash value to find the key.
	for (int i = vall; i < siz; i++) {
		if (h.pairs[i].key == keys[idx]) {
			values[idx] = h.pairs[i].value;
			return;
		} 
	}
	// If not found, wrap around and probe from the beginning.
	for (int i = 0; i < vall; i++) {
		if (h.pairs[i].key == keys[idx]) {
			values[idx] = h.pairs[i].value;
			return;
		}
	}
}

/**
 * @brief Host-side constructor for the GpuHashTable.
 * @param size The initial capacity of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	hashT.size = size;
	cntPairs = 0;
	hashT.pairs = nullptr;
	// Allocate memory on the GPU for the hash table array.
	cudaMalloc(&hashT.pairs, size * sizeof(pair));
	// Initialize all keys to KEY_INVALID (0).
	cudaMemset(hashT.pairs, 0, size * sizeof(pair));
}

/**
 * @brief Host-side destructor for the GpuHashTable.
 */
GpuHashTable::~GpuHashTable() {
	// Free the GPU memory.
	cudaFree(hashT.pairs);
}

/**
 * @brief Resizes the hash table on the GPU.
 * @param numBucketsReshape The new, larger capacity for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	// Determine the grid size for the reshape kernel launch.
	int k = hashT.size / THREADS_NO;  
	if (!(hashT.size % THREADS_NO == 0)) k = k + 1;
	
	hashTable newH;
	newH.size = numBucketsReshape;

	// Allocate a new, larger table on the GPU.
	cudaMalloc(&newH.pairs, numBucketsReshape * sizeof(pair));
	cudaMemset(newH.pairs, 0, numBucketsReshape * sizeof(pair));
	
	// Launch the kernel to re-hash elements from the old table to the new one.
	reshape_hashT<<>>(hashT, hashT.size, newH, newH.size);

	cudaDeviceSynchronize();

	// Free the old table memory and update the host-side pointer.
	cudaFree(hashT.pairs);
	hashT = newH;
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys A host pointer to an array of keys.
 * @param values A host pointer to an array of values.
 * @param numKeys The number of pairs to insert.
 * @return True on success.
 * @details Automatically triggers a reshape if the load factor exceeds a threshold.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *aKeys, *aValues, k = numKeys / THREADS_NO, nr = cntPairs + numKeys;
	if (numKeys % THREADS_NO != 0) k++;   

	// Allocate temporary device memory for the input batch.
	cudaMalloc(&aKeys, numKeys * sizeof(int));
	cudaMalloc(&aValues, numKeys * sizeof(int));

	// Check if the load factor will exceed the maximum threshold.
	if (nr / hashT.size >= LOADFACTOR_MAX) {
		// If so, grow the table to bring the load factor down to the minimum.
		reshape((int) (nr / LOADFACTOR_MIN));
	}

	// Copy data from host to device.
	cudaMemcpy(aKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(aValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	// Launch the insertion kernel.
	insert_hash<<>>(numKeys, aKeys, aValues, hashT, hashT.size);

	cudaDeviceSynchronize(); // Wait for the kernel to complete.
	cudaFree(aKeys);
	cudaFree(aValues);
	cntPairs += numKeys;
	return true;
}

/**
 * @brief Retrieves values for a batch of keys.
 * @param keys A host pointer to an array of keys.
 * @param numKeys The number of keys to look up.
 * @return A host-accessible pointer to an array of retrieved values.
 * @details The returned pointer is to managed memory and must be freed by the caller
 * after use with `cudaFree`.
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
	int *aKeys, *valls, k = numKeys / THREADS_NO;
	if (!(numKeys % THREADS_NO == 0)) k++;

	// Allocate temporary device memory for input keys.
	cudaMalloc(&aKeys, numKeys * sizeof(int));
	// Allocate managed memory for output values, accessible by both host and device.
	cudaMallocManaged(&valls, numKeys * sizeof(int));

	// Copy keys from host to device.
	cudaMemcpy(aKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	// Launch the retrieval kernel.
	get_hash<<>>(numKeys, aKeys, valls, hashT, hashT.size);
	cudaDeviceSynchronize(); // Wait for kernel to complete.
	cudaFree(aKeys);
	return valls;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float.
 */
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
