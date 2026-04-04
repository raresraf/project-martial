/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table using a two-bucket linear probing scheme.
 *
 * @details This file contains a CUDA implementation of a hash table that uses two
 * separate tables ('buckets') to resolve collisions. It appears to be a hybrid
 * of linear probing and Cuckoo Hashing concepts. For a given hash location, it
 * first attempts to insert into bucket1, and if that fails, it tries the same
 * index in bucket2 before continuing the linear probe. The implementation
 * includes significant logical flaws, especially in the rehashing kernel.
 * The header (`.hpp`) content is inlined at the end of this file.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>

#include "gpu_hashtable.hpp" // Logically included, though physically at the end.

/**
 * @brief A simple multiplicative hash function executed on the GPU.
 * @param data The key to hash.
 * @param limit The size of the hash table, used for the final modulo operation.
 * @return The hash value (initial bucket index).
 */
__device__ int myHash(int data, int limit) {
	return ((long) abs(data) * 20906033) % 5351951779 % limit;
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs using a two-bucket scheme.
 *
 * @details Each thread handles one key-value pair. It uses a "two-chance" linear
 * probing strategy. For each probe index `i`, it tries to `atomicCAS` a slot in
 * `bucket1`. If that fails, it immediately tries the slot at the same index `i`
 * in `bucket2` before continuing the probe at `i+1`.
 *
 * @param keys Input array of keys.
 * @param values Input array of values.
 * @param numKeys The number of elements to insert.
 * @param bucket1 Pointer to the first hash table in global memory.
 * @param bucket2 Pointer to the second hash table in global memory.
 * @param bucketSize The capacity of a single bucket.
 */
__global__ void kernel_insert(int *keys, int *values, int numKeys, hash_entry* bucket1, hash_entry* bucket2, int bucketSize) {


	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// BUG: The boundary check should be against numKeys, not bucketSize.
	// This can lead to out-of-bounds access on the keys/values arrays.
	if (idx >= bucketSize) return; 
	int keyOld, keyNew;
	keyNew = keys[idx];
	int hash = myHash(keyNew, bucketSize);
	// --- Two-Chance Linear Probing with Wrap-Around ---
	// Phase 1: Probe from hash to end of buckets.
	for (int i = hash; i < bucketSize; i++) {
		keyOld = atomicCAS(&bucket1[i].key, KEY_INVALID, keyNew);
		if (keyOld == KEY_INVALID || keyOld == keyNew) {
			bucket1[i].value = values[idx];
			return;
		}
		keyOld = atomicCAS(&bucket2[i].key, KEY_INVALID, keyNew);
		if (keyOld == KEY_INVALID || keyOld == keyNew) {
			bucket2[i].value = values[idx];
			return;
		}
	}
	// Phase 2: Wrap around and probe from start to hash.
	for (int i = 0; i < hash; i++) {
		keyOld = atomicCAS(&bucket1[i].key, KEY_INVALID, keyNew);
		if (keyOld == KEY_INVALID || keyOld == keyNew) {
			bucket1[i].value = values[idx];
			return;
		}
		keyOld = atomicCAS(&bucket2[i].key, KEY_INVALID, keyNew);
		if (keyOld == KEY_INVALID || keyOld == keyNew) {
			bucket2[i].value = values[idx];
			return;
		}
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 *
 * @details Each thread handles one key, searching through `bucket1` and `bucket2`
 * at each index of a linear probe. This is a read-only operation.
 */
__global__ void kernel_get(int *keys, int *values, int numItems, hash_entry* bucket1, hash_entry* bucket2, int bucketSize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// BUG: The boundary check should be against numItems, not bucketSize.
	if (idx >= bucketSize) return;
	int crtKey = keys[idx];
	int hash = myHash(crtKey, bucketSize);
	// Search from hash to end.
	for (int i = hash; i < bucketSize; i++) {
		if (bucket1[i].key == crtKey) {
			values[idx] = bucket1[i].value;
			return;
		}
		if (bucket2[i].key == crtKey) {
			values[idx] = bucket2[i].value;
			return;
		}
	}
	// Wrap around and search from start to hash.
	for (int i = 0; i < hash; i++) {
		if (bucket1[i].key == crtKey) {
			values[idx] = bucket1[i].value;
			return;
		}
		if (bucket2[i].key == crtKey) {
			values[idx] = bucket2[i].value;
			return;
		}
	}
}

/**
 * @brief CUDA kernel to rehash elements into new, larger buckets.
 * @warning This kernel has a critical logic flaw. It uses the index `idx` from the
 * old buckets as the index for the new buckets, instead of calculating a new
 * hash based on the `newBucketSize`. This will lead to incorrect placement of
 * all elements during a reshape.
 */
__global__ void kernel_rehash(hash_entry* oldBucket1, hash_entry* oldBucket2, int oldBucketSize,
hash_entry* newBucket1, hash_entry* newBucket2, int newBucketSize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= oldBucketSize) return;
	if (oldBucket1[idx].key != KEY_INVALID) {
		int keyNew, keyOld, hash;
		keyNew = oldBucket1[idx].key;
		hash = myHash(keyNew, newBucketSize);
		bool completed = false;
		for (int i = hash; i < newBucketSize; i++) {
			keyOld = atomicCAS(&newBucket1[idx].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket1[idx].value = oldBucket1[idx].value;
				completed = true;
				break;
			}
			keyOld = atomicCAS(&newBucket2[idx].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket2[idx].value = oldBucket2[idx].value;
				completed = true;
				break;
			}
		}
		if (completed == false) {
			for (int i = 0; i < hash; i++) {
				keyOld = atomicCAS(&newBucket1[idx].key, KEY_INVALID, keyNew);
				if (keyOld == KEY_INVALID) {
					newBucket1[idx].value = oldBucket1[idx].value;
					break;
				}
				keyOld = atomicCAS(&newBucket2[idx].key, KEY_INVALID, keyNew);
				if (keyOld == KEY_INVALID) {
					newBucket2[idx].value = oldBucket2[idx].value;
					break;
				}
			}
		}
	}
	if (oldBucket2[idx].key != KEY_INVALID) {
		int keyNew, keyOld, hash;
		keyNew = oldBucket2[idx].key;
		hash = myHash(keyNew, newBucketSize);
		bool completed = false;
		for (int i = hash; i < newBucketSize; i++) {
			keyOld = atomicCAS(&newBucket1[idx].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket1[idx].value = oldBucket1[idx].value;
				completed = true;
				break;
			}
			keyOld = atomicCAS(&newBucket2[idx].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket2[idx].value = oldBucket2[idx].value;
				completed = true;
				break;
			}
		}
		if (completed == false) {
			for (int i = 0; i < hash; i++) {
			keyOld = atomicCAS(&newBucket1[idx].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket1[idx].value = oldBucket1[idx].value;
				break;
			}
			keyOld = atomicCAS(&newBucket2[idx].key, KEY_INVALID, keyNew);
			if (keyOld == KEY_INVALID) {
				newBucket2[idx].value = oldBucket2[idx].value;
				break;
			}
		}
		}
	}
}

// =================================================================================
// Host-side GpuHashTable Class Implementation
// =================================================================================

GpuHashTable::GpuHashTable(int size) {
	pairsInserted = 0;
	bucketSize = size;
	bucket1 = nullptr;
	bucket2 = nullptr;
	if (cudaMalloc(&bucket1, size * sizeof(hash_entry)) != cudaSuccess) {
		return;
	}
	cudaMemset(bucket1, 0, size * sizeof(hash_entry));
	if (cudaMalloc(&bucket2, size * sizeof(hash_entry)) != cudaSuccess) {
		return;
	}
	cudaMemset(bucket2, 0, size * sizeof(hash_entry));
}

GpuHashTable::~GpuHashTable() {
	cudaFree(bucket1);
	cudaFree(bucket2);
}

void GpuHashTable::reshape(int sizeReshape) {
	hash_entry* newBucket1;
	hash_entry* newBucket2;
	if (cudaMalloc(&newBucket1, sizeReshape * sizeof(hash_entry)) != cudaSuccess) {
		return;
	}
	cudaMemset(newBucket1, 0, sizeReshape * sizeof(hash_entry));
	if (cudaMalloc(&newBucket2, sizeReshape * sizeof(hash_entry)) != cudaSuccess) {
		return;
	}
	cudaMemset(newBucket2, 0, sizeReshape * sizeof(hash_entry));
	unsigned int numBlocks = bucketSize / THREADS_PER_BLOCK;
	if (bucketSize % THREADS_PER_BLOCK != 0) numBlocks++;
	// Calls the flawed rehash kernel.
	kernel_rehash<<<numBlocks, THREADS_PER_BLOCK>>>(bucket1, bucket2, bucketSize, newBucket1, newBucket2, sizeReshape);
	cudaDeviceSynchronize();
	cudaFree(bucket1);
	cudaFree(bucket2);
	bucket1 = newBucket1;
	bucket2 = newBucket2;
	bucketSize = sizeReshape;
}

bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *keysFromDevice, *valuesFromDevice;
	unsigned int numBlocks;
	if (cudaMalloc(&keysFromDevice, numKeys * sizeof(int)) != cudaSuccess) {
		return false;
	}
	if (cudaMalloc(&valuesFromDevice, numKeys * sizeof(int)) != cudaSuccess) {
		return false;
	}


	if ((pairsInserted + numKeys) * 1.0 / bucketSize >= MAX_LOAD)
		reshape(int((pairsInserted + numKeys) / MIN_LOAD));
	cudaMemcpy(keysFromDevice, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(valuesFromDevice, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK != 0) numBlocks++;


	kernel_insert<<<numBlocks, THREADS_PER_BLOCK>>>(keysFromDevice, valuesFromDevice, numKeys, bucket1, bucket2, bucketSize);
	cudaDeviceSynchronize();
	pairsInserted += numKeys;
	cudaFree(keysFromDevice);
	cudaFree(valuesFromDevice);
	return true;
}

int *GpuHashTable::getBatch(int *keys, int numKeys) {
	int *keysFromDevice, *valuesFromDevice;
	unsigned int numBlocks;
	if (cudaMalloc(&keysFromDevice, numKeys * sizeof(int)) != cudaSuccess) {
		std::cerr<< "Memory for keys failed to allocate\n";
		return nullptr;
	}
	if (cudaMalloc(&valuesFromDevice, numKeys * sizeof(int)) != cudaSuccess) {
		std::cerr<< "Memory for values failed to allocate\n";
		return  nullptr;
	}
	cudaMemcpy(keysFromDevice, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK != 0) numBlocks++;
	// BUG: The return value of this function is a GPU pointer, which cannot be
	// safely used by the host caller without a cudaMemcpy back to host memory.
	kernel_get<<<numBlocks, THREADS_PER_BLOCK>>>(keysFromDevice, valuesFromDevice, numKeys, bucket1, bucket2, bucketSize);
	cudaDeviceSynchronize();
	cudaFree(keysFromDevice);
	return valuesFromDevice;
}

float GpuHashTable::loadFactor() {
	if (bucketSize == 0) return 0.0;
	return pairsInserted * 1.0 / bucketSize;
}


#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

#include <string> 
#include <vector> 

#define THREADS_PER_BLOCK 1024
#define MIN_LOAD 0.5
#define MAX_LOAD 0.75
#define KEY_INVALID 0

#define DIE(assertion, call_description) \
    do {    \
        if (assertion) {    \
        fprintf(stderr, "(%s, %d): ",    \
        __FILE__, __LINE__);    \
        perror(call_description);    \
        exit(errno);    \
    }    \
} while (0)

const std::size_t primeList[] =
{
		2llu, 3llu, 5llu, 7llu, 11llu, 13llu, 17llu, 23llu, 29llu, 37llu, 47llu,
		59llu, 73llu, 97llu, 127llu, 151llu, 197llu, 251llu, 313llu, 397llu,
		499llu, 631llu, 797llu, 1009llu, 1259llu, 1597llu, 2011llu, 2539llu,
		3203llu, 4027llu, 5087llu, 6421llu, 8089llu, 10193llu, 12853llu, 16193llu,
		20399llu, 25717llu, 32401llu, 40823llu, 51437llu, 64811llu, 81649llu,
		102877llu, 129607llu, 163307llu, 205759llu, 259229llu, 326617llu,
		411527llu, 518509llu, 653267llu, 823117llu, 1037059llu, 1306601llu,
		1646237llu, 2074129llu, 2613229llu, 3292489llu, 4148279llu, 5226491llu,
		6584983llu, 8296553llu, 10453007llu, 13169977llu, 16593127llu, 20906033llu,
		26339969llu, 33186281llu, 41812097llu, 52679969llu, 66372617llu,
		83624237llu, 105359939llu, 132745199llu, 167248483llu, 210719881llu,
		265490441llu, 334496971llu, 421439783llu, 530980861llu, 668993977llu,
		842879579llu, 1061961721llu, 1337987929llu, 1685759167llu, 2123923447llu,
		2675975881llu, 3371518343llu, 4247846927llu, 5351951779llu, 6743036717llu,
		8495693897llu, 10703903591llu, 13486073473llu, 16991387857llu,
		21407807219llu, 26972146961llu, 33982775741llu, 42815614441llu,
		53944293929llu, 67965551447llu, 85631228929llu, 107888587883llu,
		135931102921llu, 171262457903llu, 215777175787llu, 271862205833llu,
		342524915839llu, 431554351609llu, 543724411781llu, 685049831731llu,
		863108703229llu, 1087448823553llu, 1370099663459llu, 1726217406467llu,
		2174897647073llu, 2740199326961llu, 3452434812973llu, 4349795294267llu,
		5480398654009llu, 6904869625999llu, 8699590588571llu, 10960797308051llu,
		13809739252051llu, 17399181177241llu, 21921594616111llu, 27619478504183llu,
		34798362354533llu, 43843189232363llu, 55238957008387llu, 69596724709081llu,
		87686378464759llu, 110477914016779llu, 139193449418173llu,
		175372756929481llu, 220955828033581llu, 278386898836457llu,
		350745513859007llu, 441911656067171llu, 556773797672909llu,
		701491027718027llu, 883823312134381llu, 1113547595345903llu,
		1402982055436147llu, 1767646624268779llu, 2227095190691797llu,
		2805964110872297llu, 3535293248537579llu, 4454190381383713llu,
		5611928221744609llu, 7070586497075177llu, 8908380762767489llu,
		11223856443489329llu, 14141172994150357llu, 17816761525534927llu,
		22447712886978529llu, 28282345988300791llu, 35633523051069991llu,
		44895425773957261llu, 56564691976601587llu, 71267046102139967llu,
		89790851547914507llu, 113129383953203213llu, 142534092204280003llu,
		179581703095829107llu, 226258767906406483llu, 285068184408560057llu,
		359163406191658253llu, 452517535812813007llu, 570136368817120201llu,
		718326812383316683llu, 905035071625626043llu, 1140272737634240411llu,
		1436653624766633509llu, 1810070143251252131llu, 2280545475268481167llu,
		2873307249533267101llu, 3620140286502504283llu, 4561090950536962147llu,
		5746614499066534157llu, 7240280573005008577llu, 9122181901073924329llu,
		11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu
};




int hash1(int data, int limit) {
	return ((long) abs(data) * primeList[64]) % primeList[90] % limit;
}

int hash2(int data, int limit) {
	return ((long) abs(data) * primeList[67]) % primeList[91] % limit;
}

int hash3(int data, int limit) {
	return ((long) abs(data) * primeList[70]) % primeList[93] % limit;
}

struct hash_entry {
	int key;
	int value;
};

class GpuHashTable {
	int pairsInserted;
	hash_entry* bucket1;
	hash_entry* bucket2;
	int bucketSize;

public:
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