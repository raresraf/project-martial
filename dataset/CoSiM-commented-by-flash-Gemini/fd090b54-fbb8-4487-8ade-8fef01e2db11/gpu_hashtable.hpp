
/**
 * @file gpu_hashtable.hpp
 * @brief Header and implementation for a GPU-accelerated hash table using open addressing.
 * 
 * Algorithm: Linear probing with circular wrap-around and prime-multiplicative hashing.
 * Memory Model: Device global memory for mapping storage.
 * Synchronization: Atomic compare-and-swap (atomicCAS) for thread-safe concurrent insertion.
 * Domain: HPC, Parallel Data Structures.
 */

#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0

/**
 * @brief Pre-computed prime numbers for robust hashing across different table sizes.
 */
const size_t primeList[] = {
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
	11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu};

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
			fprintf(stderr, "(%s, %d): ",	\
				__FILE__, __LINE__);	\
			perror(call_description);	\
			exit(errno);	\
		}	\
	} while (0)

typedef unsigned long long Entry;

/**
 * @class GpuHashTable
 * @brief Host-side controller for managing a device-resident hash table.
 */
class GpuHashTable
{
public:
	unsigned long size;
	unsigned int *num_elements;
	Entry *table;

	
	unsigned long a;
	unsigned long b;

	GpuHashTable(int size);
	void reshape(int sizeReshape);

	bool insertBatch(int *keys, int* values, int numKeys);
	int* getBatch(int* key, int numItems);

	float loadFactor();
	void occupancy();
	void print(string info);
	void printTable();

	~GpuHashTable();
};

#endif

#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Multiplicative hash function for GPU-side index calculation.
 * 
 * Time Complexity: O(1)
 */
__device__ int getHash(int val, int limit)
{
	return ((long long) abs(val) * 823117) % 3452434812973 % limit;
}

/**
 * @brief CUDA kernel for parallel entry insertion.
 * 
 * Logic: Employs linear probing with circular wrap-around. Uses atomicCAS 
 * to ensure that only one thread claims a specific empty slot.
 */
__global__ void addInKern(int *keys, int *val, int max, hash_table hash)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// Pre-condition: Thread index must be within batch bounds.
	if (index >= max) return;
	int actual_key, replacing_key;
	replacing_key = keys[index];
	int hash_idx = getHash(replacing_key, hash.dim);

	/**
	 * Block Logic: Forward probe sequence.
	 * Invariant: Searches from the initial hash index to the table boundary.
	 */
	for (int i = hash_idx; i < hash.dim; i++) {
		actual_key = atomicCAS(&hash.map[i].key, KEY_INVALID, replacing_key);

		if (actual_key == KEY_INVALID || actual_key == replacing_key) {
			hash.map[i].value = val[index];
			return;
		}
	}
	
	/**
	 * Block Logic: Wrap-around probe sequence.
	 * Invariant: Searches from the start of the table back to the original hash index.
	 */
	for (int i = 0; i < hash_idx; i++) {
		actual_key = atomicCAS(&hash.map[i].key, KEY_INVALID, replacing_key);

		if (actual_key == KEY_INVALID || actual_key == replacing_key) {
			hash.map[i].value = val[index];
			return;
		}
	}
}

/**
 * @brief CUDA kernel for parallel entry retrieval.
 */
__global__ void getFromKern(int *keys, int *val, int max, hash_table hash)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= max) return;

	int key = keys[index];
	int hash_idx = getHash(keys[index], hash.dim);

	/**
	 * Block Logic: Two-pass linear search.
	 * Pre-condition: Key is hashed to find the starting search index.
	 */
	for (int i = hash_idx; i < hash.dim; i++) {
		if (hash.map[i].key == key) {
			val[index] = hash.map[i].value;
			return;
		}
	}

	for (int i = 0; i < hash_idx; i++) {
		if (hash.map[i].key == key) {
			val[index] = hash.map[i].value;
			return;
		}
	}
}

/**
 * @brief CUDA kernel for re-hashing data during table resizing.
 */
__global__ void replaceHash(hash_table current_hash, hash_table replacing_hash)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= current_hash.dim) return;

	if (current_hash.map[index].key == KEY_INVALID) return;

	int actual_key, replacing_key;
	replacing_key = current_hash.map[index].key;

	int hash_idx = getHash(replacing_key, replacing_hash.dim);

	/**
	 * Block Logic: Re-insertion into the expanded table.
	 */
	for (int i = hash_idx; i < replacing_hash.dim; i++) {
		actual_key = atomicCAS(&replacing_hash.map[i].key, KEY_INVALID, replacing_key);

		if (actual_key == KEY_INVALID || actual_key == replacing_key) {
			replacing_hash.map[i].value = current_hash.map[index].value;
			return;
		}
	}

	for (int i = 0; i < hash_idx; i++) {
		actual_key = atomicCAS(&replacing_hash.map[i].key, KEY_INVALID, replacing_key);

		if (actual_key == KEY_INVALID || actual_key == replacing_key) {
			replacing_hash.map[i].value = current_hash.map[index].value;
			return;
		}
	}
}

/**
 * @brief Constructor: Allocates GPU memory for the hash table.
 */
GpuHashTable::GpuHashTable(int size)
{
	count = 0;
	hashmap.dim = size;

	cudaMalloc(&hashmap.map, size * sizeof(Entity));
	cudaMemset(hashmap.map, 0, size * sizeof(Entity));
}

/**
 * @brief Destructor: Releases device memory.
 */
GpuHashTable::~GpuHashTable()
{
	cudaFree(hashmap.map);
}

/**
 * @brief Resizes the hash table and re-populates it with existing entries.
 */
void GpuHashTable::reshape(int numBucketsReshape)
{
	hash_table hash;
	hash.dim = numBucketsReshape;

	cudaMalloc(&hash.map, numBucketsReshape * sizeof(Entity));
	cudaMemset(hash.map, 0, numBucketsReshape * sizeof(Entity));

	unsigned int numBlocks = hashmap.dim / THREADS_PER_BLOCK + 1;
	if (hashmap.dim % THREADS_PER_BLOCK != 0) numBlocks++;
	replaceHash<<<numBlocks, THREADS_PER_BLOCK>>>(hashmap, hash);

	cudaDeviceSynchronize();

	cudaFree(hashmap.map);
	hashmap = hash;
}

/**
 * @brief Batch parallel insertion from host data.
 * 
 * Optimization: Automatically triggers reshape if the load factor exceeds 0.75.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys)
{
	int *batch_keys, *batch_values;

	cudaMalloc(&batch_keys, numKeys * sizeof(int));
	cudaMalloc(&batch_values, numKeys * sizeof(int));

	if (float(count + numKeys) / hashmap.dim >= MAX_LOAD_FACTOR)
		reshape(int((count + numKeys) / MIN_LOAD_FACTOR));

	cudaMemcpy(batch_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(batch_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	unsigned int numBlocks = numKeys / THREADS_PER_BLOCK + 1;
	if (numKeys % THREADS_PER_BLOCK != 0) numBlocks++;
	addInKern<<<numBlocks, THREADS_PER_BLOCK>>>(batch_keys, batch_values, numKeys, hashmap);

	cudaDeviceSynchronize();

	count += numKeys;

	cudaFree(batch_keys);
	cudaFree(batch_values);

	return true;
}

/**
 * @brief Batch parallel retrieval into managed memory.
 */
int *GpuHashTable::getBatch(int *keys, int numKeys)
{
	int *batch_keys, *batch_values;

	size_t memSize = numKeys * sizeof(int);
	cudaMalloc(&batch_keys, memSize);
	// Memory Hierarchy: Unified memory for convenient host-side result processing.
	cudaMallocManaged(&batch_values, memSize);

	cudaMemcpy(batch_keys, keys, memSize, cudaMemcpyHostToDevice);

	unsigned int numBlocks = numKeys / THREADS_PER_BLOCK + 1;
	if (numKeys % THREADS_PER_BLOCK != 0) numBlocks++;
	getFromKern<<<numBlocks, THREADS_PER_BLOCK>>>(batch_keys, batch_values, numKeys, hashmap);

	cudaDeviceSynchronize();

	cudaFree(batch_keys);

	return batch_values;
}

/**
 * @brief Returns the ratio of used slots to total capacity.
 */
float GpuHashTable::loadFactor()
{
	return (hashmap.dim == 0)? 0 : (float(count) / hashmap.dim);
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
