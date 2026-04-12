/**
 * @file gpu_hashtable.hpp
 * @brief Defines the structures and classes for CPU and GPU hash table implementations.
 * @details This header file contains a mix of definitions for both a CPU-based hash table
 * and a GPU-accelerated one using CUDA. It defines the main GpuHashTable class interface,
 * data structures, and includes kernel definitions that should typically reside in a .cu file.
 */

#ifndef _HASHCPU_
#define _HASHCPU_

#include <string>
#include <cstdio>
#include <cstdlib>
#include <iostream>

// The 'using namespace std;' is part of the original code and is preserved.
using namespace std;

/// @brief Defines the value representing an empty or invalid key slot in the hash table.
#define	KEY_INVALID		0

/**
 * @brief A list of large prime numbers.
 * @details This array provides a selection of prime numbers, likely intended for use in
 * various hashing algorithms to ensure good key distribution and reduce collisions.
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

/**
 * @brief A macro for basic error handling.
 * @details If the assertion is true, it prints a formatted error message
 * to stderr, including file and line number, and then exits the program.
 */
#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
			fprintf(stderr, "(%s, %d): ",	\
				__FILE__, __LINE__);	\
			perror(call_description);	\
			exit(errno);	\
		}	\
	} while (0)

/// @brief Defines a hash table entry as an unsigned long long.
typedef unsigned long long Entry;


/**
 * @class GpuHashTable
 * @brief Provides a host-side interface for managing a GPU-based hash table.
 * @details This class abstracts the complexities of CUDA memory management and
 * kernel launches, providing a cleaner API for hash table operations.
 */
class GpuHashTable
{
public:
	/// The total capacity of the hash table.
	unsigned long size;
	/// A pointer to a device memory location holding the count of elements.
	unsigned int *num_elements;
	/// A device pointer to the array of hash table entries.
	Entry *table;

	
	/// A parameter for the hash function, likely a multiplicative constant.
	unsigned long a;
	/// A parameter for the hash function, likely a prime modulus.
	unsigned long b;

	GpuHashTable(int size);
	~GpuHashTable();

	void reshape(int sizeReshape);
	bool insertBatch(int *keys, int* values, int numKeys);
	int* getBatch(int* key, int numItems);
	float loadFactor();
	void occupancy();
	void print(string info);
	void printTable();
};

#endif


#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define FIRST 823117
#define SECOND 3452434812973

__device__ int getHash(int val, int limit)
{
	return ((long long) abs(val) * FIRST) % SECOND % limit;
}

__global__ void addInKern(int *keys, int *val, int max, hash_table hash)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= max) return;
	int actual_key, replacing_key;
	replacing_key = keys[index];
	int hash = getHash(replacing_key, hash.dim);



	for (int i = hash; i < hash.dim; i++) {
		actual_key = atomicCAS(&hash.map[i].key, KEY_INVALID, replacing_key);

		if (actual_key == KEY_INVALID || actual_key == replacing_key) {
			hash.map[i].value = val[index];
			return;
		}
	}
	for (int i = 0; i < hash; i++) {
		actual_key = atomicCAS(&hash.map[i].key, KEY_INVALID, replacing_key);

		if (actual_key == KEY_INVALID || actual_key == replacing_key) {
			hash.map[i].value = val[index];
			return;
		}
	}
}

__global__ void getFromKern(int *keys, int *val, int max, hash_table hash)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= max) return;

	int key = keys[index];


	int hash = getHash(keys[index], hash.dim);

	for (int i = hash; i < hash.dim; i++) {
		if (hash.map[i].key == key) {
			val[index] = hash.map[i].value;
			return;
		}
	}

	for (int i = 0; i < hash; i++) {
		if (hash.map[i].key == key) {
			val[index] = hash.map[i].value;
			return;
		}
	}
}



__global__ void replaceHash(hash_table current_hash, hash_table replacing_hash)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= current_hash.dim) return;

	if (current_hash.map[index].key == KEY_INVALID) return;

	int actual_key, replacing_key;
	replacing_key = current_hash.map[index].key;

	int hash = getHash(replacing_key, replacing_hash.dim);

	for (int i = hash; i < replacing_hash.dim; i++) {
		actual_key = atomicCAS(&replacing_hash.map[i].key, KEY_INVALID, replacing_key);

		if (actual_key == KEY_INVALID || actual_key == replacing_key) {
			replacing_hash.map[i].value = current_hash.map[index].value;
			return;
		}
	}

	for (int i = 0; i < hash; i++) {
		actual_key = atomicCAS(&replacing_hash.map[i].key, KEY_INVALID, replacing_key);

		if (actual_key == KEY_INVALID || actual_key == replacing_key) {
			replacing_hash.map[i].value = current_hash.map[index].value;
			return;
		}
	}
}



GpuHashTable::GpuHashTable(int size)
{
	count = 0;
	hashmap.dim = dim;

	cudaMalloc(&hashmap.map, dim * sizeof(Entity));
	cudaMemset(hashmap.map, 0, dim * sizeof(Entity));
}

GpuHashTable::~GpuHashTable()
{
	cudaFree(hashmap.map);
}

void GpuHashTable::reshape(int numBucketsReshape)
{
	hash_table hash;
	hash.dim = numBucketsReshape;

	cudaMalloc(&hash.map, numBucketsReshape * sizeof(Entity));
	cudaMemset(hash.map, 0, numBucketsReshape * sizeof(Entity));

	unsigned int numBlocks = hashmap.dim / THREADS_PER_BLOCK + 1;
	if (hashmap.dim % THREADS_PER_BLOCK != 0) numBlocks++;
	replaceHash>>(hashmap, hash);

	cudaDeviceSynchronize();

	cudaFree(hashmap.map);
	hashmap = hash;
}

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
	addInKern>>(batch_keys, batch_values, numKeys, hashmap);

	cudaDeviceSynchronize();

	count += numKeys;

	cudaFree(batch_keys);
	cudaFree(batch_values);

	return true;
}

int *GpuHashTable::getBatch(int *keys, int numKeys)
{
	int *batch_keys, *batch_values;

	size_t memSize = numKeys * sizeof(int);
	cudaMalloc(&batch_keys, memSize);
	cudaMallocManaged(&batch_values, memSize);

	cudaMemcpy(batch_keys, keys, memSize, cudaMemcpyHostToDevice);

	unsigned int numBlocks = numKeys / THREADS_PER_BLOCK + 1;


	if (numKeys % THREADS_PER_BLOCK != 0) numBlocks++;
	getFromKern>>(batch_keys, batch_values, numKeys, hashmap);

	cudaDeviceSynchronize();

	cudaFree(batch_keys);

	return batch_values;
}

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
