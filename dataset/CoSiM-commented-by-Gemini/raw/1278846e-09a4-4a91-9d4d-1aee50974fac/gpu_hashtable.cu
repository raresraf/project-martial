/**
 * @file gpu_hashtable.cu
 * @brief A self-contained implementation of a GPU-accelerated hash table.
 * @details This file provides a complete hash table using CUDA, featuring an
 * open-addressing, linear-probing collision strategy. The implementation uses
 * robust, race-free atomic CAS operations for insertions and includes an
 * efficient, fully device-side reshape kernel. Its main performance drawback
 * is an inefficient pre-check in `insertBatch`, which calls `getBatch` to
 * determine the number of new keys before deciding whether to resize.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>

#include "gpu_hashtable.hpp"


/**
 * @brief Constructs a GpuHashTable object.
 */
GpuHashTable::GpuHashTable(int size) {
	hash_size = size; 
	num_entries = 0;  
	// Memory Hierarchy: Allocate the main hash table storage on the device.
	cudaMalloc((void **) &hashtable, size * sizeof(entry));
	// Initialize all keys to KEY_INVALID to mark slots as empty.
	cudaMemset(hashtable, KEY_INVALID, size * sizeof(entry));
}

/**
 * @brief Destroys the GpuHashTable object.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashtable);
}

/**
 * @brief Computes a hash value for a given key on the device.
 */
__device__ uint32_t hash_func(int data, int limit) {
	// Algorithm: Multiplicative hashing with large prime numbers.
	return ((long)abs(data) * 105359939) % 1685759167 % limit;
}


/**
 * @brief CUDA kernel to rehash all elements from an old table to a new one.
 * @param hashtable Device pointer to the old hash table data.
 * @param new_hash Device pointer to the new (empty) hash table data.
 * @param hash_size The capacity of the old table.
 * @param numBucketsReshape The capacity of the new table.
 * @details Each thread migrates one valid element, re-calculates its hash for the
 * new table size, and inserts it using linear probing and atomicCAS.
 */
__global__ void resize(GpuHashTable::entry *hashtable, GpuHashTable::entry *new_hash,
						int hash_size, int numBucketsReshape) {
	
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < hash_size) {
		// Pre-condition: Only process valid entries from the old table.
		if (hashtable[tid].key == KEY_INVALID)
			return;
		
		uint32_t key_hash = hash_func(hashtable[tid].key, numBucketsReshape);
		// Block Logic: Linear probing loop to find an empty slot in the new table.
		while (true) {
			// Atomic Operation: Attempt to claim an empty slot.
			uint32_t prev = atomicCAS(&new_hash[key_hash].key, KEY_INVALID, hashtable[tid].key);
			// Invariant: If the previous value was KEY_INVALID, this thread has successfully claimed the slot.
			if (prev == KEY_INVALID) {
				new_hash[key_hash].value = hashtable[tid].value;
				break;
			}
			// Collision: Probe the next slot with wrap-around.
			key_hash++;
			key_hash %= numBucketsReshape;
		}
	}
}

/**
 * @brief Resizes the hash table and rehashes all existing elements.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	// Kernel Launch Configuration: A fixed block size is used.
	uint32_t block_size = 100;
	uint32_t blocks_no = (hash_size + block_size - 1) / block_size;
	struct entry *new_hash;
	
	cudaMalloc((void **) &new_hash, numBucketsReshape * sizeof(entry));
	cudaMemset(new_hash, KEY_INVALID, numBucketsReshape * sizeof(entry));
	// Launch the fully device-side reshape kernel.
	resize<<<blocks_no, block_size>>>(hashtable, new_hash, hash_size, numBucketsReshape);
	cudaDeviceSynchronize();
	cudaFree(hashtable);
	hashtable = new_hash;
	hash_size = numBucketsReshape;
}


/**
 * @brief CUDA kernel to insert or update a batch of key-value pairs.
 * @details Each thread handles one pair, using linear probing and a race-free
 * `atomicCAS` loop to find a slot. It correctly handles both new insertions and
 * updates to existing keys.
 */
__global__ void insert(GpuHashTable::entry *hashtable, int hash_size,
						int *keys, int* values, int numKeys) {
	
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < numKeys) {
		
		uint32_t key_hash = hash_func(keys[tid], hash_size);
		while (true) {
			// Atomic Operation: Attempt to claim an empty slot or find a matching key.
			uint32_t prev = atomicCAS(&hashtable[key_hash].key, KEY_INVALID, keys[tid]);
			// Invariant: If slot was empty or already held our key, the write is safe.
			if (prev == keys[tid] || prev == KEY_INVALID) {
				hashtable[key_hash].value = values[tid];
				return;
			}
			// Collision: Probe the next slot.
			key_hash++;
			key_hash %= hash_size;
		}
	}
}

/**
 * @brief Inserts a batch of key-value pairs.
 * @warning The resizing strategy is inefficient. It performs a full `getBatch` call
 * (a device-to-host roundtrip) just to count new keys before insertion, which
 * creates a significant performance bottleneck.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *new_values;
	
	// Performance Issue: Calling getBatch here to pre-calculate the number of
	// new entries is very inefficient.
	new_values = getBatch(keys, numKeys);
	for (int i = 0; i < numKeys; i++)
		if (new_values[i] == KEY_INVALID)
			num_entries++;
	// Pre-condition: Check load factor and resize if it exceeds the threshold.
	if ((float)(num_entries) / hash_size >= 0.9)
		reshape(num_entries + (int)(0.1 * num_entries));

	// Kernel Launch Configuration
	uint32_t block_size = 100;
	uint32_t blocks_no = (numKeys + block_size - 1) / block_size;
	int *dev_keys = 0;
	int *dev_values = 0;
	
	cudaMalloc((void **) &dev_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &dev_values, numKeys * sizeof(int));
	// Data Transfer: Copy input data from host to device.
	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	// Launch insertion kernel.
	insert<<<blocks_no, block_size>>>(hashtable, hash_size, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize();
	cudaFree(dev_keys);
	cudaFree(dev_values);
	free(new_values);
	return true;
}


/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @details Each thread searches for one key. If the key is found, its value is
 * returned; otherwise, KEY_INVALID is returned. It correctly terminates the
 * search if an empty slot is found, as this signifies the key is not present.
 */
__global__ void get(GpuHashTable::entry *hashtable, int hash_size,
						int *keys, int *values, int numKeys) {
	
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < numKeys) {
		
		uint32_t key_hash = hash_func(keys[tid], hash_size);
		while (true) {
			// Pre-condition: Check if the key at the current slot matches.
			if (hashtable[key_hash].key == keys[tid]) {
				values[tid] = hashtable[key_hash].value;
				break;
			}
			// Invariant: If an empty slot is found, the key does not exist in the table.
			if (hashtable[key_hash].key == KEY_INVALID) {
				values[tid] = KEY_INVALID;
				break;
			}
			// Collision: Probe the next slot.
			key_hash++;
			key_hash %= hash_size;
		}
	}
}

/**
 * @brief Retrieves values for a batch of keys.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *results = (int *)malloc(numKeys * sizeof(int));
	// Kernel Launch Configuration
	uint32_t block_size = 100;
	uint32_t blocks_no = (numKeys + block_size - 1) / block_size;
	int *dev_keys = 0;
	int *dev_values = 0;
	
	cudaMalloc((void **) &dev_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &dev_values, numKeys * sizeof(int));
	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(dev_values, KEY_INVALID, numKeys * sizeof(int));
	// Launch the retrieval kernel.
	get<<<blocks_no, block_size>>>(hashtable, hash_size, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize();
	
	// Data Transfer: Copy results from device back to host.
	cudaMemcpy(results, dev_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_keys);
	cudaFree(dev_values);
	// The caller is responsible for freeing the returned host memory.
	return results;
}

/**
 * @brief Calculates the current load factor.
 */
float GpuHashTable::loadFactor() {
	return (float)num_entries / hash_size; 
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#define HASH_DESTROY GpuHashTable.~GpuHashTable();

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0

#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)
	
const size_t primeList[] =
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
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}




class GpuHashTable
{
	public:
		struct entry {
			uint32_t key;
			uint32_t value;
		};
		int hash_size;
		int num_entries;
		struct entry *hashtable;

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
