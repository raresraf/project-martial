/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 * @details This file provides a hash table data structure that leverages the parallelism of NVIDIA GPUs
 * through CUDA. It supports batch insertion, retrieval, and dynamic resizing (rehashing). The collision
 * resolution strategy is open addressing with linear probing.
 *
 * @note Similar to other examples in this dataset, the file structure is unconventional. It includes
 * a C++ test file directly and defines types and classes that would typically reside in a header file.
 *
 * Algorithm: Open addressing with linear probing.
 * - Hashing: Multiplicative hashing using large prime numbers.
 * - Collision Resolution: Linear probing with wrap-around.
 * - Concurrency: Thread-safe insertions are managed with atomic Compare-and-Swap (atomicCAS) operations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include <string>
#include <vector>

#include "gpu_hashtable.hpp"

/// A prime number used as a multiplier in the hash function.
#define PRIME_A 33186281l
/// A large prime number used as a modulus in the hash function to ensure good key distribution.
#define PRIME_B 350745513859007l


/**
 * @brief Computes a hash value for a given key on the GPU.
 * @details Implements a multiplicative hash function. The absolute value of the key is multiplied
 * by a prime constant, and the result is taken modulo another large prime, finally constrained
 * to the table size. This is a common technique to distribute keys evenly.
 * @param data The input integer key.
 * @param limit The size of the hash table, used to scale the hash value.
 * @return The computed hash index.
 */
__device__ int hashing(int data, int limit) {
	return ((long)abs(data) * PRIME_A) % PRIME_B % limit;
}


/**
 * @brief CUDA kernel for batch retrieval of values from the hash table.
 * @details Each thread processes one key from the input batch. It computes the hash and performs
 * linear probing to locate the key. If found, the corresponding value is written to the output array.
 * @param table The GPU hash table.
 * @param keys An array of keys to look up (on GPU).
 * @param values An array to store the found values (on GPU).
 * @param numKeys The total number of keys in the batch.
 */
__global__ void get(hashtable table, int *keys, int *values, int numKeys) {
	//- Kernel Logic: Assign one thread per key lookup.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int k, hash_idx;
	
	//- Pre-condition: Ensure the thread index is within the bounds of the key batch.
	if (idx >= numKeys)
		return;
	
	k  = keys[idx];
	
	//- Block Logic: Calculate the initial position in the hash table.
	hash_idx = hashing(k, table.size);
	
	/**
	 * Block Logic: Perform linear probing to find the key.
	 * Invariant: The loop scans from the initial hash index to the end of the table. If the key is
	 * present, its value is retrieved and the thread returns.
	 */
	for (int i = hash_idx; i < table.size; i++)
		if (table.ht[i].key == k) {
			values[idx] = table.ht[i].value;
			return;
		}
	/**
	 * Block Logic: Handle wrap-around case for linear probing.
	 * Invariant: If the key was not in the first segment, this loop scans from the beginning of the
	 * table up to the initial hash index.
	 */
	for (int i = 0; i < hash_idx; i++)
		if (table.ht[i].key == k) {
			values[idx] = table.ht[i].value;
			return;
		}
			
}

/**
 * @brief CUDA kernel for batch insertion of key-value pairs into the hash table.
 * @details Each thread processes one key-value pair. It uses atomicCAS to thread-safely claim an
 * empty slot (where key is 0) or update an existing key. It also counts keys that already existed.
 * @param table The GPU hash table.
 * @param keys An array of keys to insert (on GPU).
 * @param values An array of values to insert (on GPU).
 * @param numKeys The total number of key-value pairs in the batch.
 * @param numKeys_not_ins A counter (on GPU) to track the number of keys that already existed in the table.
 */
__global__ void insert(hashtable table, int *keys, int *values, int numKeys, int *numKeys_not_ins) {
	//- Kernel Logic: Assign one thread per key-value insertion.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int k, v, old_k, hash_idx;
	
	//- Pre-condition: Ensure the thread index is within the bounds of the key-value batch.
	if (idx >= numKeys) 
		return;
	
	k = keys[idx];
	v = values[idx];
	
	//- Block Logic: Calculate the initial hash position.
	hash_idx = hashing(k, table.size);
	
	/**
	 * Block Logic: Perform linear probing to find an empty slot or an existing key.
	 * Invariant: The loop continues until a slot is claimed for the key.
	 */
	for (int i = hash_idx; i < table.size; i++) {
		//- Inline: Atomically attempt to replace an empty slot (key 0) with the new key.
		old_k = atomicCAS(&table.ht[i].key, 0, k);
		
		//- Inline: If the key was already present, increment the 'not inserted' counter.
		if (old_k == k) {
			atomicAdd(&numKeys_not_ins[0], 1);
		}
		
		//- Inline: If the slot was empty (old_k=0) or already contained our key, write the value.
		if (old_k == 0 || old_k == k) {	
			table.ht[i].value = v;
			return;
		}
	}
	/**
	 * Block Logic: Handle wrap-around case for linear probing.
	 */
	for (int i = 0; i < hash_idx; i++) {
		old_k = atomicCAS(&table.ht[i].key, 0, k);
		
		if (old_k == k) {
			atomicAdd(&numKeys_not_ins[0], 1);
		}
		
		if (old_k == 0 || old_k == k) {
			table.ht[i].value = v;
			return;
		}
	}
	
}

/**
 * @brief CUDA kernel to rehash all elements from an old table to a new, larger one.
 * @details Each thread is responsible for moving one key-value pair. It reads from the old table
 * and uses the same atomic insertion logic as the `insert` kernel to place the item in the new table.
 * @param old_table The source hash table.
 * @param new_table The destination hash table.
 */
__global__ void rehash(hashtable old_table, hashtable new_table) {
	//- Kernel Logic: Assign one thread per slot in the old hash table.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int new_k, old_k, new_h_idx;
	
	//- Pre-condition: Check thread bounds against the old table's size.
	if (idx >= old_table.size) 
		return;
	
	//- Pre-condition: Skip empty slots in the old table.
	if (old_table.ht[idx].key == 0) 
		return;
	
	new_k = old_table.ht[idx].key;
	//- Block Logic: Calculate the hash position in the new, larger table.
	new_h_idx = hashing(new_k, new_table.size);
	
	/**
	 * Block Logic: Perform linear probing in the new table to insert the element.
	 * Invariant: The loop continues until an empty slot is found and claimed.
	 */
	for (int j = new_h_idx; j < new_table.size; j++) {
		//- Inline: Atomically claim an empty slot.
		old_k = atomicCAS(&new_table.ht[j].key, 0, new_k);
		
		if (old_k == 0) {
			new_table.ht[j].value = old_table.ht[idx].value;
			return;
		}
	}

	/**
	 * Block Logic: Handle wrap-around for linear probing in the new table.
	 */
	for (int j = 0; j < new_h_idx; j++) {
		old_k = atomicCAS(&new_table.ht[j].key, 0, new_k);
		
		if (old_k == 0) {
			new_table.ht[j].value = old_table.ht[idx].value;
			return;
		}
	}
	
}

/**
 * @brief Constructs a GpuHashTable.
 * @param size The initial capacity (number of buckets) of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	
	table.size = size;
	table.no_kvs = 0;
	
	//- Functional Utility: Allocate memory for the hash table buckets on the GPU.
	cudaError_t __cudaMalloc_err = cudaMalloc((void **) &(table.ht), size * sizeof(kvpair));
	
	if ( __cudaMalloc_err != cudaSuccess) 
		return;
	
	//- Functional Utility: Initialize the GPU memory to zero, marking all slots as empty.
	cudaMemset(table.ht, 0, size * sizeof(kvpair));		
}

/**
 * @brief Destroys the GpuHashTable, freeing its GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	
	cudaFree(table.ht);
}

/**
 * @brief Resizes the hash table to a new capacity and rehashes all elements.
 * @param numBucketsReshape The new number of buckets for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {

	int nblocks;
	hashtable new_table;
	
	new_table.size = numBucketsReshape;
	new_table.no_kvs = table.no_kvs;
	//- Functional Utility: Allocate memory for the new, larger table.
	cudaError_t __cudaMalloc_err = cudaMalloc((void **) &(new_table.ht), 
											numBucketsReshape * sizeof(kvpair));
	if ( __cudaMalloc_err != cudaSuccess) 
	
		return;
	cudaMemset(new_table.ht, 0, numBucketsReshape * sizeof(kvpair));
	
	//- Block Logic: Calculate the number of blocks needed for the rehash kernel.
	nblocks = table.size / BLOCKSIZE;
	if (table.size % BLOCKSIZE != 0) 
		nblocks++;

	//- Kernel Launch: Launch the rehash kernel to move elements.
	rehash<<<nblocks, BLOCKSIZE>>>(table, new_table);	
	
	//- Functional Utility: Wait for the kernel to finish.
	cudaDeviceSynchronize();
	cudaFree(table.ht);
	
	//- Functional Utility: Replace the old table with the new one.
	table = new_table;
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @details This method handles the entire batch insertion process, including potential resizing.
 * @param keys Host pointer to an array of keys.
 * @param values Host pointer to an array of values.
 * @param numKeys The number of key-value pairs to insert.
 * @return True on success, false on memory allocation failure.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {

	int nblocks;
	int num_buckets_reshape;
	int *device_keys;
	int *device_values;
	
	
	int *numKeys_not_ins;
	
	//- Functional Utility: Allocate memory on the GPU for keys, values, and a counter.
	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	cudaMallocManaged(&numKeys_not_ins, sizeof(int));
	if (!device_keys || !device_values || !numKeys_not_ins) {
		return false;
	}
	cudaMemset(numKeys_not_ins, 0, sizeof(int));
	
	//- Functional Utility: Copy data from host to device.
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	//- Pre-condition: Check if the load factor will exceed a threshold after insertion.
	if ( float(numKeys + table.no_kvs) / table.size > 0.85f) {
		//- Block Logic: If so, calculate a new size and reshape the table.
		num_buckets_reshape = int((numKeys + table.no_kvs) / 0.8);
		reshape(num_buckets_reshape);
	}
	
	//- Block Logic: Calculate the number of blocks for the insert kernel.
	nblocks = numKeys / BLOCKSIZE;
	if (numKeys % BLOCKSIZE != 0) 
		nblocks++;
	//- Kernel Launch: Launch the insert kernel.
	insert<<<nblocks, BLOCKSIZE>>>(table, device_keys, device_values, numKeys, numKeys_not_ins);
	
	cudaDeviceSynchronize();
	
	//- Block Logic: Update the count of elements in the table.
	table.no_kvs +=  numKeys - numKeys_not_ins[0];
	
	cudaFree(device_keys);
	cudaFree(device_values);
	cudaFree(numKeys_not_ins);
	return true;
}

/**
 * @brief Retrieves values for a batch of keys.
 * @param keys Host pointer to an array of keys to look up.
 * @param numKeys The number of keys to retrieve.
 * @return A host pointer to an array of retrieved values. The caller is responsible for freeing this memory.
 *         Returns NULL on memory allocation failure.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {

	int nblocks;
	int *device_keys;
	int *values;
	int *device_values;
	
	//- Functional Utility: Allocate host memory for the results.
	values = (int*)malloc(numKeys * sizeof(int));
	
	//- Functional Utility: Allocate device memory for keys and values.
	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaMalloc((void **)&device_values, numKeys * sizeof(int));
	if (!device_keys || !values || !device_values) {
		return NULL;
	}
	
	//- Functional Utility: Copy keys from host to device.
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	//- Block Logic: Calculate block count for the get kernel.
	nblocks = numKeys / BLOCKSIZE;
	if (numKeys % BLOCKSIZE != 0) 
		nblocks++;
	//- Kernel Launch: Launch the get kernel.
	get<<<nblocks, BLOCKSIZE>>>(table, device_keys, device_values, numKeys);
	
	//- Functional Utility: Copy the results from device to host.
	cudaMemcpy(values, device_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	
	cudaFree(device_keys);
	cudaFree(device_values);

	return values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor, which is the ratio of stored elements to total capacity.
 */
float GpuHashTable::loadFactor() {
	
	float load = 0;
	
	if (table.size > 0)	
		load = float(table.no_kvs) / table.size;
	
	return load; 
}


//- Architectural Intent: These macros simplify the API, providing a C-style interface.
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

//- Architectural Intent: This includes a test file directly into the source,
//- which is a highly unconventional practice for a library file.
#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

/// Defines the value used to represent an empty or invalid key in a hash table slot.
#define	KEY_INVALID		0
/// Defines the number of threads per CUDA block.
#define BLOCKSIZE 1024
/**
 * @brief A macro for error checking and exiting on failure.
 * @details Checks an assertion and, if it is true, prints an error message
 * including the file and line number, and then exits the program.
 */
#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)
	
/// A list of prime numbers used for hashing to reduce collisions.
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



/// An alternative hash function, likely for experimental use (e.g., double hashing).
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
/// A second alternative hash function.
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
/// A third alternative hash function.
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

/**
 * @struct _kvpair
 * @brief Defines a key-value pair for the hash table.
 */
typedef struct _kvpair { 
		/// The key for the entry. '0' is treated as invalid/empty.
		int key;	
		/// The value associated with the key.
		int value;
}kvpair;

/**
 * @struct _hashtable
 * @brief Defines the core structure of the GPU hash table.
 */
typedef struct _hashtable {
		/// The total capacity of the hash table.
		int size; 
		/// The number of key-value pairs currently stored.
		int no_kvs; 
		/// A pointer to the array of key-value pairs on the GPU.
		kvpair *ht; 
} hashtable;



/**
 * @class GpuHashTable
 * @brief The host-side interface for managing the GPU hash table.
 */
class GpuHashTable
{
	

	hashtable table;

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
