/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 * @details This file contains the CUDA kernels and host-side methods for a hash table
 * that operates on the GPU. It supports batch insertion and retrieval of key-value pairs.
 * The implementation uses open addressing with linear probing for collision resolution.
 *
 * @note The file structure is unconventional, with parts of a header file included
 * at the end. It also includes "test_map.cpp", which is unusual for a library file.
 *
 * Algorithm: Open addressing with linear probing.
 * - Hashing: Multiplicative hashing.
 * - Collision Resolution: Linear probing with wrap-around.
 * - Concurrency: Thread-safe insertions are handled using atomic compare-and-swap (atomicCAS) operations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <iostream>

#include "gpu_hashtable.hpp"

/// The minimum load factor before a reshape is considered.
#define minim 0.82
/// The maximum load factor before a reshape is triggered.
#define maxim 0.93
/// The number of threads per block for CUDA kernels.
#define num_threads 256


/**
 * @brief CUDA kernel to reshape the hash table.
 * @details This kernel is launched when the hash table's load factor exceeds a threshold.
 * It re-inserts all valid key-value pairs from the old table into a new, larger table.
 * The insertion uses linear probing and atomicCAS to handle concurrent writes from different threads.
 * @param oldHash The old hash table.
 * @param newHash The new, larger hash table.
 */
__global__ void do_reshape(hashtable oldHash, hashtable newHash) {
	//- Kernel Logic: Each thread is responsible for one entry in the old hash table.
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	//- Pre-condition: Ensure the thread is within the bounds of the old table and the key is valid.
	if (index < oldHash.size &&
		oldHash.table[index].key != KEY_INVALID) {
		int key = oldHash.table[index].key;

		//- Block Logic: Calculate the initial hash position in the new table.
		int hash = Hash(key, newHash.size);
		int size = newHash.size, i;
		
		
		/**
		 * Block Logic: Perform linear probing to find an empty slot in the new table.
		 * It starts from the hashed position and wraps around if it reaches the end.
		 * Invariant: The loop continues until an empty slot is found and claimed.
		 */
		for (i = hash; i < size; i++){
			//- Inline: Use atomicCAS to thread-safely insert the key into an empty slot.
			if (atomicCAS(&newHash.table[i].key, KEY_INVALID, key) == KEY_INVALID) {
				newHash.table[i].value = oldHash.table[index].value;
				return;
			}
		}
		for (i = 0; i < hash; i++){
			//- Inline: Use atomicCAS to thread-safely insert the key into an empty slot (wrap-around part).
			if (atomicCAS(&newHash.table[i].key, KEY_INVALID, key) == KEY_INVALID) {
				newHash.table[i].value = oldHash.table[index].value;
				return;
			}
		}
	}
}


/**
 * @brief CUDA kernel to insert a batch of key-value pairs.
 * @details Each thread handles the insertion of one key-value pair. The insertion uses
 * linear probing and atomicCAS to find an empty slot.
 * @note This implementation has a race condition for updating existing keys. If two threads
 * try to insert the same key, both might pass the atomicCAS, but the final value will be
 * determined by which thread writes last.
 * @param keys An array of keys to insert.
 * @param values An array of values corresponding to the keys.
 * @param hashmap The target hash table.
 * @param count The number of key-value pairs to insert.
 */
__global__ void insert(int *keys, int *values, hashtable hashmap, int count) {
	//- Kernel Logic: Each thread is responsible for one key-value pair from the input batch.
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	//- Pre-condition: Ensure the thread is within the bounds of the input arrays.
	if (index < count) {
		int key = keys[index];
		//- Block Logic: Calculate the initial hash position.
		int hash = Hash(key, hashmap.size);
		int size = hashmap.size, i;
		
		
		/**
		 * Block Logic: Perform linear probing to find a slot for the key.
		 * Invariant: The loop attempts to claim an invalid slot or finds the key if it already exists.
		 */
		for(i = hash; i < size; i++){
			//- Inline: Attempt to claim an empty slot. This is the first part of a potential race condition.
			atomicCAS(&hashmap.table[i].key, KEY_INVALID, key);
			//- Inline: Check if the slot now holds the key. This check is not atomic with the CAS above.
			if (hashmap.table[i].key == key) {
				hashmap.table[i].value = values[index];
				return;
			}
		}
		for(i = 0; i < hash; i++){
			//- Inline: Wrap-around part of the linear probing.
			atomicCAS(&hashmap.table[i].key, KEY_INVALID, key);
			if (hashmap.table[i].key == key) {
				hashmap.table[i].value = values[index];
				return;
			}
		}
	}
	
}

/**
 * @brief CUDA kernel to retrieve a batch of values for given keys.
 * @details Each thread handles the retrieval of one value. It performs linear probing
 * to find the corresponding key in the hash table.
 * @param keys An array of keys to look up.
 * @param values An array where the retrieved values will be stored.
 * @param hashmap The hash table to search in.
 * @param count The number of keys to look up.
 */
__global__ void get(int *keys, int *values, hashtable hashmap, int count) {
	//- Kernel Logic: Each thread is responsible for one key lookup.
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	//- Pre-condition: Ensure the thread is within the bounds of the input array.
	if (index < count) {
		int key = keys[index];
		//- Block Logic: Calculate the initial hash position.
		int hash = Hash(keys[index], hashmap.size);
		int size = hashmap.size, i;
		
		/**
		 * Block Logic: Perform linear probing to find the key.
		 * Invariant: The loop continues until the key is found or all possible slots are checked.
		 */
		for (i = hash; i < size; i++){
			if (hashmap.table[i].key == key) {
				values[index] = hashmap.table[i].value;
				return;
			}
		}
		//- Block Logic: Wrap-around part of the linear probing.
		for (i = 0; i < hash; i++){
			if (hashmap.table[i].key == key) {
				values[index] = hashmap.table[i].value;
				return;
			}
		}
	}
}

/**
 * @brief Constructs a GpuHashTable object.
 * @param size The initial capacity of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	h.real_size = 0;
	h.size = size;
	//- Functional Utility: Allocate memory for the hash table on the GPU.
	cudaMalloc(&h.table, size * sizeof(assoc));
	//- Functional Utility: Initialize all keys to KEY_INVALID (0).
	cudaMemset(h.table, 0, size * sizeof(assoc));
}

/**
 * @brief Destroys the GpuHashTable object.
 */
GpuHashTable::~GpuHashTable() {
	//- Functional Utility: Free the GPU memory used by the hash table.
	cudaFree(h.table);
}

/**
 * @brief Resizes and rehashes the hash table.
 * @param numBucketsReshape The new capacity of the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hashtable newh;
	newh.size = numBucketsReshape;
	newh.real_size = h.real_size;
	cudaMalloc(&newh.table, numBucketsReshape * sizeof(assoc));
	cudaMemset(newh.table, 0, numBucketsReshape * sizeof(assoc));

	//- Block Logic: Determine grid and block dimensions for the reshape kernel.
	//- Note: The if/else branches are identical, suggesting a copy-paste error or redundant code.
	if (h.size / num_threads * num_threads == h.size) {
		do_reshape<<>>(h, newh);
	} else {
		do_reshape<<>>(h, newh);
	}
	//- Functional Utility: Wait for the reshape kernel to complete.
	cudaDeviceSynchronize();
	//- Functional Utility: Free the old table and assign the new one.
	cudaFree(h.table);
	h = newh;
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys A pointer to the host array of keys.
 * @param values A pointer to the host array of values.
 * @param numKeys The number of key-value pairs in the batch.
 * @return Always returns false.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *deviceKeys, *deviceValues;
	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	cudaMalloc(&deviceValues, numKeys * sizeof(int));
	h.real_size += numKeys;

	
	//- Pre-condition: Check if the load factor exceeds the maximum threshold.
	if (float(h.real_size) >= (float)(h.size * (float)maxim))
		//- Block Logic: If so, reshape the table to a larger size based on the minimum load factor.
		reshape(int((h.real_size) / minim));

	
	//- Functional Utility: Copy keys and values from host to device memory.
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	//- Block Logic: Determine grid and block dimensions for the insert kernel.
	//- Note: The if/else branches are identical, suggesting a copy-paste error or redundant code.
	if (numKeys / num_threads * num_threads == numKeys) {
		insert<<<numKeys / num_threads, num_threads>>>
			(deviceKeys, deviceValues, h, numKeys);
	} else {
		insert<<<numKeys / num_threads + 1, num_threads>>>
			(deviceKeys, deviceValues, h, numKeys);
	}
	
	//- Functional Utility: Wait for the insert kernel to complete.
	cudaDeviceSynchronize();
	cudaFree(deviceKeys);
	cudaFree(deviceValues);
	return false;
}

/**
 * @brief Retrieves a batch of values corresponding to a batch of keys.
 * @param keys A pointer to the host array of keys to look up.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to a host array containing the retrieved values.
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {

	int *deviceKeys, *values;
	//- Functional Utility: Allocate managed memory for keys and values, accessible from both host and device.
	cudaMallocManaged(&deviceKeys, numKeys * sizeof(int));
	cudaMallocManaged(&values, numKeys * sizeof(int));

	
	//- Functional Utility: Copy keys from host to device.
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	//- Block Logic: Determine grid and block dimensions for the get kernel.
	//- Note: The if/else branches are identical, suggesting a copy-paste error or redundant code.
	if (numKeys / num_threads * num_threads == numKeys) {
		get<<<numKeys / num_threads, num_threads>>>
			(deviceKeys, values, h, numKeys);
	} else {
		get<<<numKeys / num_threads + 1, num_threads>>>
			(deviceKeys, values, h, numKeys);
	}
	//- Functional Utility: Wait for the get kernel to complete before returning values.
	cudaDeviceSynchronize();
	cudaFree(deviceKeys);
	return values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor (ratio of elements to capacity).
 */
float GpuHashTable::loadFactor() {
	if (h.size > 0)
		return (float(h.real_size) / h.size);
	return 0.f;
}


//- Architectural Intent: These macros provide a simplified C-style interface for the C++ GpuHashTable class.
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

//- Architectural Intent: This includes a test file directly, which is highly unconventional for a library.
#include "test_map.cpp"

#ifndef _HASHCPU_
#define _HASHCPU_

#include <stdio.h>
#include <stdlib.h>

/// Defines the value used to mark a slot in the hash table as empty.
#define KEY_INVALID 0

/// A prime number used as a multiplier in the hash function.
#define A 518509
/// A large prime number used as a modulus in the hash function to distribute keys.
#define B 6904869625999

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

/**
 * @brief The primary hash function.
 * @details Implements a multiplicative hash function. It takes an integer key,
 * multiplies it by a constant A, and then uses modulo operations to map it
 * to a value within the hash table's size limit.
 * @param data The input key.
 * @param limit The size of the hash table (upper bound for the hash value).
 * @return The calculated hash value.
 */
__device__ int Hash(int data, int limit) {
	return ((long long) abs(data) * A) % B % limit;
}

/// A list of prime numbers, likely for use in various hashing schemes.
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

/// An alternative hash function, likely for experiments in double hashing or other schemes.
int hash1(int data, int limit) {
	return ((long) abs(data) * primeList[64]) % primeList[90] % limit;
}

/// A second alternative hash function.
int hash2(int data, int limit) {
	return ((long) abs(data) * primeList[67]) % primeList[91] % limit;
}

/// A third alternative hash function.
int hash3(int data, int limit) {


	return ((long) abs(data) * primeList[70]) % primeList[93] % limit;
}

/**
 * @struct assoc
 * @brief Represents a single key-value pair in the hash table.
 */
struct assoc {
	/// The key. A value of KEY_INVALID indicates an empty slot.
	int key;
	/// The value associated with the key.
	int value;
};

/**
 * @struct hashtable
 * @brief Represents the internal state of the hash table on the GPU.
 */
struct hashtable {
	/// The total capacity of the hash table (number of slots).
	int size;
	/// The number of elements currently stored in the hash table.
	int real_size;
	/// A pointer to the array of key-value pairs on the GPU.
	assoc *table;
};

/**
 * @class GpuHashTable
 * @brief Host-side interface for managing the GPU hash table.
 */
class GpuHashTable {
	hashtable h;
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
