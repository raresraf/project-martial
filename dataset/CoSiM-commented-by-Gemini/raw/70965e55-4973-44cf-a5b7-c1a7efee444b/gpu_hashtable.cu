/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 * @details This file implements a hash table on the GPU using open addressing with linear probing.
 * It provides batch insertion and retrieval capabilities, along with dynamic resizing to manage the load factor.
 * The implementation is designed for high-throughput key-value operations on an NVIDIA GPU.
 *
 * @note The code follows an unconventional structure by including a test file and defining
 * structures and class declarations that would typically be in a separate header file.
 *
 * Algorithm: Open addressing with linear probing.
 * - Hashing: Multiplicative hashing with large prime numbers.
 * - Collision Resolution: Linear probing with wrap-around, implemented via a do-while loop.
 * - Concurrency: Thread-safe insertions and updates are handled via atomicCAS operations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include <vector>

#include "gpu_hashtable.hpp"

/**
 * @brief Computes a hash value for a given key on the GPU.
 * @details This device function uses multiplicative hashing. It multiplies the absolute value of the key
 * by a small prime and then takes the result modulo a very large prime before scaling it to the table size.
 * This helps in distributing keys uniformly across the hash table.
 * @param data The integer key to be hashed.
 * @param limit The size of the hash table, which is the upper bound for the hash value.
 * @return The computed hash index.
 */
__device__ int computeHash(int data, int limit) {
    return ((long)abs(data) * 59llu) % 10703903591llu % limit;
}


/**
 * @brief CUDA kernel for batch insertion of key-value pairs.
 * @details Each thread in the grid is responsible for inserting one key-value pair. It uses linear probing
 * with a do-while loop to find an available slot. The `atomicCAS` operation ensures that each slot is claimed
 * by only one thread, and it also handles updates if the key already exists.
 * @param keys A GPU pointer to the array of keys.
 * @param values A GPU pointer to the array of values.
 * @param numKeys The total number of keys to insert.
 * @param hashtable The GPU hash table structure.
 */
__global__ void insert(int *keys, int *values, int numKeys, hash_table hashtable) {
	
    //- Kernel Logic: Each thread gets a unique index for a key-value pair.
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
 
    
    //- Pre-condition: Ensure the thread is within the bounds of the input data.
    if (i < numKeys) {

        unsigned int newKey = keys[i];
        unsigned int hash = computeHash(newKey, hashtable.size);

        unsigned int counter = hash;

        /**
         * Block Logic: Perform linear probing to find a slot.
         * Invariant: The loop continues until an empty slot is found or the key is updated.
         * It probes cyclically and terminates if it returns to the starting hash position,
         * which implies the table is full.
         */
        do {
            //- Inline: Atomically attempt to insert the new key into an empty (KEY_INVALID) slot.
            unsigned int key = atomicCAS(&hashtable.table[counter].key, KEY_INVALID, newKey);

            //- Inline: If the slot was empty or already contained our key, the operation is successful.
            if (key == KEY_INVALID || key == newKey) {
                hashtable.table[counter].value = values[i];
                return; 
            }
            //- Inline: Move to the next slot with wrap-around.
            counter = (counter + 1) % hashtable.size;
        } while (counter != hash);
    }
        
}


/**
 * @brief CUDA kernel for batch retrieval of values for given keys.
 * @details Each thread searches for a single key. The search uses the same linear probing strategy
 * as the insertion kernel. If a key is found, its value is written to the output array.
 * @param keys A GPU pointer to the array of keys to search for.
 * @param values A GPU pointer to the array where found values will be stored.
 * @param numKeys The number of keys to retrieve.
 * @param hashtable The GPU hash table structure.
 */
__global__ void get(int *keys, int *values, int numKeys, hash_table hashtable) {
    
    //- Kernel Logic: Each thread is assigned one key to look up.
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    
    //- Pre-condition: Ensure the thread is within the bounds of the input data.
    if (i < numKeys) {

        unsigned int searchKey = keys[i];
        unsigned int hash = computeHash(searchKey, hashtable.size);

        unsigned int counter = hash;

        /**
         * Block Logic: Perform linear probing to find the search key.
         * Invariant: The loop continues until the key is found or the probe cycle completes.
         */
        do {
            unsigned int currentKey = hashtable.table[counter].key;

            if (searchKey == currentKey) {
                values[i] = hashtable.table[counter].value;
                return; 
            }
            //- Inline: Move to the next slot with wrap-around.
            counter = (counter + 1) % hashtable.size;
        } while (counter != hash);
    }
    
}

/**
 * @brief CUDA kernel to rehash elements from an old table to a new, larger table.
 * @details Each thread is responsible for moving one element. It reads an entry from the old table,
 * re-computes its hash for the new table's size, and inserts it into the new table using the
 * same atomic linear probing strategy as the `insert` kernel.
 * @param newTable The destination hash table (on GPU).
 * @param oldTable The source hash table (on GPU).
 * @bug The linear probing wrap-around uses `oldTable.size` instead of `newTable.size`.
 *      This is incorrect and can lead to failed insertions if a probe needs to wrap
 *      around in a new table that is larger than the old one. The line should be:
 *      `counter = (counter + 1) % newTable.size;`
 */
__global__ void reshape_table(hash_table newTable, hash_table oldTable) {

    
    //- Kernel Logic: Each thread is assigned to one slot from the old table.
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    
    //- Pre-condition: Ensure thread is within bounds and the old slot is not empty.
    if (i < oldTable.size) {

        unsigned int oldKey = oldTable.table[i].key;

        if (oldKey == KEY_INVALID)
            return;

        //- Block Logic: Re-calculate the hash for the new table dimensions.
        unsigned int newHash = computeHash(oldKey, newTable.size);

        unsigned int counter = newHash;

        /**
         * Block Logic: Perform linear probing to insert the old element into the new table.
         * Invariant: The loop continues until an empty slot is found and claimed.
         */
        do {
            
            //- Inline: Atomically claim an empty slot in the new table.
            unsigned int newKey = atomicCAS(&newTable.table[counter].key, KEY_INVALID, oldKey);

            if (newKey == KEY_INVALID || newKey == oldKey) {
                newTable.table[counter].value = oldTable.table[i].value;
                return; 
            }

            //- Bug: The wrap-around logic here is incorrect. It should use `newTable.size`.
            counter = (counter + 1) % oldTable.size;
        } while (counter != newHash);
    }

}


/**
 * @brief Constructs a GpuHashTable object.
 * @param size The initial capacity (number of slots) for the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
    hashtable.size = size;
    hashtable.occupied = 0;

    //- Functional Utility: Allocate memory for the table on the GPU.
    cudaError_t rc = cudaMalloc(&hashtable.table, size * sizeof(entry));
    DIE(rc != cudaSuccess, cudaGetErrorString(rc));

    //- Functional Utility: Initialize all table entries to zero, marking them as empty.
    cudaMemset(hashtable.table, 0, size * sizeof(entry));
}


/**
 * @brief Destroys the GpuHashTable object and frees its GPU memory.
 */
GpuHashTable::~GpuHashTable() {
    cudaFree(hashtable.table);
}


/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The new size (number of buckets) for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {

    hash_table newHashTable;
    newHashTable.occupied = hashtable.occupied;
    newHashTable.size = numBucketsReshape;

    //- Functional Utility: Allocate memory for the new, larger table.
    cudaError_t rc = cudaMalloc(&newHashTable.table, numBucketsReshape * sizeof(entry));
    DIE(rc != cudaSuccess, cudaGetErrorString(rc));

    cudaMemset(newHashTable.table, 0, numBucketsReshape * sizeof(entry));

    //- Block Logic: Calculate the number of blocks needed for the reshape kernel.
    unsigned int blocks_no = hashtable.size / BLOCK_SIZE;
    if (hashtable.size % BLOCK_SIZE) 
        ++blocks_no;

    //- Kernel Launch: Launch the kernel to re-insert all elements.
    reshape_table<<<blocks_no, BLOCK_SIZE>>>(newHashTable, hashtable);

	cudaDeviceSynchronize();

    //- Functional Utility: Free the old table memory and update the class member.
    cudaFree(hashtable.table);

	hashtable = newHashTable;
}


/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys Host pointer to an array of keys.
 * @param values Host pointer to an array of values.
 * @param numKeys The number of items to insert.
 * @return Returns true on success. The `DIE` macro handles failures by exiting.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
    
    int *device_array_keys;
    int *device_array_values;

    size_t array_size = numKeys * sizeof(int);

    //- Functional Utility: Allocate GPU memory for the keys and values.
    cudaError_t rc = cudaMalloc(&device_array_keys, array_size);

    DIE(rc != cudaSuccess, cudaGetErrorString(rc));

    rc = cudaMalloc(&device_array_values, array_size);

    DIE(rc != cudaSuccess, cudaGetErrorString(rc));

	
	//- Pre-condition: Check if the load factor will exceed the threshold (0.8f).
	if (1.f * (hashtable.occupied + numKeys) / hashtable.size >= 0.8f)
		//- Block Logic: If so, resize the table to a larger capacity to maintain performance.
		reshape(int((hashtable.occupied + numKeys) / 0.6f));

	//- Functional Utility: Copy keys and values from the host to the device.
	cudaMemcpy(device_array_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_array_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	//- Block Logic: Calculate the number of blocks for the insert kernel launch.
	unsigned int blocks_no = hashtable.size / BLOCK_SIZE;
    if (hashtable.size % BLOCK_SIZE) 
        ++blocks_no;



	//- Kernel Launch: Launch the insertion kernel.
	insert<<<blocks_no, BLOCK_SIZE>>>(device_array_keys,
        device_array_values, numKeys, hashtable);


    cudaDeviceSynchronize();
    
    hashtable.occupied += numKeys;

	cudaFree(device_array_keys);
	cudaFree(device_array_values);

	return true;
}


/**
 * @brief Retrieves values for a batch of keys.
 * @param keys Host pointer to an array of keys to look up.
 * @param numKeys The number of keys to retrieve.
 * @return A host pointer to a newly allocated array containing the retrieved values. The caller is responsible for freeing this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {

	int *device_array_keys;
    int *device_array_values;
    int *host_array_values;

    //- Functional Utility: Allocate GPU memory for keys and values.
    cudaError_t rc = cudaMalloc(&device_array_keys, numKeys * sizeof(int));
    DIE(rc != cudaSuccess, cudaGetErrorString(rc));

    rc = cudaMalloc(&device_array_values, numKeys * sizeof(int));
    DIE(rc != cudaSuccess, cudaGetErrorString(rc));

    //- Functional Utility: Allocate host memory for the results.
    host_array_values = (int*)malloc(numKeys * sizeof(int));
    DIE(!host_array_values, "Failed host malloc in getBatch");


	//- Functional Utility: Copy keys from host to device.
	cudaMemcpy(device_array_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	//- Block Logic: Calculate the number of blocks for the get kernel.
	unsigned int blocks_no = hashtable.size / BLOCK_SIZE;
    if (hashtable.size % BLOCK_SIZE) 
        ++blocks_no;

	//- Kernel Launch: Launch the retrieval kernel.
	get<<<blocks_no, BLOCK_SIZE>>>(device_array_keys,
        device_array_values, numKeys, hashtable);


    cudaDeviceSynchronize();
    
    //- Functional Utility: Copy the retrieved values from device back to host.
    cudaMemcpy(host_array_values, device_array_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(device_array_keys);
	cudaFree(device_array_values);

	return host_array_values;
}


/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float (occupied slots / total size).
 */
float GpuHashTable::loadFactor() {
    
    if (hashtable.size == 0)
        return 0.f;



    return 1.f * hashtable.occupied / hashtable.size;
}



//- Architectural Intent: These macros define a simplified C-style API for the hash table.
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

//- Architectural Intent: This includes a C++ test file directly, an unconventional practice.
#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

/// Defines the value representing an empty slot in the hash table.
#define	KEY_INVALID		0

/// Defines the number of threads per CUDA block.
#define BLOCK_SIZE		1024

/**
 * @brief A macro for fatal error checking.
 * @details If the assertion is true, it prints an error description and exits.
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
	
/// A list of pre-selected prime numbers for use in hashing algorithms.
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




/// An alternative hash function, likely for experimentation.
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
 * @struct entry
 * @brief Represents a key-value pair in the hash table.
 */
struct entry {
	unsigned int key, value;
};

/**
 * @struct hash_table
 * @brief The core data structure for the GPU hash table.
 */
struct hash_table {
	/// The total capacity of the hash table.
	unsigned int size;
	/// The number of occupied slots in the hash table.
	unsigned int occupied;
	/// A GPU pointer to the array of key-value entries.
	entry *table;
};



/**
 * @class GpuHashTable
 * @brief The host-side API for managing and interacting with the GPU hash table.
 */
class GpuHashTable {
	hash_table hashtable;

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
