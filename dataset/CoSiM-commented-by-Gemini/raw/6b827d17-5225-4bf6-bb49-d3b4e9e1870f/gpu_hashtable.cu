/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 * @details This file implements a hash table designed to run on an NVIDIA GPU.
 * It uses open addressing with linear probing for collision resolution. The implementation
 * provides batch insertion and retrieval operations, along with dynamic resizing (reshaping)
 * when the load factor exceeds a certain threshold.
 *
 * @note The file includes "test_map.cpp" and defines its own structures and class
 * declarations, which is an unconventional layout for a reusable library.
 *
 * Algorithm: Open addressing with linear probing.
 * - Hashing: Multiplicative hashing with large prime numbers.
 * - Collision Resolution: Linear probing with a two-step wrap-around.
 * - Concurrency: Thread safety during insertion is achieved with atomicCAS operations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include <vector>

#include "gpu_hashtable.hpp"


/**
 * @brief Computes a hash for a given key on the GPU.
 * @details This device function implements a multiplicative hash. It multiplies the key's
 * absolute value by one prime number, takes the result modulo a second larger prime,
 * and finally scales it to the hash table's size.
 * @param data The integer key to hash.
 * @param limit The size of the hash table, defining the range of the output hash.
 * @return The calculated hash index.
 */
__device__ int myhashKernel(int data, int limit) {
	return ((long long)abs(data) * PRIMENO) % PRIMENO2 % limit;
}

/**
 * @brief CUDA kernel for inserting a batch of key-value pairs.
 * @details Each thread handles one key-value pair. It calculates the hash and then performs
 * linear probing to find a slot. `atomicCAS` is used to claim an empty slot or update
 * an existing one in a thread-safe manner.
 * @param keys GPU pointer to the array of keys to insert.
 * @param values GPU pointer to the array of values to insert.
 * @param limit The number of key-value pairs in the batch.
 * @param myTable The GPU hash table structure.
 */
__global__ void insert_kerFunc(int *keys, int *values, int limit, hashtable myTable) {
	int newIndex;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int old, i, j;
	int low, high;

	//- Pre-condition: Ensure the thread is within the bounds of the input batch.
	if (index < limit) {
		newIndex = myhashKernel(keys[index], myTable.size);
		
		/**
		 * Block Logic: Perform a two-step linear probe. The first step probes from the
		 * initial hash index to the end of the table. The second step (if needed)
		 * probes from the beginning of the table to the initial hash index (wrap-around).
		 * Invariant: The loop continues until a slot is secured for the key.
		 */
		low = newIndex;
		high = myTable.size;
		for(j = 0; j < 2; j++) {
			for (i = low; i < high; i++) {
				//- Inline: Atomically attempt to swap an empty slot (key 0) with the current key.
				old = atomicCAS(&myTable.list[i].key, 0, keys[index]);
				//- Inline: If the slot was empty or already held our key, the insertion/update is successful.
				if (old == keys[index] || old == 0) {
					myTable.list[i].value = values[index];
					return;
				}
			}
			//- Block Logic: Set up the ranges for the wrap-around probe.
			low = 0;
			high = newIndex;
		}

	}
}

/**
 * @brief CUDA kernel for retrieving values for a batch of keys.
 * @details Each thread searches for one key. The search follows the same two-step
 * linear probing strategy as the insertion kernel to locate the key.
 * @param myTable The GPU hash table structure.
 * @param keys GPU pointer to the array of keys to look up.
 * @param values GPU pointer to an array where the found values will be stored.
 * @param limit The number of keys in the batch.
 */
__global__ void get_kerFunc(hashtable myTable, int *keys, int *values, int limit) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j;
	int newIndex;
	int low, high;

	//- Pre-condition: Ensure the thread is within the bounds of the input batch.
	if (index < limit) {
		newIndex = myhashKernel(keys[index], myTable.size);
		
		/**
		 * Block Logic: Perform a two-step linear probe to find the key. The first step
		 * probes from the initial hash index to the end, and the second handles the wrap-around.
		 * Invariant: The loop continues until the key is found or all possibilities are exhausted.
		 */
		low = newIndex;
		high = myTable.size;
		for(j = 0; j < 2; j++) {
			for (i = low; i < high; i++) { 
				if (myTable.list[i].key == keys[index]) {
					values[index] = myTable.list[i].value;
					return;
				}
			}
			//- Block Logic: Set up the ranges for the wrap-around probe.
			low = 0;
			high = newIndex;	
		}
	}
}

/**
 * @brief CUDA kernel to rehash elements from an old table to a new one.
 * @details This kernel is launched during a reshape operation. Each thread is responsible
 * for re-inserting one element from the old table into the new, larger table using
 * the same atomic, two-step linear probing logic as the insertion function.
 * @param oldTable The old hash table to read from.
 * @param newTable The new hash table to write to.
 * @param size The size of the new hash table.
 */
__global__ void reshapeKerFunc(hashtable oldTable, hashtable newTable, int size) {
	int newIndex, i, j;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int low, high;

	//- Pre-condition: Ensure the thread index is within the bounds of the table being reshaped.
	if (index >= oldTable.size)
		return;

	//- Block Logic: Re-hash the old key into the new table's coordinate space.
	newIndex = myhashKernel(oldTable.list[index].key, size);
	
	/**
	 * Block Logic: Perform a two-step linear probe to insert the old element into the new table.
	 * Invariant: The loop continues until an empty slot is atomically claimed.
	 */
	low = newIndex;
	high = size;
	for(j = 0; j < 2; j++) {
		for (i = low; i < high; i++) {
			if (atomicCAS(&newTable.list[i].key, 0,
				oldTable.list[index].key) == 0) {
				newTable.list[i].value = oldTable.list[index].value;
				return;
			}
		}
		//- Block Logic: Set up the ranges for the wrap-around probe.
		low = 0;
		high = newIndex;
	}
}


/**
 * @brief Constructs a GpuHashTable object.
 * @param size The initial capacity of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	myTable.size = size;
	myTable.slotsTaken = 0;

	//- Functional Utility: Allocate memory on the GPU for the hash table's node list.
	DIE(cudaMalloc(&myTable.list, size * sizeof(node)) != cudaSuccess, "BOO COULDNT ALLOC MYTABLE LIST INIT
"); 
	//- Functional Utility: Initialize all slots to zero, marking them as empty.
	cudaMemset(myTable.list, 0, size * sizeof(node));
}


/**
 * @brief Destroys the GpuHashTable object, freeing its GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(myTable.list);
}


/**
 * @brief Resizes the hash table to a new, larger capacity.
 * @param numBucketsReshape The desired new number of buckets.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hashtable newTable;
	int noBlocks;

	newTable.size = numBucketsReshape;
	newTable.slotsTaken = myTable.slotsTaken;

	//- Functional Utility: Allocate memory for the new table.
	DIE(cudaMalloc(&newTable.list, numBucketsReshape * sizeof(node)) != cudaSuccess, "BOO COULDNT ALLOC MYTABLE LIST INIT
"); 	
	cudaMemset(newTable.list, 0, numBucketsReshape * sizeof(node));

	//- Block Logic: Calculate the number of blocks required to launch the reshape kernel.
	noBlocks = (myTable.size % BLOCK_SIZE != 0) ?
		(myTable.size / BLOCK_SIZE + 1) : (myTable.size / BLOCK_SIZE);
	//- Kernel Launch: Start the kernel to re-insert all elements into the new table.
	reshapeKerFunc<<<noBlocks, BLOCK_SIZE>>>(myTable, newTable, numBucketsReshape);
	
	//- Functional Utility: Wait for the reshape to complete.
	cudaDeviceSynchronize();

	//- Functional Utility: Free the old table's memory and update the table pointer.
	cudaFree(myTable.list);
	myTable = newTable;
}


/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys Host pointer to an array of keys.
 * @param values Host pointer to an array of values.
 * @param numKeys The number of key-value pairs to insert.
 * @return Returns true on success. The `DIE` macro handles failures.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *device_keys, *device_values;
	int noBlocks;
	myTable.slotsTaken += numKeys;

	size_t memSize = numKeys * sizeof(int);
	//- Functional Utility: Allocate GPU memory for the keys and values.
	DIE(cudaMalloc(&device_keys, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE KEYS
");
	DIE(cudaMalloc(&device_values, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE VALUES
");

	//- Pre-condition: Check if the load factor exceeds the threshold (0.95).
	if ((float(myTable.slotsTaken) / myTable.size) >= 0.95)
		//- Block Logic: If so, reshape the table to a larger size to maintain performance.
		reshape(int((myTable.slotsTaken) / 0.8));

	//- Functional Utility: Copy keys and values from host to device.
	cudaMemcpy(device_keys, keys, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, memSize, cudaMemcpyHostToDevice);

	//- Block Logic: Calculate the number of blocks needed for the insertion kernel.
	noBlocks = (numKeys % BLOCK_SIZE != 0) ?
		(numKeys / BLOCK_SIZE + 1) : (numKeys / BLOCK_SIZE);
	//- Kernel Launch: Launch the kernel to insert the batch.
	insert_kerFunc<<<noBlocks, BLOCK_SIZE>>>(device_keys, device_values, numKeys, myTable);

	cudaDeviceSynchronize();

	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}


/**
 * @brief Retrieves values for a batch of keys.
 * @param keys Host pointer to an array of keys to retrieve.
 * @param numKeys The number of keys in the batch.
 * @return A GPU pointer (from `cudaMallocManaged`) to an array containing the retrieved values.
 */
 int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_values;
    int *device_keys;
    int noBlocks;

	//- Functional Utility: Allocate managed memory for the results, accessible by both host and device.
	DIE(cudaMallocManaged(&device_values, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE VALUES
");
	//- Functional Utility: Allocate device memory for the input keys.
	DIE(cudaMalloc(&device_keys, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE KEYS
");

	//- Functional Utility: Copy keys from host to device.
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	//- Block Logic: Calculate the number of blocks for the retrieval kernel.
	noBlocks = (numKeys % BLOCK_SIZE != 0) ?
		(numKeys / BLOCK_SIZE + 1) : (numKeys / BLOCK_SIZE);

	//- Kernel Launch: Launch the kernel to perform the batch lookup.
	get_kerFunc<<<noBlocks, BLOCK_SIZE>>> (myTable, device_keys, device_values, numKeys);
	
	cudaDeviceSynchronize();
	
	cudaFree(device_keys);
	return device_values;
}


/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor, calculated as (slots taken) / (total size).
 */
 float GpuHashTable::loadFactor() {
	return (myTable.size > 0) ? float(myTable.slotsTaken) / myTable.size : 0;
}


//- Architectural Intent: These macros provide a simplified C-style interface for the hash table operations.
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

//- Architectural Intent: This directly includes a test file, which is unconventional.
#include "test_map.cpp"

#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

/// Defines the value representing an empty slot in the hash table.
#define	KEY_INVALID		0

/**
 * @brief A macro for basic error handling.
 * @details If the assertion is true, it prints a formatted error message
 * to stderr and exits the program with an error code.
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
	
/// A large list of prime numbers, used for hashing to minimize collisions.
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
/// Prime number used as a multiplier in the hash function.
#define PRIMENO 4148279llu
/// A larger prime number used as a modulus in the hash function.
#define PRIMENO2 53944293929llu
/// Defines the number of threads in a CUDA block.
#define BLOCK_SIZE 1024


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
 * @struct node
 * @brief Represents a single key-value pair entry in the hash table.
 */
struct node {
	int key;
	int value;
};

/**
 * @struct hashtable
 * @brief Represents the core data structure for the GPU hash table.
 */
struct hashtable {
	/// The total capacity of the hash table.
	int size;
	/// The number of slots currently occupied.
	int slotsTaken;
	/// A GPU pointer to the array of nodes (key-value pairs).
	node *list;
};

/**
 * @class GpuHashTable
 * @brief The host-side API for interacting with the GPU hash table.
 */
class GpuHashTable
{	
	hashtable myTable;
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
