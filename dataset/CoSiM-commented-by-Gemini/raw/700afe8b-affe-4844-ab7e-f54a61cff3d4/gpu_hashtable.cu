/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 * @details This file provides a hash table data structure designed for high-performance execution on GPUs.
 * It uses open addressing with a linear probing collision resolution strategy. The implementation supports
 * dynamic resizing (rehashing) when the load factor exceeds a certain threshold. A key feature is the
 * use of separate host and device counters for the number of elements, requiring synchronization
 * after mutation operations.
 *
 * Algorithm: Open addressing with linear probing.
 * - Hashing: Multiplicative hashing with large prime constants.
 * - Collision Resolution: Linear probing with wrap-around. The probe sequence continues until an
 *   empty slot is found or the entire table has been traversed.
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


/**
 * @brief Constructs a GpuHashTable.
 * @param size The initial capacity (number of slots) of the hash table.
 * @note This constructor allocates memory for the hash table on the device, as well as
 *       both a host-side and a device-side counter for the number of used slots.
 *       The host-side pointer (`size_used_host`) is leaked in the destructor.
 */
GpuHashTable::GpuHashTable(int size)
{
	
	capacity = size;

	
	cudaMalloc((void **)&size_used_device, sizeof(int));
	DIE(size_used_device == 0, "memory allocation err");
	cudaMemset(size_used_device, 0 , sizeof(int));

	
	size_used_host = (int *)malloc(sizeof(int));
	DIE(size_used_host == 0, "memory allocation err");
	memset(size_used_host, 0 , sizeof(int));

	
	int num_bytes = size * sizeof(HashTableNode);
	cudaMalloc((void **)&hashTable, num_bytes);
	DIE(hashTable == 0, "memory allocation err");
	
	cudaMemset(hashTable, 0, num_bytes);
}



/**
 * @brief Destroys the GpuHashTable.
 * @note This destructor frees the main hash table memory but leaks the
 *       `size_used_device` and `size_used_host` allocations from the constructor.
 */
GpuHashTable::~GpuHashTable()
{
	
	cudaFree(hashTable);
}





/**
 * @brief CUDA kernel to re-insert all keys from an old hash table into a new, larger one.
 * @details This kernel is the core of the reshape operation. Each thread is responsible for one
 * slot in the old table. If the slot contains a valid key, the thread re-hashes it and
 * inserts it into the new table using atomic linear probing.
 * @param newHT Pointer to the new hash table on the device.
 * @param capacity The capacity of the new hash table.
 * @param oldHT Pointer to the old hash table on the device.
 * @param old_size The capacity of the old hash table.
 */
 __global__ void reinsertKeys(HashTableNode *newHT, int capacity, HashTableNode *oldHT, int old_size)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	
	//- Pre-condition: Ensure the thread index is within the bounds of the old table.
	if (idx >= old_size)
		return;

	int key = oldHT[idx].key;
	int value = oldHT[idx].value;

	
	//- Pre-condition: Ignore empty or invalid slots from the old table.
	if (key <= 0 || value <= 0)
		return;

	
	//- Block Logic: Calculate the initial position for the key in the new table.
	int pos = getHash(key, capacity);

	int count = 0;
	/**
	 * Block Logic: Perform linear probing to find an empty slot in the new table.
	 * Invariant: The loop continues until an empty slot is found or the entire table has been probed.
	 */
	while (count < capacity) {

		
		//- Inline: Atomically attempt to claim an empty slot (key 0).
		int old = atomicCAS(&(newHT[pos].key), 0, key);

		
		if (old == 0) {
			newHT[pos].value = value;
			return;
		}

		
		//- Inline: Move to the next slot with wrap-around.
		pos = (pos + 1) % capacity;
		count++;
	}
}


/**
 * @brief Resizes the hash table to a new, larger capacity and rehashes all elements.
 * @param numBucketsReshape The new number of slots for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape)
{

	
	int old_size = capacity;
	capacity = numBucketsReshape;

	
	HashTableNode *oldHT = hashTable;
	int num_bytes = capacity * sizeof(HashTableNode);

	//- Functional Utility: Allocate a new, larger table on the GPU.
	hashTable = NULL;
	cudaMalloc((void **)&hashTable, num_bytes);
	DIE(hashTable == 0, "memory allocation err");
	cudaMemset(hashTable, 0, num_bytes);

	
	//- Pre-condition: If the table was empty, there's no need to rehash.
	if (*size_used_host == 0) {
		cudaFree(oldHT);
		return;
	}

	
	//- Block Logic: Calculate the grid dimensions for the re-insertion kernel.
	int nrBlocks = old_size / BLOCKSIZE;
	if (old_size % BLOCKSIZE != 0)
		nrBlocks++;

	
	//- Kernel Launch: Start the process of re-inserting all keys into the new table.
	reinsertKeys<<<nrBlocks, BLOCKSIZE>>>(hashTable, capacity, oldHT, old_size);
	
	cudaDeviceSynchronize();

	
	//- Functional Utility: Free the memory of the old table.
	cudaFree(oldHT);
}






/**
 * @brief CUDA kernel for inserting a batch of key-value pairs.
 * @details Each thread handles one key-value pair. It calculates the hash and performs atomic linear
 * probing to find a slot. It handles new insertions, updates to existing keys, and increments
 * a shared counter for new insertions.
 * @param hashTable The device pointer to the hash table.
 * @param capacity The total capacity of the hash table.
 * @param size_used_device A device pointer to an integer tracking the number of used slots.
 * @param keys Device pointer to the array of keys to insert.
 * @param values Device pointer to the array of values to insert.
 * @param numKeys The number of items in the batch.
 */
__global__ void insertKey(HashTableNode *hashTable, int capacity, int *size_used_device, int *keys, int *values, int numKeys)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numKeys)
		return;

	int key = keys[idx];
	int value = values[idx];

	
	//- Pre-condition: This implementation considers keys/values <= 0 as invalid.
	if (key <= 0 || value <= 0)
		return;

	
	int pos = getHash(key, capacity);

	int count = 0;
	/**
	 * Block Logic: Perform linear probing to find or create an entry for the key.
	 * Invariant: The loop continues until the key is inserted/updated or the table is found to be full.
	 */
	while (count < capacity) {

		
		//- Inline: Atomically attempt to claim an empty slot (key 0) with the new key.
		int old = atomicCAS(&(hashTable[pos].key), 0, key);

		
		//- Block Logic: If the slot was empty, we have successfully inserted a new key.
		if (old == 0) {
			hashTable[pos].value = value;
			//- Inline: Atomically increment the global counter for used slots.
			atomicAdd(size_used_device, 1);
			return;
		}

		
		//- Block Logic: If the key was already present, update its value.
		if (old == key) {
			hashTable[pos].value = value;
			return;
		}

		
		//- Inline: If the slot was occupied by another key, move to the next slot.
		pos = (pos + 1) % capacity;
		count++;
	}
}



/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys Host pointer to an array of keys.
 * @param values Host pointer to an array of values.
 * @param numKeys The number of items to insert.
 * @return Returns true on success. Failures are handled by the `DIE` macro.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys)
{
	
	//- Pre-condition: Check if the insertion would exceed the current capacity.
	if (*size_used_host + numKeys >= capacity) {

		
		//- Block Logic: If so, calculate a new size (1.25x the required size) and reshape.
		int newSize = (*size_used_host + numKeys) * 1.25;
		this->reshape(newSize);
	}


	
	//- Block Logic: Calculate the number of blocks for the insertion kernel.
	int nrBlocks = numKeys / BLOCKSIZE;
	if (numKeys % BLOCKSIZE != 0)
		nrBlocks++;

	
	//- Functional Utility: Allocate temporary GPU memory for the batch of keys and values.
	int *k = 0, *v = 0;
	cudaMalloc((void **)&k, numKeys * sizeof(int));
	cudaMalloc((void **)&v, numKeys * sizeof(int));
	DIE(k == 0, "memory allocation err");
	DIE(v == 0, "memory allocation err");

	//- Functional Utility: Copy data from host to device.
	cudaMemcpy(k, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(v, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	
	//- Kernel Launch: Execute the insertion kernel.
	insertKey<<<nrBlocks, BLOCKSIZE>>>(hashTable, capacity, size_used_device, k, v, numKeys);
	
	cudaDeviceSynchronize();
	
	
	//- Functional Utility: Synchronize the host-side counter with the updated device-side counter.
	cudaMemcpy(size_used_host, size_used_device, sizeof(int), cudaMemcpyDeviceToHost);



	cudaFree(k);
	cudaFree(v);

	return true;
}






/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param hashTable Device pointer to the hash table.
 * @param capacity The capacity of the hash table.
 * @param values Device pointer to the output array for storing found values.
 * @param keys Device pointer to the input array of keys to search for.
 * @param numKeys The number of keys to retrieve.
 */
__global__ void getValues(HashTableNode *hashTable, int capacity, int *values, int *keys, int numKeys)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numKeys)
		return;

	int key = keys[idx];

	
	//- Pre-condition: Ignore invalid keys.
	if (key <= 0)
		return;

	
	int pos = getHash(key, capacity);

	int count = 0;
	/**
	 * Block Logic: Perform linear probing to find the key.
	 * Invariant: The loop continues until the key is found or the entire table has been checked.
	 */
	while (count < capacity) {

		
		if (hashTable[pos].key == key) {
			*(values + idx) = hashTable[pos].value;
			return;
		}

		
		pos = (pos + 1) % capacity;

		count++;
	}
}


/**
 * @brief Retrieves values for a batch of keys.
 * @param keys Host pointer to an array of keys.
 * @param numKeys The number of keys to retrieve.
 * @return A host pointer to a newly allocated array containing the values. The caller must free this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys)
{

	int num_bytes = numKeys * sizeof(int);
	int *k = 0, *v = 0;
	//- Functional Utility: Allocate temporary GPU memory for the operation.
	cudaMalloc((void **)&k, num_bytes);
	cudaMalloc((void **)&v, num_bytes);
	DIE(k == 0, "memory allocation err");
	DIE(v == 0, "memory allocation err");

	
	//- Functional Utility: Copy input keys from host to device.
	cudaMemcpy(k, keys, num_bytes, cudaMemcpyHostToDevice);
	
	cudaMemset(v, 0, num_bytes);

	
	//- Block Logic: Calculate kernel launch configuration.
	int nrBlocks = numKeys / BLOCKSIZE;
	if (numKeys % BLOCKSIZE != 0)
		nrBlocks++;

	
	//- Kernel Launch: Execute the retrieval kernel.
	getValues<<<nrBlocks, BLOCKSIZE>>>(hashTable, capacity, v, k, numKeys);
	
	cudaDeviceSynchronize();

	
	//- Functional Utility: Allocate host memory for the results and copy them back from the device.
	int *res;
	res = (int *)malloc(num_bytes);
	cudaMemcpy(res, v, num_bytes, cudaMemcpyDeviceToHost);

	cudaFree(k);
	cudaFree(v);

	return res;
}





/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float (number of used slots / total capacity).
 */
float GpuHashTable::loadFactor()
{
	return (float) (*size_used_host) / capacity;
}







#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0

#define BLOCKSIZE 1024

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


__device__ int getHash(int key, int limit)
{
	return (((long)abs(key) * 1337987929llu) % 718326812383316683llu) % limit;
}


int getHashHost(int key, int limit)
{
	return (((long)abs(key) * 1337987929llu) % 718326812383316683llu) % limit;
}




typedef struct N {
	int key;
	int value;
} HashTableNode;




class GpuHashTable
{
	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		
		float loadFactor();
		void occupancy();
		void print(string info);
	
		~GpuHashTable();


	
	public:
		
		int capacity;
		
		int *size_used_device;
		
		int *size_used_host;
		
		HashTableNode *hashTable = 0;
};

#endif
