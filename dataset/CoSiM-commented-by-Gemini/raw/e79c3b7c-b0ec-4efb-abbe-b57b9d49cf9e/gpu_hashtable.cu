/**
 * @file gpu_hashtable.cu
 * @brief Implements a lock-free, linear-probing hash table on the GPU using CUDA.
 *
 * This file contains the host-side C++ class `GpuHashTable` to manage the hash table
 * and the device-side CUDA kernels (`thread_insert`, `thread_get`) that perform
 * the actual hash table operations in parallel. The implementation uses atomic
 * operations to handle concurrent insertions and lookups without locks.
 *
 * Note: The file structure appears to be an aggregation of multiple files,
 * including implementation, declarations, and test code.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include <string>

#include "gpu_hashtable.hpp"

// Forward declaration for the device-side hash function.
__device__ int hash1(int data, int limit);

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 *
 * Each thread is responsible for inserting one key-value pair. Linear probing is
 * used for collision resolution. `atomicCAS` is used to ensure thread-safe
 * insertion and updates.
 *
 * @param keys      (Device Pointer) An array of keys to insert.
 * @param values    (Device Pointer) An array of values corresponding to the keys.
 * @param numKeys   The number of key-value pairs in the batch.
 * @param hmap      The hash map data structure (passed by value).
 */
__global__ void thread_insert(int *keys, int *values, int numKeys, Hmap hmap) {
  	// Standard CUDA pattern to calculate a unique global thread ID.
  	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  	if(i >= numKeys)
  		return;

    // A key of 0 is treated as invalid/empty and is not inserted.
  	if( keys[i] == 0)
  		return;

  	int hashed_idx = hash1(keys[i], hmap.size);

    // Linear probing loop to find a slot.
  	while(1) {
		// Attempt to claim an empty slot. `atomicCAS` (Compare-And-Swap) ensures
        // that only one thread can claim a slot if multiple threads hash to it.
		if( atomicCAS(&hmap.keys[hashed_idx], 0, keys[i]) == 0 ){
			// If we successfully claimed the slot, write the value.
			atomicExch(&hmap.values[hashed_idx], values[i]);
			break;
		}

		// If the slot is already occupied by the same key, this is an update.
		if( atomicCAS(&hmap.keys[hashed_idx], keys[i], keys[i]) == keys[i]){
			// Atomically update the value for the existing key.
			atomicExch(&hmap.values[hashed_idx], values[i]);
			break;
		}

    	// If the slot is occupied by a different key, move to the next slot.
    	hashed_idx++;
     	hashed_idx %= hmap.size; // Wrap around if we reach the end of the table.
    }
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 *
 * Each thread is responsible for finding one key and writing its value to the
 * results array. Linear probing is used to find the key.
 *
 * @param keys      (Device Pointer) An array of keys to look up.
 * @param numKeys   The number of keys in the batch.
 * @param hmap      The hash map data structure (passed by value).
 * @param results   (Device Pointer) The output array to store the found values.
 */
__global__ void thread_get(int *keys, int numKeys, Hmap hmap, int *results) {
  	// Calculate unique global thread ID.
  	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  	if(i >= numKeys)
  		return;

  	int hashed_idx = hash1(keys[i], hmap.size);

  	// Linear probing loop to find the key.
  	while(1) {
    	// Check if the key at the current slot matches the key we're looking for.
        // `atomicCAS` is used here in a "read-only" way to ensure we see a consistent value.
    	if(atomicCAS(&hmap.keys[hashed_idx], keys[i], keys[i]) == keys[i]){
    		// Key found, atomically read the value and write it to the results array.
    		atomicExch(&results[i], hmap.values[hashed_idx]);
    		break;
    	}

    	// Key not found, move to the next slot.
    	hashed_idx++;
     	hashed_idx %= hmap.size; // Wrap around.
    }
}

/**
 * @brief Host-side constructor for GpuHashTable.
 * Allocates memory on the GPU for the keys and values arrays.
 * @param size The initial capacity of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	hm.current_no_of_pairs = 0;
	hm.size = size;
	cudaMalloc((void **) &hm.keys, size * sizeof(int));
	cudaMalloc((void **) &hm.values, size * sizeof(int));

	// Initialize all keys and values to 0. 0 is considered an empty slot.
	cudaMemset(hm.keys, 0, size * sizeof(int));
	cudaMemset(hm.values, 0, size * sizeof(int));
}

/**
 * @brief Host-side destructor for GpuHashTable.
 * Frees the GPU memory allocated for the hash table.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hm.keys);
	cudaFree(hm.values);
}

/**
 * @brief Resizes the hash table to a new capacity.
 * This involves creating a new, larger table on the GPU and re-inserting
 * all elements from the old table.
 * @param numBucketsReshape The new capacity.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	Hmap new_hm;
	new_hm.size = numBucketsReshape;

	// Allocate new, larger arrays on the GPU.
	cudaMalloc((void **) &new_hm.keys, numBucketsReshape * sizeof(int));
	cudaMalloc((void **) &new_hm.values, numBucketsReshape * sizeof(int));
	cudaMemset(new_hm.keys, 0, numBucketsReshape * sizeof(int));
	cudaMemset(new_hm.values, 0, numBucketsReshape * sizeof(int));

	// Configure kernel launch parameters to re-insert old elements.
	unsigned int no_of_threads_in_block = 256;
	unsigned int no_of_blocks = hm.size / no_of_threads_in_block;
	if(hm.size % no_of_threads_in_block != 0)
		no_of_blocks ++;

	// Launch the insert kernel to re-hash all old elements into the new table.
	thread_insert>>(hm.keys, hm.values, hm.size, new_hm);
	cudaDeviceSynchronize(); // Wait for the re-hashing to complete.

	new_hm.current_no_of_pairs += hm.current_no_of_pairs;

	// Free the old GPU memory.
	cudaFree(hm.keys);
	cudaFree(hm.values);

	// Replace the old hash map with the new one.
	hm = new_hm;
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys      (Host Pointer) An array of keys to insert.
 * @param values    (Host Pointer) An array of values to insert.
 * @param numKeys   The number of pairs to insert.
 * @return True on success.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys, *device_values;

	// Configure kernel launch parameters.
	unsigned int no_of_threads_in_block = 256;
	unsigned int size = numKeys * sizeof(int);
	unsigned int no_of_blocks = numKeys / no_of_threads_in_block;
	if(numKeys % no_of_threads_in_block != 0)
		no_of_blocks++;

    // Check load factor and reshape if it exceeds a threshold (0.9).
	if (float(hm.current_no_of_pairs + numKeys) / hm.size >= 0.9)
		reshape(int((hm.current_no_of_pairs + numKeys) / 0.8));

    // Allocate temporary device memory for the input batch.
	cudaMalloc(&device_keys, size);
	cudaMalloc(&device_values, size);

	// Copy the batch from host to device.
	cudaMemcpy(device_keys, keys, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, size, cudaMemcpyHostToDevice);

    // Launch the insertion kernel.
	thread_insert>>(device_keys, device_values, numKeys, hm);
	cudaDeviceSynchronize(); // Wait for insertions to complete.

	hm.current_no_of_pairs += numKeys;

    // Clean up temporary device memory.
	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}

/**
 * @brief Retrieves values for a batch of keys.
 * @param keys      (Host Pointer) An array of keys to look up.
 * @param numKeys   The number of keys to look up.
 * @return A pointer to an array containing the results. This memory is
 *         managed by CUDA (`cudaMallocManaged`) and must be freed by the caller
 *         using `cudaFree`.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_keys;

	// Configure kernel launch parameters.
	unsigned int no_of_threads_in_block = 256;
	unsigned int size = numKeys * sizeof(int);
	unsigned int no_of_blocks = numKeys / no_of_threads_in_block;
	if(numKeys % no_of_threads_in_block != 0)
		no_of_blocks++;

    // Allocate and copy keys to device.
	cudaMalloc(&device_keys, size);
	cudaMemcpy(device_keys, keys, size, cudaMemcpyHostToDevice);

	// Allocate managed memory for results, accessible from both host and device.
	int *results;
	cudaMallocManaged(&results, size);

    // Launch the get kernel.
	thread_get>>(device_keys, numKeys, hm, results);
	cudaDeviceSynchronize(); // Wait for lookups to complete.

	cudaFree(device_keys);
	return results;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor (number of pairs / capacity).
 */
float GpuHashTable::loadFactor() {
	if(hm.size == 0)
		return 0;
	return float(hm.current_no_of_pairs)/hm.size;
}

// The following section appears to be a mix of preprocessor macros for a simplified
// API, inclusion of a test file, and redundant/alternative definitions.

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"

// The block below seems to be a separate or alternative set of definitions.
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

// A list of prime numbers, likely for use in hashing functions.
__device__ const size_t primeList[] =
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

/**
 * @brief Simple hashing function executed on the device.
 */
__device__ int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}

/**
 * @brief Redundant-looking definition for the hash map structure.
 * This should ideally be in the gpu_hashtable.hpp header file.
 */
typedef struct hashmap {
	int *keys;
	int *values;
	unsigned int current_no_of_pairs;
	unsigned int size;
} Hmap;

/**
 * @brief Redundant-looking class declaration for GpuHashTable.
 * The implementation is provided earlier in the file. This should be in a header.
 */
class GpuHashTable
{
	Hmap hm;
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
