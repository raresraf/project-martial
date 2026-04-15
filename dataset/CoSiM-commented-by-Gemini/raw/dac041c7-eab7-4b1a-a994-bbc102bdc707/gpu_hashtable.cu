/**
 * @file gpu_hashtable.cu
 * @brief This file contains a CUDA implementation of a GPU-based hash table.
 * @raw/dac041c7-eab7-4b1a-a994-bbc102bdc707/gpu_hashtable.cu
 *
 * This implementation uses open addressing with linear probing for collision resolution.
 * A key feature of this implementation is its iterative approach to handling insertions
 * and retrievals. Instead of resolving all collisions within a single kernel launch,
 * the kernels (`insertKeysandValues`, `getbatch`) are launched in a loop. In each iteration,
 * threads attempt to process their assigned keys. If a collision occurs, the thread
_modifies_
 * its target hash code (by incrementing it) and retries in the next iteration of the host loop.
 * This continues until all keys in the batch are processed.
 *
 * Algorithm: Open Addressing with Linear Probing (Iterative Kernel Launch)
 * Time Complexity:
 *  - Insertion: Heavily dependent on load factor and collisions. The iterative nature can lead to many kernel launches.
 *  - Retrieval: Similar to insertion, performance depends on collisions.
 * Space Complexity: O(N)
 */
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "gpu_hashtable.hpp"


/**
 * @brief Constructor for the GpuHashTable class.
 * @param size The initial capacity of the hash table.
 *
 * Allocates an array of `hT` structs on the GPU's global memory and initializes
 * the memory to zero, marking all slots as empty.
 */
GpuHashTable::GpuHashTable(int size)
{
	cudaError_t error;

	
	error = cudaMalloc(&(GpuHashTable::hashtable), size * sizeof(hT));
	DIE(error != cudaSuccess || GpuHashTable::hashtable == NULL, "cudaMalloc hashtable error");
	error = cudaMemset(GpuHashTable::hashtable, 0, size * sizeof(hT));
	DIE(error != cudaSuccess, "cudaMemset hashtable error");

	
	GpuHashTable::currentTableSize = 0;
	GpuHashTable::tableSize = size;
}



/**
 * @brief Destructor for the GpuHashTable class.
 *
 * Frees the GPU memory allocated for the hash table.
 */
GpuHashTable::~GpuHashTable()
{
	cudaError_t error;

	
	error = cudaFree(GpuHashTable::hashtable);
	DIE(error != cudaSuccess, "cudaFree hashtable error");

	
	GpuHashTable::currentTableSize = 0;
	GpuHashTable::tableSize = 0;
}


/**
 * @brief CUDA kernel to copy existing key-value pairs from the hash table into flat arrays.
 * @param hashtable Pointer to the source hash table.
 * @param tableSize The capacity of the source hash table.
 * @param device_keys Destination array for keys on the device.
 * @param device_values Destination array for values on the device.
 * @param counter An atomic counter to get a unique index for each copied element.
 *
 * This kernel is the first step in the `reshape` process. Each thread checks one slot
 * in the hash table. If the slot is occupied, it atomically increments a counter to get a
 * unique index and copies the key-value pair to the corresponding position in the
 * destination arrays.
 */
__global__ void copyForReshape(hT *hashtable, int tableSize,
								int *device_keys, int *device_values,
								int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if(idx < tableSize) {
		// Pre-condition: Check if the current slot is occupied.
		if (hashtable[idx].key != 0) {
			
			// Atomically get a unique index in the output arrays.
			int index = atomicAdd(counter, 1);

			device_keys[index] = hashtable[idx].key;
			device_values[index] = hashtable[idx].value;
		}
	}
}



/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The desired number of elements to accommodate, used to calculate the new size.
 *
 * This function orchestrates a complex reshape operation:
 * 1. Allocates a new, larger hash table.
 * 2. If the current table is not empty, it launches the `copyForReshape` kernel to
 *    extract all key-value pairs into temporary flat arrays on the device.
 * 3. Copies these key-value pairs back to the host.
 * 4. Destroys the old hash table and replaces it with the new one.
 * 5. Calls `insertBatch` to re-insert all the elements from the host arrays into the new table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t error;
	hT *newHashTable = NULL;
	// Invariant: The new size is set to be 20% larger than the number of buckets requested
	// to maintain a low load factor.
	int new_size = 1.2f * numBucketsReshape;

	
	error = cudaMalloc(&newHashTable, new_size * sizeof(hT));
	DIE(error != cudaSuccess || newHashTable == NULL, "cudaMalloc new hashtable error");
	error = cudaMemset(newHashTable, 0, new_size * sizeof(hT));
	DIE(error != cudaSuccess, "cudaMemset new hashtable error");


	
	if(GpuHashTable::currentTableSize != 0) {
		const size_t block_size = 1024;
		size_t blocks_no = GpuHashTable::tableSize / block_size;
 
		if (GpuHashTable::tableSize % block_size) 
			++blocks_no;

		int *device_keys = NULL;
		int *counter = NULL;
		int *device_values = NULL;


		error = cudaMalloc(&device_keys, GpuHashTable::currentTableSize * sizeof(int));
		DIE(error != cudaSuccess || device_keys == NULL, "cudaMalloc device_keys error");

		error = cudaMalloc(&device_values, GpuHashTable::currentTableSize * sizeof(int));
		DIE(error != cudaSuccess || device_values == NULL, "cudaMalloc device_values error");

		
		error = cudaMalloc(&counter, sizeof(int));
		DIE(error != cudaSuccess || counter == NULL, "cudaMalloc counter error");
		error = cudaMemset(counter, 0, sizeof(int));
		DIE(error != cudaSuccess, "cudaMemset counter error");


		int *host_keys = (int *)malloc(GpuHashTable::currentTableSize * sizeof(int));
		DIE(host_keys == NULL, "malloc host_keys error");

		int *host_values = (int *)malloc(GpuHashTable::currentTableSize * sizeof(int));
		DIE(host_values == NULL, "malloc host_values error");

		
		
		
		// Launch kernel to copy all existing elements to flat arrays.
		copyForReshape>>(GpuHashTable::hashtable,
												GpuHashTable::tableSize,
												device_keys, device_values,
												counter);
		error = cudaDeviceSynchronize();
		DIE(error != cudaSuccess, "cudaDeviceSynchronize error");


		
		// Copy the flattened data back to the host.
		error = cudaMemcpy(host_keys, device_keys,
							GpuHashTable::currentTableSize * sizeof(int),
							cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_keys error");

		error = cudaMemcpy(host_values, device_values,
							GpuHashTable::currentTableSize * sizeof(int),
							cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_values error");


		
		// Destroy the old table and replace it with the new one.
		GpuHashTable::~GpuHashTable();

		
		GpuHashTable::tableSize = new_size;
		GpuHashTable::hashtable = newHashTable;
		GpuHashTable::currentTableSize = 0;
		

		int numKeys = 0;
		error = cudaMemcpy(&numKeys, counter, sizeof(int), cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy numKeys error");
		
		
		// Re-insert all the elements into the new, larger table.
		insertBatch(host_keys, host_values, numKeys);


		error = cudaFree(device_keys);
		DIE(error != cudaSuccess, "cudaFree device_keys error");

		error = cudaFree(device_values);
		DIE(error != cudaSuccess, "cudaFree device_values error");

		error = cudaFree(counter);
		DIE(error != cudaSuccess, "cudaFree counter error");

		free(host_keys);
		free(host_values);

		return;

	}

	
	GpuHashTable::~GpuHashTable();

	
	GpuHashTable::tableSize = new_size;

	GpuHashTable::hashtable = newHashTable;
}





/**
 * @brief CUDA kernel to pre-compute hash codes for a batch of keys.
 * @param keys The input keys on the device.
 * @param hashcodes The output array for the computed hash codes.
 * @param numkeys The number of keys in the batch.
 * @param tablesize The current capacity of the hash table.
 *
 * Each thread computes the hash for one key. This allows the initial hash
 * positions to be calculated in parallel before the insertion or retrieval process begins.
 */
__global__ void getHashCode(int *keys, int *hashcodes, int numkeys, int tablesize)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numkeys)
		hashcodes[idx] = (((long)keys[idx]) * 1402982055436147llu)
								% 452517535812813007llu % tablesize;
}







/**
 * @brief CUDA kernel that performs one iteration of the batch insertion process.
 * @param hashcodes The current hash codes (probe positions) for each key.
 * @param keys The keys to be inserted.
 * @param values The corresponding values.
 * @param numKeys The total number of keys in the batch.
 * @param hashtable The hash table itself.
 * @param currentTableSize A device pointer to the count of elements in the table.
 * @param tablesize The capacity of the hash table.
 * @param counter A counter for successfully inserted keys in this iteration.
 *
 * Each thread attempts to insert one key at its current `hashcodes` position.
 * - If the slot is empty, it uses `atomicCAS` to claim it and insert the value.
 * - If the slot contains the same key, it updates the value.
 * - If the slot is occupied by a different key (a collision), it increments its
 *   `hashcodes` value to probe the next slot in the next iteration of the host loop.
 * Keys that are successfully inserted are marked with 0 to be ignored in subsequent iterations.
 */
__global__ void insertKeysandValues(int *hashcodes, int *keys, int *values,
									int numKeys, hT *hashtable,


									int *currentTableSize, int tablesize,
									int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numKeys) {
		// Pre-condition: Ignore keys that have already been inserted (marked as 0).
		if (keys[idx] == 0)
			return;

		int key = atomicCAS(&hashtable[hashcodes[idx]].key, 0, keys[idx]);
		
		// Invariant: The slot was empty, so we successfully claimed it.
		if (key == 0) {
			hashtable[hashcodes[idx]].value = values[idx];
			
			// Mark the key as processed.
			keys[idx] = 0;
			atomicAdd(currentTableSize, 1);
			atomicAdd(counter, 1);
		// Invariant: The slot already contained our key, so we update the value.
		} else if (key == keys[idx]) {
			
			hashtable[hashcodes[idx]].value = values[idx];
			atomicAdd(counter, 1);
			
			// Mark the key as processed.
			keys[idx] = 0;
		} else {
			
			// Block Logic: A collision occurred. Increment the hash code to probe the next
			// slot in the subsequent kernel launch. This implements linear probing across
			// multiple kernel launches.
			hashcodes[idx] += 1;
			hashcodes[idx] %= tablesize;
		}
	}
}






/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys Pointer to the host array of keys.
 * @param values Pointer to the host array of values.
 * @param numKeys The number of key-value pairs to insert.
 * @return `true` if the insertion process was successful.
 *
 * This function orchestrates the batch insertion. It first ensures there is enough
 * capacity, reshaping if necessary. It then pre-computes the hash codes and enters
 * a `while(1)` loop, repeatedly launching the `insertKeysandValues` kernel. Each launch
 * attempts to insert the remaining keys. The loop terminates when a counter indicates
 * that all keys in the batch have been successfully inserted.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys = NULL;
	int *device_values = NULL;
	cudaError_t error;


	
	
	// Pre-condition: Check if the table needs to be resized to accommodate the new keys.
	if ((GpuHashTable::currentTableSize + numKeys) > GpuHashTable::tableSize)
		reshape((GpuHashTable::currentTableSize + numKeys));


	
	error = cudaMalloc(&device_keys, numKeys * sizeof(int));
	DIE(error != cudaSuccess || device_keys == NULL, "cudaMalloc device_keys error");

	error = cudaMalloc(&device_values, numKeys * sizeof(int));


	DIE(error != cudaSuccess || device_values == NULL, "cudaMalloc device_values error");


	
	error = cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(error != cudaSuccess, "cudaMemcpy keys error");

	error = cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(error != cudaSuccess, "cudaMemcpy values error");


	
	// Pre-compute initial hash codes for all keys in parallel.
	int *hashcodes = NULL;
	error = cudaMalloc(&hashcodes, numKeys * sizeof(int));
	DIE(error != cudaSuccess || hashcodes == NULL, "cudaMalloc hashcodes error");

	const size_t block_size = 1024;
	size_t blocks_no = numKeys / block_size;
 
	if (numKeys % block_size) 
		++blocks_no;

	getHashCode>>(device_keys, hashcodes,
											numKeys, GpuHashTable::tableSize);
	error = cudaDeviceSynchronize();
	DIE(error != cudaSuccess, "cudaDeviceSynchronize error");



	
	
	int *device_current = NULL;
	error = cudaMalloc(&device_current, sizeof(int));
	DIE(error != cudaSuccess, "cudaMalloc device_current error");
	error = cudaMemset(device_current, 0, sizeof(int));
	DIE(error != cudaSuccess, "cudaMemset device_current error");



	
	
	int *device_counter = NULL;
	error = cudaMalloc(&device_counter, sizeof(int));
	DIE(error != cudaSuccess || device_counter == NULL, "cudaMalloc device_counter error");
	error = cudaMemset(device_counter, 0, sizeof(int));
	DIE(error != cudaSuccess, "cudaMemset device_counter error");


	int host_counter = 0;
	int old_counter = 0;
	// Block Logic: This loop repeatedly launches the insertion kernel until all keys are processed.
	// This is an iterative approach to handling collisions.
	while(1) {
		
		insertKeysandValues>>(hashcodes, device_keys,
													device_values, numKeys,
													GpuHashTable::hashtable,
													device_current,
													GpuHashTable::tableSize,
													device_counter);
		error = cudaDeviceSynchronize();
		DIE(error != cudaSuccess, "cudaDeviceSynchronize error");

		old_counter = host_counter;
		error = cudaMemcpy(&host_counter, device_counter, sizeof(int), cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_counter error");

		
		GpuHashTable::currentTableSize += host_counter - old_counter;

		// Invariant: The loop terminates when the number of successfully inserted keys
		// equals the total number of keys in the batch.
		if(host_counter == numKeys)
			break;
	}
	

	
	error = cudaFree(device_keys);
	DIE(error != cudaSuccess, "cudaFree device_keys error");

	error = cudaFree(device_values);
	DIE(error != cudaSuccess, "cudaFree device_values error");

	error = cudaFree(hashcodes);
	DIE(error != cudaSuccess, "cudaFree hashcodes error");

	error = cudaFree(device_current);
	DIE(error != cudaSuccess, "cudaFree device_current error");

	error = cudaFree(device_counter);
	DIE(error != cudaSuccess, "cudaFree device_counter error");

	return true;
}




/**
 * @brief CUDA kernel that performs one iteration of the batch retrieval process.
 * @param values The output array for retrieved values.
 * @param hashcodes The current hash codes (probe positions) for each key.
 * @param keys The keys to be retrieved.
 * @param numKeys The total number of keys in the batch.
 * @param hashtable The hash table itself.
 * @param tablesize The capacity of the hash table.
 * @param counter A counter for successfully retrieved keys in this iteration.
 *
 * Each thread attempts to retrieve one key. If the key is found at the current
 * probe position, the value is written to the output array, and the key is marked
 * as processed (set to 0). If not found, the hash code is incremented for the next
 * iteration of the host loop.
 */
__global__ void getbatch(int *values, int *hashcodes, int *keys, int numKeys,
						hT *hashtable, int tablesize, int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numKeys) {
		// Pre-condition: Ignore keys that have already been found.
		if (keys[idx] == 0)
			return;

		if(hashtable[hashcodes[idx]].key == keys[idx]) {
			
			values[idx] = hashtable[hashcodes[idx]].value;
			atomicAdd(counter, 1);
			keys[idx] = 0;
		} else {
			
			// Block Logic: Key not found at this position. Increment hash code
			// to probe the next slot in the next kernel launch.
			hashcodes[idx] += 1;
			hashcodes[idx] %= tablesize;
		}
	}
}





/**
 * @brief Retrieves the values for a batch of keys.
 * @param keys Pointer to the host array of keys to search for.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to a host array containing the retrieved values.
 *
 * This function orchestrates the batch retrieval. Similar to `insertBatch`, it
 * pre-computes hash codes and then enters a `while(1)` loop, repeatedly launching
 * the `getbatch` kernel until all keys have been found or the probing is exhausted.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *ret_values = (int *)malloc(numKeys * sizeof(int));
	cudaError_t error;

	
	int *device_ret_values = NULL;
	error = cudaMalloc(&device_ret_values, numKeys * sizeof(int));
	DIE(error != cudaSuccess || device_ret_values == NULL, "cudaMalloc device_ret_values error");

	
	int *device_keys = NULL;
	error = cudaMalloc(&device_keys, numKeys * sizeof(int));
	DIE(error != cudaSuccess || device_keys == NULL, "cudaMalloc device_keys error");
	error = cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(error != cudaSuccess, "cudaMemcpy keys error");


	
	// Pre-compute initial hash codes.
	int *hashcodes = NULL;

	error = cudaMalloc(&hashcodes, numKeys * sizeof(int));
	DIE(error != cudaSuccess || hashcodes == NULL, "cudaMalloc hashcodes error");

	const size_t block_size = 1024;
	size_t blocks_no = numKeys / block_size;
 
	if (numKeys % block_size) 
		++blocks_no;

	getHashCode>>(device_keys, hashcodes, numKeys, GpuHashTable::tableSize);
	error = cudaDeviceSynchronize();
	DIE(error != cudaSuccess, "cudaDeviceSynchronize error");



	int *device_counter = NULL;
	error = cudaMalloc(&device_counter, sizeof(int));
	DIE(error != cudaSuccess || device_counter == NULL, "cudaMalloc device_counter error");
	error = cudaMemset(device_counter, 0, sizeof(int));
	DIE(error != cudaSuccess, "cudaMemset device_counter error");

	// Block Logic: Iteratively launch the `getbatch` kernel until all keys are found.
	while(1) {
		getbatch>>(device_ret_values, hashcodes,
											device_keys, numKeys,
											GpuHashTable::hashtable,
											GpuHashTable::tableSize, device_counter);
		error = cudaDeviceSynchronize();
		DIE(error != cudaSuccess, "cudaDeviceSynchronize error");

		int host_counter = 0;
		error = cudaMemcpy(&host_counter, device_counter, sizeof(int), cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_counter error");

		// Invariant: Loop terminates when all keys in the batch have been found.
		if(host_counter == numKeys)
			break;
	}


	


	// Copy the retrieved values from device to host.
	error = cudaMemcpy(ret_values, device_ret_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	DIE(error != cudaSuccess, "cudaMemcpy ret_values error");


	
	error = cudaFree(device_keys);
	DIE(error != cudaSuccess, "cudaFree device_keys error");

	error = cudaFree(device_ret_values);
	DIE(error != cudaSuccess, "cudaFree device_ret_values error");

	error = cudaFree(hashcodes);
	DIE(error != cudaSuccess, "cudaFree hashcodes error");

	error = cudaFree(device_counter);
	DIE(error != cudaSuccess, "cudaFree device_counter error");

	return ret_values;
}



/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float.
 */
float GpuHashTable::loadFactor() {
	return ((float)GpuHashTable::currentTableSize) / ((float)GpuHashTable::tableSize);
}


/**
 * @brief These macros provide a simplified API for the hash table operations,
 * likely intended for use in the included test file (test_map.cpp).
 */
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
/**
 * @brief This block contains a CPU version of a hash table implementation.
 * It is included for testing, comparison, or as a fallback implementation.
 * It uses a large list of prime numbers, likely for resizing strategies.
 */
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
	
// A list of prime numbers, likely used for choosing hash table sizes to minimize collisions.
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




// Alternative hash functions, not used in the current implementation.
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

/**
 * @brief Defines the structure for a hash table entry.
 */
typedef struct hashtableCell{
	int key;
	int value;
} hT;




class GpuHashTable
{
	public:
		hT *hashtable;
		int tableSize;
		int currentTableSize;
		

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
