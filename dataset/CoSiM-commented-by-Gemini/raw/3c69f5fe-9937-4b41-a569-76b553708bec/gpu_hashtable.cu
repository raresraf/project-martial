


/**

 * @file gpu_hashtable.cu

 * @brief CUDA implementation of a GPU-resident hash table.

 *

 * @details

 * This file provides the implementation for a hash table that resides entirely in

 * the GPU's device memory. It is designed for massively parallel insertions and

 * lookups.

 *

 * @algorithm

 * - **Hash Table Structure**: The hash table is a single array of 64-bit integers,

 *   where each entry packs a 32-bit key and a 32-bit value.

 * - **Collision Resolution**: Open addressing with linear probing is used.

 *   When a collision occurs, the algorithm checks the next consecutive slot.

 * - **Concurrency**: The `insertHash` kernel uses `atomicCAS` (Compare-And-Swap)

 *   to handle race conditions, allowing multiple threads to attempt insertions

 *   into the same slot safely. `atomicInc` is used to track the number of

 *   successful insertions.

 * - **Rehashing**: When the load factor exceeds a threshold, the table is resized.

 *   This is a costly operation that involves transferring all data to the host,

 *   re-calculating hashes for a new table size, and transferring it back to the device.

 */

#include <stdio.h>

#include <stdlib.h>

#include <string.h>

#include <cuda.h>

#include <cuda_runtime.h>

#include <iostream>



#include "gpu_hashtable.hpp"

#define LOAD_FACTOR 0.8f



// Assumes `hash1(key, cap)` is defined in "gpu_hashtable.hpp", likely as `key % cap`.





/**

 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table in parallel.

 * @param hashmap Pointer to the hash table in global device memory.

 * @param cap Total capacity of the hash table.

 * @param keys Device pointer to the array of keys to insert.

 * @param values Device pointer to the array of values to insert.

 * @param numKeys The number of key-value pairs in the batch.

 * @param num_inserted Device pointer to a counter for successful insertions.

 *

 * @details Each thread in the grid processes one key-value pair. It uses linear probing

 * and atomicCAS to find a slot and safely insert the data, even under contention from

 * other threads.

 */

__global__ void insertHash(unsigned long long *hashmap, int cap, int *keys, int *values, int numKeys, unsigned int *num_inserted) {

	// Assign one thread per key-value pair using a standard grid-stride loop pattern.

	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;



	if(i < numKeys)

	{	





		unsigned long long key;

		

		// Pack the 32-bit key and 32-bit value into a single 64-bit integer.

		// Key occupies the most significant 32 bits, value the least significant.

		key = (unsigned long long)keys[i] << 32;

		key += values[i];

		// Calculate the initial bucket index using the hash function.

		int index = hash1(keys[i], cap);

		

		

		

		/**

		 * @brief Linear probing loop to find an available slot.

		 * Invariant: 'index' is the current slot being checked.

		 */

		while(true)

		{

			// Atomically attempt to insert the packed key-value pair.

			// If the slot `hashmap[index]` is 0, `atomicCAS` will write `key` into it

			// and return 0. The old value is returned.

			unsigned long long old_val = atomicCAS(





				(unsigned long long *) &hashmap[index], 

				(unsigned long long) 0, 

				key);



			// If the old value was not 0, the slot was occupied.

			if(old_val != 0)

			{

				// Extract the key from the occupied slot.

				unsigned long long k = old_val >> 32;

				// Check if the keys match.

				if(k == (unsigned long long)keys[i])

				{

					// Key exists, so update the value atomically.

					atomicCAS(&hashmap[index], old_val, key);

					return; // Insertion (update) complete.

				}	

				// Key does not match (collision), so probe the next slot.

				index = (index + 1) % cap;

			} else 

			{

				// The insertion was successful (the slot was empty).

				// Atomically increment the counter for successful insertions.

				atomicInc(num_inserted, 4294967295);

				return;

			}			



		}

	}	

}





/**

 * @brief CUDA kernel to retrieve values for a batch of keys in parallel.

 * @param hash Pointer to the hash table in global device memory.

 * @param cap Total capacity of the hash table.

 * @param keys Device pointer to the array of keys to look up.

 * @param values Device pointer to an array where the found values will be written.

 * @param numKeys The number of keys to look up.

 *

 * @details Each thread searches for one key. It uses linear probing. If the key

 * is found, the corresponding value is written to the output; otherwise, 0 is written.

 */

__global__ void getHash(unsigned long long *hash, int cap, int *keys, int *values, int numKeys) {

	

	// Assign one thread per key lookup.

	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;



	if(i < numKeys)

	{

		int index = hash1(keys[i], cap);

		int start = index; // Store the starting index to detect a full loop.

		

		

		/**

		 * @brief Linear probing loop to find the key.

		 */

		while(true)

		{

			// Check if the key in the current slot matches the search key.

			if(hash[index] >> 32 == keys[i])

			{

				// Key found. Extract the value (lower 32 bits).

				unsigned long long temp = hash[index];

				temp = temp << 32;

				temp = temp >> 32;

				values[i] =(int) temp;

				return;

			} else if( hash[index] == 0) // Note: original code `hash[index] >> 32 == 0` is buggy if key 0 is valid.

			{

				// An empty slot means the key is not in the table.

				values[i] = 0;

				return;

			} else 

			{

				// Collision: move to the next slot.

				index = (index + 1) % cap;

				// If we have looped back to the start, the key is not in the table.

				if(start == index)

				{

					values[i] = 0;

					return;

				}	

			}

		}

	}

	

}

 



// Constructor: Allocates and zeros out the hash table in GPU memory.

GpuHashTable::GpuHashTable(int size) {

	cudaMalloc((void**) &hashmap, size * sizeof(unsigned long long));

	if(!hashmap)

	{

		perror("hashmap alloc");

		return;

	}



	





	cudaMemset(hashmap, 0, size * sizeof(unsigned long long));

	tot_size = size;

	num_elements = 0;

}





// Destructor: Frees the GPU memory.

GpuHashTable::~GpuHashTable() {

	cudaFree(hashmap);

	hashmap = NULL;

	tot_size = 0;

	num_elements = 0;

}





/**

 * @brief Resizes the hash table. This is a very expensive operation.

 * @param numBucketsReshape The new capacity for the hash table.

 * @details This function rehashes all existing elements into a new, larger table.

 * It does this by copying all data to the host, extracting key-value pairs,

 * copying them back to the device, and running the `insertHash` kernel.

 */

void GpuHashTable::reshape(int numBucketsReshape) {

	

	int threads_num_block = 1024, num_blocks = tot_size / threads_num_block + 1;

	int *k_h = 0, *v_h = 0, *k_d = 0, *v_d = 0;

	unsigned long long *temp = 0;



	// 1. Allocate a new, larger table on the device.

	if(cudaMalloc((void**) &temp, numBucketsReshape * sizeof(unsigned long long)) != cudaSuccess)

	{

		perror("temp alloc");

		return;

	}



	

	cudaMemset(temp, 0, numBucketsReshape * sizeof(unsigned long long));



	

	// If there are no elements, just swap the pointers and update the size.

	if(num_elements == 0)

	{

		cudaFree(hashmap);

		hashmap = 0;

		hashmap = temp;

		tot_size = numBucketsReshape;

		return;

	}



	int num_el = 0;



	// 2. Allocate host memory for keys and values.

	k_h = (int *) malloc (num_elements * sizeof(int));

	v_h = (int *) malloc (num_elements * sizeof(int));

	if( !k_h || !v_h)

	{

		perror("k_h v_h");

		return;

	}



	// 3. Copy the entire old hash table from device to host.

	unsigned long long *hash_host = 0;

	hash_host = (unsigned long long *) malloc ( tot_size * sizeof(unsigned long long));

	cudaMemcpy(hash_host, hashmap, tot_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);



	if(!hash_host)

	{

		perror("hash_host alloc");

		return;

	}



	// 4. Iterate through the host copy to extract all valid key-value pairs.

	unsigned long long el;

	for(int i = 0; i < tot_size; i++)

	{





		el = hash_host[i];

		if(el >> 32 != 0) // If key is not 0

		{

			k_h[num_el] = el >> 32;

			el = el << 32;

			el = el >> 32;

			v_h[num_el] = el;

			num_el++;

		}

	}

	

	// Note: The host-side buffer `hash_host` is not freed, causing a memory leak.





	// 5. Allocate device memory and copy the extracted keys/values back to the device.

	if(cudaMalloc((void **) &k_d, num_el * sizeof(int)) != cudaSuccess)

		perror("cudaMalloc");



	if(cudaMalloc((void **) &v_d, num_el * sizeof(int)) != cudaSuccess)

		perror("cudaMalloc");

	

	cudaMemcpy(k_d, k_h, num_el * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(v_d, v_h, num_el * sizeof(int), cudaMemcpyHostToDevice);



	unsigned int *num_inserted = 0;





	cudaMalloc((void **) &num_inserted, sizeof(unsigned int));

	cudaMemset(num_inserted, 0, sizeof(unsigned int));



	// 6. Launch the insert kernel to re-hash all elements into the new table.

	// Note: The `>>` syntax is invalid. It should be `<<<grid, block>>>`.

	insertHash>>(temp, numBucketsReshape, k_d, v_d, num_el, num_inserted);



	cudaDeviceSynchronize();

		

	// 7. Update the hashmap pointer and size, and free all temporary memory.

	hashmap = temp;

	tot_size = numBucketsReshape;



	cudaFree(k_d);

	k_d = 0;

	cudaFree(v_d);	

	v_d = 0;

	cudaFree(num_inserted);

	num_inserted = 0;

	free(k_h);

	k_h = 0;

	free(v_h);

	v_h = 0;

	temp = 0;

}





/**

 * @brief Inserts a batch of key-value pairs from the host.

 * @param keys Host pointer to an array of keys.

 * @param values Host pointer to an array of values.

 * @param numKeys The number of pairs to insert.

 * @return True on success, false on failure.

 *

 * @details Checks if the table needs resizing based on the load factor before

 * transferring data to the device and launching the insertion kernel.

 */

bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {

	int *k_d = 0, *v_d = 0;

	unsigned int *num_inserted_d = 0, *num_inserted_h = 0;

	int acc = 0, threads_num_block = 1024;	

	int num_blocks = numKeys / threads_num_block + 1;



	// Allocate device memory for keys and values.

	cudaMalloc((void**) &k_d, numKeys * sizeof(int));

	if(!k_d)

	{

		perror("key alloc");

		return false;

	}



	cudaMalloc((void**) &v_d, numKeys * sizeof(int));

	if(!v_d)

	{





		printf("value alloc");

		return false;

	}	



	// Copy keys and values from host to device.

	cudaMemcpy(k_d, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(v_d, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);	



	// Count valid keys to be inserted.

	for(int i = 0; i < numKeys; i++)

		if(keys[i] > 0 && values[i] > 0)

			acc++;		



	// Check if resizing is needed based on the load factor.

	int new_cap;

	if( (float)(acc + num_elements) / tot_size > LOAD_FACTOR) {

		new_cap = (acc + num_elements) / LOAD_FACTOR;

		reshape((int) (1.5 * new_cap));

	}





	// Allocate and initialize insertion counter.

	num_inserted_h = (unsigned int *) malloc (sizeof(unsigned int));

	memset(num_inserted_h, 0, sizeof(unsigned int));







	cudaMalloc((void **) &num_inserted_d, sizeof(unsigned int));

	cudaMemcpy(num_inserted_d, num_inserted_h, sizeof(unsigned int), cudaMemcpyHostToDevice);

	

	// Launch the kernel to perform the insertions.

	// Note: The `>>` syntax is invalid. It should be `<<<grid, block>>>`.

	insertHash>>(hashmap, tot_size, k_d, v_d, numKeys, num_inserted_d);

	cudaDeviceSynchronize();



	cudaError_t error = cudaGetLastError();

	if(error != cudaSuccess)

		perror(cudaGetErrorString(error));



	// Copy the number of insertions back to the host and update the total count.

	cudaMemcpy(num_inserted_h, num_inserted_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);	

	num_elements += (*num_inserted_h);



	// Free temporary device and host memory.

	cudaFree(k_d);

	k_d = 0;

	cudaFree(v_d);

	v_d = 0;

	cudaFree(num_inserted_d);

	num_inserted_d = 0;

	free(num_inserted_h);

	num_inserted_h = 0;



	return true;

}





/**

 * @brief Retrieves values for a batch of keys from the host.

 * @param keys Host pointer to an array of keys to look up.

 * @param numKeys The number of keys to look up.

 * @return A host pointer to an array containing the values. The caller is

 *         responsible for freeing this memory.

 */

int* GpuHashTable::getBatch(int* keys, int numKeys) {





	int *k_d, *v_d, *values;

	int threads_num_block = 1024;

	int num_blocks = numKeys / threads_num_block + 1;



	// Allocate device memory for keys and the results.

	if(cudaMalloc((void **) &k_d, numKeys * sizeof(int)) != cudaSuccess)

		perror("cudaMalloc");

	

	if(cudaMalloc((void **) &v_d, numKeys * sizeof(int)) != cudaSuccess)

		perror("cudaMalloc");



	// Copy keys to device.

	cudaMemcpy(k_d, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);



	// Launch the get kernel.

	// Note: The `>>` syntax is invalid. It should be `<<<grid, block>>>`.

	getHash>>(hashmap,tot_size,k_d, v_d, numKeys);



	// Allocate host memory for the results and copy them back from the device.

	values = (int *) malloc (numKeys * sizeof(int));

	cudaMemcpy(values, v_d, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	

	// Free device memory.

	cudaFree(k_d);

	cudaFree(v_d);

	k_d = 0;

	v_d = 0;



	return values;

}





// Returns the current load factor of the hash table.

float GpuHashTable::loadFactor() {

	return num_elements / (tot_size*1.0f); 

}





/**

 * @brief The following macros and include are unconventional. They appear to be

 * part of a specific test harness setup where this file's contents are

 * included directly into another file (`test_map.cpp`) rather than being

 * compiled and linked as a separate object.

 */

#define HASH_INIT GpuHashTable GpuHashTable(1);

#define HASH_RESERVE(size) GpuHashTable.reshape(size);



#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)

#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)



#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()



#include "test_map.cpp"


