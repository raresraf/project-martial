
/**
 * @file gpu_hashtable.cu
 * @brief High-performance GPGPU hash table implementation using open addressing and linear probing.
 * 
 * Functional Intent: Provides a massively parallel key-value store optimized for CUDA-enabled 
 * accelerators. Employs atomic compare-and-swap (atomicCAS) for thread-safe concurrent 
 * insertions and features dynamic resizing (rehashing) to maintain O(1) expected 
 * search complexity under high load factors.
 * 
 * Domain: HPC, Parallel Data Structures, CUDA.
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "gpu_hashtable.hpp"


/**
 * @brief Constructor: Allocates and clears global memory for the hash table structure.
 * 
 * @param size Initial capacity (number of buckets) for the hash table.
 */
GpuHashTable::GpuHashTable(int size)
{
	cudaError_t error;

	// Memory Hierarchy: Allocates the core entry array in GPU VRAM (Global Memory).
	error = cudaMalloc(&(GpuHashTable::hashtable), size * sizeof(hT));
	DIE(error != cudaSuccess || GpuHashTable::hashtable == NULL, "cudaMalloc hashtable error");
	
	// Pre-condition: All keys must be initialized to 0 (KEY_INVALID) for probing logic to function.
	error = cudaMemset(GpuHashTable::hashtable, 0, size * sizeof(hT));
	DIE(error != cudaSuccess, "cudaMemset hashtable error");

	GpuHashTable::currentTableSize = 0;
	GpuHashTable::tableSize = size;
}


/**
 * @brief Destructor: Releases device-resident memory resources.
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
 * @brief CUDA kernel for identifying and extracting valid entries during a reshape operation.
 * 
 * Logic: Performs a parallel sweep over the old table, using an atomic counter to pack 
 * sparse active entries into dense temporary buffers for subsequent rehashing.
 */
__global__ void copyForReshape(hT *hashtable, int tableSize,
								int *device_keys, int *device_values,
								int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	// Invariant: Only threads mapped to valid buckets contribute to the migration buffers.
	if(idx < tableSize) {
		if (hashtable[idx].key != 0) {
			
			// Synchronization: Uses atomicAdd to coordinate insertion index across thousands of threads.
			int index = atomicAdd(counter, 1);

			device_keys[index] = hashtable[idx].key;
			device_values[index] = hashtable[idx].value;
		}
	}
}


/**
 * @brief Dynamically expands the hash table and re-populates it with existing data.
 * 
 * Algorithm: Parallel migration and re-insertion.
 * 1. Allocates a larger buffer.
 * 2. Compresses existing entries into a dense linear list on the device.
 * 3. Pulls compressed data to host and re-inserts into the new device buffer.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t error;
	hT *newHashTable = NULL;
	// Optimization: Oversizes the table by 20% to reduce probe collisions and maintain performance.
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

		
		// Block Logic: Executing parallel extraction of valid mappings.
		copyForReshape<<<blocks_no, block_size>>>(GpuHashTable::hashtable,
												GpuHashTable::tableSize,
												device_keys, device_values,
												counter);
		// Synchronization: Blocks CPU until all GPU extraction tasks are completed.
		error = cudaDeviceSynchronize();
		DIE(error != cudaSuccess, "cudaDeviceSynchronize error");


		error = cudaMemcpy(host_keys, device_keys,
							GpuHashTable::currentTableSize * sizeof(int),
							cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_keys error");

		error = cudaMemcpy(host_values, device_values,
							GpuHashTable::currentTableSize * sizeof(int),
							cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_values error");


		GpuHashTable::~GpuHashTable();

		// Update Table State
		GpuHashTable::tableSize = new_size;
		GpuHashTable::hashtable = newHashTable;
		GpuHashTable::currentTableSize = 0;
		

		int numKeys = 0;
		error = cudaMemcpy(&numKeys, counter, sizeof(int), cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy numKeys error");
		
		// Recursive Re-insertion: populates the new table using the standard batch API.
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
 * @brief CUDA kernel for calculating multiplicative hash indices for a batch of keys.
 */
__global__ void getHashCode(int *keys, int *hashcodes, int numkeys, int tablesize)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numkeys)
		// Algorithm: 64-bit integer multiplicative hashing for high dispersion.
		hashcodes[idx] = (((long)keys[idx]) * 1402982055436147llu)
								% 452517535812813007llu % tablesize;
}


/**
 * @brief CUDA kernel for parallel entry insertion using linear probing.
 * 
 * Functional Utility: Attempts to claim a bucket for each input key. If a collision occurs 
 * (bucket taken by a different key), it incrementaly probes the next index in a circular loop.
 * 
 * @param counter Tracks successful *new* insertions to update the global occupancy state.
 */
__global__ void insertKeysandValues(int *hashcodes, int *keys, int *values,
									int numKeys, hT *hashtable,
									int *currentTableSize, int tablesize,
									int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numKeys) {
		if (keys[idx] == 0)
			return;

		/**
		 * Synchronization: Atomic Compare-and-Swap.
		 * Logic: Claims an empty bucket by setting its key. Returns the prior key value.
		 */
		int key = atomicCAS(&hashtable[hashcodes[idx]].key, 0, keys[idx]);
		
		if (key == 0) {
			// Logic: Success, slot was empty.
			hashtable[hashcodes[idx]].value = values[idx];
			
			// Inline: Sentinel value to mark this specific input element as processed.
			keys[idx] = 0;
			atomicAdd(currentTableSize, 1);
			atomicAdd(counter, 1);
		} else if (key == keys[idx]) {
			// Logic: Slot already held this exact key; update the value (upsert).
			hashtable[hashcodes[idx]].value = values[idx];
			atomicAdd(counter, 1);
			
			keys[idx] = 0;
		} else {
			// Block Logic: Collision resolution.
			// Invariant: Circularly increments the probe index.
			hashcodes[idx] += 1;
			hashcodes[idx] %= tablesize;
		}
	}
}


/**
 * @brief Public interface for parallel batch insertion from host memory.
 * 
 * Optimization: Automatically triggers a global table reshape if current 
 * occupancy + batch size exceeds capacity.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys = NULL;
	int *device_values = NULL;
	cudaError_t error;


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


	int *hashcodes = NULL;
	error = cudaMalloc(&hashcodes, numKeys * sizeof(int));
	DIE(error != cudaSuccess || hashcodes == NULL, "cudaMalloc hashcodes error");

	const size_t block_size = 1024;
	size_t blocks_no = numKeys / block_size;
 
	if (numKeys % block_size) 
		++blocks_no;

	// Parallel Phase 1: Compute initial hash buckets.
	getHashCode<<<blocks_no, block_size>>>(device_keys, hashcodes,
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
	
	/**
	 * Block Logic: Iterative probing synchronization.
	 * Logic: Continues re-invoking the insertion kernel as long as some keys remain 
	 * uninserted due to bucket collisions.
	 */
	while(1) {
		
		insertKeysandValues<<<blocks_no, block_size>>>(hashcodes, device_keys,
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

		// Invariant: Exits when the number of successfully placed keys matches the batch size.
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
 * @brief CUDA kernel for parallel batch retrieval using linear probing.
 */
__global__ void getbatch(int *values, int *hashcodes, int *keys, int numKeys,
						hT *hashtable, int tablesize, int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numKeys) {
		if (keys[idx] == 0)
			return;

		// Logic: Checks if the current probed bucket contains the target key.
		if(hashtable[hashcodes[idx]].key == keys[idx]) {
			
			values[idx] = hashtable[hashcodes[idx]].value;
			atomicAdd(counter, 1);
			// Inline: Zeroes out the input key to mark search fulfillment.
			keys[idx] = 0;
		} else {
			// Block Logic: Search probing.
			hashcodes[idx] += 1;
			hashcodes[idx] %= tablesize;
		}
	}
}


/**
 * @brief Public interface for parallel key lookup from host memory.
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


	
	int *hashcodes = NULL;

	error = cudaMalloc(&hashcodes, numKeys * sizeof(int));
	DIE(error != cudaSuccess || hashcodes == NULL, "cudaMalloc hashcodes error");

	const size_t block_size = 1024;
	size_t blocks_no = numKeys / block_size;
 
	if (numKeys % block_size) 
		++blocks_no;

	getHashCode<<<blocks_no, block_size>>>(device_keys, hashcodes, numKeys, GpuHashTable::tableSize);
	error = cudaDeviceSynchronize();
	DIE(error != cudaSuccess, "cudaDeviceSynchronize error");



	int *device_counter = NULL;
	error = cudaMalloc(&device_counter, sizeof(int));
	DIE(error != cudaSuccess || device_counter == NULL, "cudaMalloc device_counter error");
	error = cudaMemset(device_counter, 0, sizeof(int));
	DIE(error != cudaSuccess, "cudaMemset device_counter error");

	/**
	 * Block Logic: Iterative search synchronization.
	 * Invariant: Probes until all keys are found or the search space is exhausted.
	 */
	while(1) {
		getbatch<<<blocks_no, block_size>>>(device_ret_values, hashcodes,
											device_keys, numKeys,
											GpuHashTable::hashtable,
											GpuHashTable::tableSize, device_counter);
		error = cudaDeviceSynchronize();
		DIE(error != cudaSuccess, "cudaDeviceSynchronize error");

		int host_counter = 0;
		error = cudaMemcpy(&host_counter, device_counter, sizeof(int), cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_counter error");

		if(host_counter == numKeys)
			break;
	}


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
 * @brief Returns the current occupancy ratio of the hash table.
 */
float GpuHashTable::loadFactor() {
	return ((float)GpuHashTable::currentTableSize) / ((float)GpuHashTable::tableSize);
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

#include <iostream>
#include <vector>

using namespace std;

#define	KEY_INVALID		0

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)
	
/**
 * @brief Pre-computed prime numbers for robust multiplicative hashing.
 */
const size_t primeList[] =
{
	2llu, 3llu, 5llu, 7llu, 11llu, 13llu, 17llu, 23llu, 29llu, 37llu, 47llu,
	59llu, 73llu, 97llu, 127llu, 151llu, 197llu, 251llu, 313llu, 397llu,
	499llu, 631llu, 797llu, 1009llu, 1259llu, 1597llu, 2011llu, 2539llu,
	3203llu, 4027llu, 5087llu, 6421llu, 8089llu, 10193llu, 12853llu, 16193llu,
...
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

/**
 * @struct hashtableCell
 * @brief Representation of a single key-value mapping in GPU memory.
 */
typedef struct hashtableCell{
	int key;
	int value;
} hT;


/**
 * @class GpuHashTable
 * @brief Controller for managing a GPU-resident hash table.
 */
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
