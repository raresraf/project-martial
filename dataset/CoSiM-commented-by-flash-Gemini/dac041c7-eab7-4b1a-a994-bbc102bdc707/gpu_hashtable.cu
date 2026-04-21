
#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "gpu_hashtable.hpp"

/**
 * @file gpu_hashtable.cu
 * @brief High-performance GPGPU Hash Table using Open Addressing and Atomic Operations.
 * 
 * Functional Intent: Implements a thread-safe hash table optimized for parallel 
 * insertion and retrieval on NVIDIA GPUs. It employs a linear probing strategy 
 * for collision resolution, leveraging `atomicCAS` to manage concurrent updates 
 * to the key-value slots. The architecture supports dynamic resizing (reshape) 
 * by migrating existing entries to a larger device-side memory buffer.
 * 
 * Domain: HPC, Parallel Data Structures, CUDA.
 */

/**
 * GpuHashTable constructor - Initializes global memory on the device.
 */
GpuHashTable::GpuHashTable(int size)
{
	cudaError_t error;

	// Optimization: Zero-initialization of the hash table memory using cudaMemset.
	// This ensures that all 'key' fields are initially 0 (KEY_INVALID).
	error = cudaMalloc(&(GpuHashTable::hashtable), size * sizeof(hT));
	DIE(error != cudaSuccess || GpuHashTable::hashtable == NULL, "cudaMalloc hashtable error");
	error = cudaMemset(GpuHashTable::hashtable, 0, size * sizeof(hT));
	DIE(error != cudaSuccess, "cudaMemset hashtable error");

	GpuHashTable::currentTableSize = 0;
	GpuHashTable::tableSize = size;
}

/**
 * GpuHashTable destructor - Safely releases device-side memory.
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
 * @kernel copyForReshape
 * @brief Collects all non-empty entries from the hash table into a compact device array.
 * 
 * Logic: Each thread inspects a single bucket. If non-empty, it uses `atomicAdd` 
 * on a global counter to obtain a unique destination index in the compact buffers.
 */
__global__ void copyForReshape(hT *hashtable, int tableSize,
								int *device_keys, int *device_values,
								int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if(idx < tableSize) {
		if (hashtable[idx].key != 0) {
			// Synchronization: Atomic counter ensures non-overlapping writes to compact array.
			int index = atomicAdd(counter, 1);

			device_keys[index] = hashtable[idx].key;
			device_values[index] = hashtable[idx].value;
		}
	}
}

/**
 * reshape - Resizes the hash table to accommodate a larger dataset.
 * 
 * Algorithm: Full Table Migration.
 * 1. Allocates a new, larger device buffer (1.2x safety factor).
 * 2. Compares active entries using `copyForReshape` kernel.
 * 3. Pulls active entries to host and re-inserts into the new table.
 * 
 * Invariant: Successfully migrates all data while maintaining the hash table's 
 * internal consistency during the pointer swap.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t error;
	hT *newHashTable = NULL;
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

		// Block Logic: Kernel execution to extract active keys/values.
		copyForReshape<<<blocks_no, block_size>>>(GpuHashTable::hashtable,
												GpuHashTable::tableSize,
												device_keys, device_values,
												counter);
		error = cudaDeviceSynchronize();
		DIE(error != cudaSuccess, "cudaDeviceSynchronize error");

		// Synchronization: Host-Device reconciliation of extracted data.
		error = cudaMemcpy(host_keys, device_keys,
							GpuHashTable::currentTableSize * sizeof(int),
							cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_keys error");

		error = cudaMemcpy(host_values, device_values,
							GpuHashTable::currentTableSize * sizeof(int),
							cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_values error");

		GpuHashTable::~GpuHashTable();

		// Invariant: Update internal state to point to the new, larger table.
		GpuHashTable::tableSize = new_size;
		GpuHashTable::hashtable = newHashTable;
		GpuHashTable::currentTableSize = 0;
		
		int numKeys = 0;
		error = cudaMemcpy(&numKeys, counter, sizeof(int), cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy numKeys error");
		
		// Functional Utility: Re-hashes all existing keys into the new table structure.
		insertBatch(host_keys, host_values, numKeys);

		error = cudaFree(device_keys);
		error = cudaFree(device_values);
		error = cudaFree(counter);
		free(host_keys);
		free(host_values);
		return;
	}

	GpuHashTable::~GpuHashTable();
	GpuHashTable::tableSize = new_size;
	GpuHashTable::hashtable = newHashTable;
}

/**
 * @kernel getHashCode
 * @brief Computes a high-entropy hash for a batch of keys using linear congruential principles.
 */
__global__ void getHashCode(int *keys, int *hashcodes, int numkeys, int tablesize)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numkeys)
		hashcodes[idx] = (((long)keys[idx]) * 1402982055436147llu)
								% 452517535812813007llu % tablesize;
}

/**
 * @kernel insertKeysandValues
 * @brief Performs atomic insertion of keys into the device hash table.
 * 
 * Algorithm: Open Addressing with Linear Probing.
 * 1. Checks if the key in the batch is valid.
 * 2. Attempts to claim a bucket using `atomicCAS`.
 * 3. If `atomicCAS` returns 0, the thread successfully claimed an empty slot; update the value.
 * 4. If `atomicCAS` returns the same key, perform an in-place update.
 * 5. Otherwise (collision), increment the hash index (linear probing) for next attempt.
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

		// Synchronization: atomicCAS ensures only one thread claims an empty bucket.
		int key = atomicCAS(&hashtable[hashcodes[idx]].key, 0, keys[idx]);
		
		if (key == 0) {
			// Case: Successfully claimed an empty slot.
			hashtable[hashcodes[idx]].value = values[idx];
			keys[idx] = 0; // Marker to stop further attempts for this key.
			atomicAdd(currentTableSize, 1);
			atomicAdd(counter, 1);
		} else if (key == keys[idx]) {
			// Case: Key already exists, perform update.
			hashtable[hashcodes[idx]].value = values[idx];
			atomicAdd(counter, 1);
			keys[idx] = 0;
		} else {
			// Case: Collision; update local hash index for re-evaluation in the next pass.
			hashcodes[idx] += 1;
			hashcodes[idx] %= tablesize;
		}
	}
}

/**
 * insertBatch - Orchestrates batch insertion of key-value pairs from host to device.
 * 
 * Logic: Implements an iterative retry mechanism. If collisions prevent all 
 * keys from being inserted in one kernel pass, the kernel is re-launched 
 * (probing new positions) until all keys are successfully mapped.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys = NULL;
	int *device_values = NULL;
	cudaError_t error;

	// Pre-condition: Check load factor and resize if necessary to maintain O(1) performance.
	if ((GpuHashTable::currentTableSize + numKeys) > GpuHashTable::tableSize)
		reshape((GpuHashTable::currentTableSize + numKeys));

	error = cudaMalloc(&device_keys, numKeys * sizeof(int));
	error = cudaMalloc(&device_values, numKeys * sizeof(int));
	error = cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	error = cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int *hashcodes = NULL;
	error = cudaMalloc(&hashcodes, numKeys * sizeof(int));

	const size_t block_size = 1024;
	size_t blocks_no = (numKeys + block_size - 1) / block_size;

	getHashCode<<<blocks_no, block_size>>>(device_keys, hashcodes, numKeys, GpuHashTable::tableSize);
	cudaDeviceSynchronize();

	int *device_current = NULL;
	cudaMalloc(&device_current, sizeof(int));
	cudaMemset(device_current, 0, sizeof(int));

	int *device_counter = NULL;
	cudaMalloc(&device_counter, sizeof(int));
	cudaMemset(device_counter, 0, sizeof(int));

	int host_counter = 0;
	int old_counter = 0;
	
	/**
	 * Block Logic: Insertion Retry Loop.
	 * Invariant: Loop continues until 'host_counter' equals 'numKeys', 
	 * indicating every key in the batch has been successfully placed in the table.
	 */
	while(1) {
		insertKeysandValues<<<blocks_no, block_size>>>(hashcodes, device_keys,
													device_values, numKeys,
													GpuHashTable::hashtable,
													device_current,
													GpuHashTable::tableSize,
													device_counter);
		cudaDeviceSynchronize();

		old_counter = host_counter;
		cudaMemcpy(&host_counter, device_counter, sizeof(int), cudaMemcpyDeviceToHost);
		GpuHashTable::currentTableSize += host_counter - old_counter;

		if(host_counter == numKeys)
			break;
	}

	cudaFree(device_keys);
	cudaFree(device_values);
	cudaFree(hashcodes);
	cudaFree(device_current);
	cudaFree(device_counter);
	return true;
}

/**
 * @kernel getbatch
 * @brief Performs batch retrieval of values associated with provided keys.
 * 
 * Logic: Similar to insertion, uses linear probing. If the bucket's key 
 * matches the search key, retrieval is complete. Otherwise, increments 
 * the hash index and awaits the next kernel pass.
 */
__global__ void getbatch(int *values, int *hashcodes, int *keys, int numKeys,
						hT *hashtable, int tablesize, int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numKeys) {
		if (keys[idx] == 0)
			return;

		if(hashtable[hashcodes[idx]].key == keys[idx]) {
			// Case: Match found.
			values[idx] = hashtable[hashcodes[idx]].value;
			atomicAdd(counter, 1);
			keys[idx] = 0;
		} else {
			// Case: Collision or empty slot; probe next.
			hashcodes[idx] += 1;
			hashcodes[idx] %= tablesize;
		}
	}
}

/**
 * getBatch - Retrieves values for a set of keys in parallel.
 * 
 * Logic: Employs the same iterative retry logic as insertion to resolve 
 * collisions encountered during the search.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *ret_values = (int *)malloc(numKeys * sizeof(int));
	cudaError_t error;

	int *device_ret_values = NULL;
	cudaMalloc(&device_ret_values, numKeys * sizeof(int));

	int *device_keys = NULL;
	cudaMalloc(&device_keys, numKeys * sizeof(int));
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int *hashcodes = NULL;
	cudaMalloc(&hashcodes, numKeys * sizeof(int));

	const size_t block_size = 1024;
	size_t blocks_no = (numKeys + block_size - 1) / block_size;

	getHashCode<<<blocks_no, block_size>>>(device_keys, hashcodes, numKeys, GpuHashTable::tableSize);
	cudaDeviceSynchronize();

	int *device_counter = NULL;
	cudaMalloc(&device_counter, sizeof(int));
	cudaMemset(device_counter, 0, sizeof(int));

	while(1) {
		getbatch<<<blocks_no, block_size>>>(device_ret_values, hashcodes,
											device_keys, numKeys,
											GpuHashTable::hashtable,
											GpuHashTable::tableSize, device_counter);
		cudaDeviceSynchronize();

		int host_counter = 0;
		cudaMemcpy(&host_counter, device_counter, sizeof(int), cudaMemcpyDeviceToHost);

		if(host_counter == numKeys)
			break;
	}

	cudaMemcpy(ret_values, device_ret_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(device_keys);
	cudaFree(device_ret_values);
	cudaFree(hashcodes);
	cudaFree(device_counter);
	return ret_values;
}

/**
 * loadFactor - Computes the current utilization ratio of the hash table.
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
