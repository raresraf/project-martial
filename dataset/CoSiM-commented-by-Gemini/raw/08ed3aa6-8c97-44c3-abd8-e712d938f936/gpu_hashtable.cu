/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 *
 * @details This implementation uses two separate arrays for keys and values and
 * employs open addressing with linear probing. Concurrency during insertion is
 * managed with atomicCAS. The host-side class handles batch operations and resizing.
 * The header file content is unusually included at the end of this .cu file.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>

#include "gpu_hashtable.hpp" // Logically included, though physically at the end.

/**
 * @brief CUDA kernel to insert a batch of key-value pairs.
 *
 * @details Each thread handles one key-value pair. It calculates an initial hash
 * and then uses linear probing with wrap-around to find a slot. `atomicCAS`
 * ensures thread-safe insertion into an empty slot (key=0) or update of an
 * existing key.
 *
 * @param keyBuffer Pointer to the key array in global memory.
 * @param valueBuffer Pointer to the value array in global memory.
 * @param capacity The total capacity of the hash table arrays.
 * @param keys Input array of keys for this batch.
 * @param values Input array of values for this batch.
 * @param numKeys The number of elements to insert.
 */
__global__ void insert(int *keyBuffer, int *valueBuffer, unsigned int capacity, int *keys, int *values, int numKeys) {
	// Map global thread ID to a key-value pair in the input batch.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= numKeys || keys[i] <= 0) { // Keys must be positive; 0 is reserved for empty slots.
		return;
	}

	long long hashed = ((long long)keys[i] * PRIME1) % PRIME2 % capacity;

	// Linear probing loop. Probes at most 'capacity' times.
	for (int j = 0; j < capacity; j++) {
        // Attempt to atomically claim a slot or find the existing key.
		int result = atomicCAS(&keyBuffer[hashed], 0, keys[i]);
		if (result == 0 || result == keys[i]) {
            // Success: either we claimed an empty slot (result=0) or found the existing key.
			valueBuffer[hashed] = values[i];
			return;
		}
        // Collision: move to the next slot with wrap-around.
		hashed = (hashed + 1) % capacity;
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 *
 * @details Each thread handles one key. It uses the same linear probing logic
 * as the insert kernel to find the key. This is a read-only operation and
 * does not require atomic instructions.
 *
 * @param keyBuffer Pointer to the key array in global memory.
 * @param valueBuffer Pointer to the value array in global memory.
 * @param capacity The total capacity of the hash table arrays.
 * @param keys Input array of keys to look up.
 * @param values Output array for the retrieved values.
 * @param numKeys The number of keys to retrieve.
 */
__global__ void get(int *keyBuffer, int *valueBuffer, unsigned int capacity, int *keys, int *values, int numKeys) {
	// Map global thread ID to a key.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= numKeys) {
		return;
	}

	long long hashed = ((long long)keys[i] * PRIME1) % PRIME2 % capacity;

	// Linear probing loop for searching.
	for (int j = 0; j < capacity; j++) {
        // If we find the key, or an empty slot (meaning the key isn't in the table),
        // we can stop.
		if (keyBuffer[hashed] == keys[i] || keyBuffer[hashed] == 0) {
			values[i] = valueBuffer[hashed];
			return;
		}
        // Collision: move to the next slot with wrap-around.
		hashed = (hashed + 1) % capacity;
	}
}

// =================================================================================
// Host-side GpuHashTable Class Implementation
// =================================================================================

GpuHashTable::GpuHashTable(int size) {
	this->size = 0; // Number of elements currently in the table.
	this->capacity = size; // Total number of available slots.

	// Allocate separate buffers for keys and values on the GPU.
	cudaMalloc(&this->keyBuffer, size * sizeof(int));
	DIE(this->keyBuffer == NULL, "GPU HashTable alloc failed");

	cudaMalloc(&this->valueBuffer, size * sizeof(int));
	DIE(this->valueBuffer == NULL, "GPU HashTable alloc failed");

	// Initialize all slots to 0, marking them as empty.
	cudaMemset(this->keyBuffer, 0, size * sizeof(int));
	cudaMemset(this->valueBuffer, 0, size * sizeof(int));
}

GpuHashTable::~GpuHashTable() {
	cudaFree(this->keyBuffer);
	cudaFree(this->valueBuffer);
}

void GpuHashTable::reshape(int numBucketsReshape) {
	int *GPUkeys, *GPUvalues;
	int numBlocks = (this->capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int oldCapacity = this->capacity;

	// Create a temporary copy of the old data on the GPU.
	cudaMalloc(&GPUkeys, this->capacity * sizeof(int));
	DIE(GPUkeys == NULL, "reshape alloc failed");
	cudaMemcpy(GPUkeys, this->keyBuffer, this->capacity * sizeof(int), cudaMemcpyDeviceToDevice);

	cudaMalloc(&GPUvalues, this->capacity * sizeof(int));
	DIE(GPUvalues == NULL, "reshape alloc failed");
	cudaMemcpy(GPUvalues, this->valueBuffer, this->capacity * sizeof(int), cudaMemcpyDeviceToDevice);

	// Free old buffers.
	cudaFree(this->keyBuffer);
	cudaFree(this->valueBuffer);

	// Allocate new, larger buffers.
	cudaMalloc((void **) &this->keyBuffer, numBucketsReshape * sizeof(int));
	DIE(this->keyBuffer == NULL, "GPU HashTable realloc failed");

	cudaMalloc((void **) &this->valueBuffer, numBucketsReshape * sizeof(int));
	DIE(this->valueBuffer == NULL, "GPU HashTable realloc failed");
	
	cudaMemset(this->keyBuffer, 0, numBucketsReshape * sizeof(int));
	cudaMemset(this->valueBuffer, 0, numBucketsReshape * sizeof(int));

	this->capacity = numBucketsReshape;

	// Re-insert all elements from the temporary copy into the new table.
	insert<<<numBlocks, BLOCK_SIZE>>>(this->keyBuffer, this->valueBuffer, this->capacity, GPUkeys, GPUvalues, oldCapacity);
	cudaDeviceSynchronize();

	cudaFree(GPUkeys);
	cudaFree(GPUvalues);
}

bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	float loadAfterInsert = (float)(this->size + numKeys) /  this->capacity;
	int numBlocks = (numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int *GPUkeys, *GPUvalues;
	
	// If the load factor will exceed the maximum, resize the table.
	if (loadAfterInsert > MAX_LOAD) {
		int resize = (this->size + numKeys) / MIN_LOAD;
		this->reshape(resize);
	}

	// Allocate temporary GPU buffers for the batch.
	cudaMalloc(&GPUkeys, numKeys * sizeof(int));
	DIE(GPUkeys == NULL, "insert alloc failed");
	cudaMemcpy(GPUkeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&GPUvalues, numKeys * sizeof(int));
	DIE(GPUvalues == NULL, "insert alloc failed");
	cudaMemcpy(GPUvalues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Launch the insert kernel.
	insert<<<numBlocks, BLOCK_SIZE>>>(this->keyBuffer, this->valueBuffer, this->capacity, GPUkeys, GPUvalues, numKeys);
	cudaDeviceSynchronize();

	this->size += numKeys; // Unsafe if multiple host threads call this concurrently.

	cudaFree(GPUkeys);
	cudaFree(GPUvalues);

	return true;
}

int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *values, *GPUvalues, *GPUkeys;
	int numBlocks = (numKeys + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// Allocate GPU memory for keys and for the results.
	cudaMalloc(&GPUkeys, numKeys * sizeof(int));
	DIE(GPUkeys == NULL, "get alloc failed");
	cudaMemcpy(GPUkeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&GPUvalues, numKeys * sizeof(int));
	DIE(GPUvalues == NULL, "get alloc failed");

	// Allocate host memory for the final results.
	values = (int *)calloc(numKeys, sizeof(int));
	DIE(values == NULL, "get alloc failed");

	// Launch the get kernel.
	get<<<numBlocks, BLOCK_SIZE>>>(this->keyBuffer, this->valueBuffer, this->capacity, GPUkeys, GPUvalues, numKeys);
	cudaDeviceSynchronize();

	// Copy results from GPU back to host.
	cudaMemcpy(values, GPUvalues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(GPUkeys);
	cudaFree(GPUvalues);

	return values;
}

float GpuHashTable::loadFactor() {
	return (float)(this->size) /  this->capacity;
}

// =================================================================================
// Test Macros and Inlined Header Content
// =================================================================================

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"

/**
 * @file gpu_hashtable.hpp
 * @brief (Inlined) Header file for the GPU-accelerated hash table.
 */
#ifndef _HASHCPU_
#define _HASHCPU_

#include <string>

// Note: using namespace std is active from the top of the .cu file.
using namespace std;

// Defines various constants for the hash table's operation.
#define	KEY_INVALID		0 // Value for an empty key slot.
#define BLOCK_SIZE 256
#define MIN_LOAD 0.81 // Minimum load factor after resizing.
#define MAX_LOAD 0.9  // Load factor threshold that triggers a resize.
#define PRIME1 10193llu // Primes for hashing.
#define PRIME2 6904869625999llu

/**
 * @brief Macro for error handling, particularly for CUDA calls.
 */
#define DIE(assertion, call_description) \
	do { \
		if (assertion) { \
		fprintf(stderr, "(%s, %d): ", \
		__FILE__, __LINE__); \
		perror(call_description); \
		exit(errno); \
	} \
} while (0)

// A large list of prime numbers, likely for use in hashing functions.
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

// Unused hash function declarations.
int hash1(int data, int limit);
int hash2(int data, int limit);
int hash3(int data, int limit);

/**
 * @class GpuHashTable
 * @brief Host-side management class for the GPU hash table.
 *
 * @details This class encapsulates the GPU hash table, providing methods to
 * insert and retrieve data in batches, resize the table, and query its state.
 */
class GpuHashTable
{
	public:
		// These members are public, which is not ideal encapsulation.
		int *keyBuffer, *valueBuffer;
		unsigned int size;
		unsigned int capacity;

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