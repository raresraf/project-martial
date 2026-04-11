/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 *
 * @details
 * This file provides a GPU-based hash table using open addressing with a
 * linear probing collision resolution strategy. The implementation supports
 * batch insertion, batch retrieval, and dynamic resizing (reshaping) based
 * on the table's load factor.
 *
 * @algorithm
 * - Hashing: A bitwise hash function, `hashFunc`, maps keys to indices.
 * - Collision Resolution: Linear probing is employed. When a hash collision
 *   occurs, the algorithm probes the next bucket sequentially (with wrap-around)
 *   until an empty slot is found.
 * - Concurrency: The `atomicCAS` (Compare-And-Swap) operation is used during
 *   insertions and reshaping to handle race conditions, ensuring that multiple
 *   threads can safely write to the hash table.
 *
 * @note The file follows an unconventional structure, appearing to concatenate the
 * implementation with its own header definitions and a test file include.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @details Each thread handles one key-value pair, using linear probing and
 * atomicCAS to find an empty slot or update an existing key.
 */
__global__ void insert_entries(int* keys, int* values, int numKeys, KeyValue* hashtable, unsigned int capacity) {
	// Thread Indexing: Each thread is assigned one key-value pair from the batch.
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int key, value, hashedKey, oldVal;

	if (idx >= numKeys)
		return;

	key = keys[idx];
	value = values[idx];
	hashedKey = hashFunc(key, capacity);

	// Block Logic: Linear probing loop to find an available slot.
	while (true) {
		// Synchronization: Atomically attempt to claim an empty slot (KEY_INVALID).
		oldVal = atomicCAS(&hashtable[hashedKey].key, KEY_INVALID, key);

		// Case 1: The slot was empty and we successfully claimed it.
		// Case 2: The slot already contained our key.
		if (oldVal == KEY_INVALID || oldVal == key) {
			hashtable[hashedKey].value = value; // Set or update the value.
			break; // Exit on success.
		}

		// Collision: Probe the next slot with wrap-around.
		++hashedKey;
		hashedKey %= capacity;
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @details Each thread searches for one key using linear probing.
 */
__global__ void get_values(int* keys, int numKeys, KeyValue* hashtable, unsigned int capacity, int* deviceResult) {
	// Thread Indexing: Each thread is assigned one key to look up.
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int key, hashedKey, count;

	if (idx >= numKeys)
		return;

	key = keys[idx];
	count = capacity + 1; // Safety break to prevent potential infinite loops.
	hashedKey = hashFunc(key, capacity);

	// Block Logic: Linear probing search.
	while (count) {
		if (hashtable[hashedKey].key == key) {
			// Key found, store the value in the output array.
			deviceResult[idx] = hashtable[hashedKey].value;
			break;
		}
		// If an empty slot is found, the key is not in the table. This is an
		// implicit stop condition for the search in a non-full table.
		
		count--;
		hashedKey++;
		hashedKey %= capacity;
	}
}

/**
 * @brief CUDA kernel to copy and rehash all elements from an old table to a new one.
 * @details Each thread is responsible for rehashing one element from the old table.
 */
__global__ void copy_and_rehash(KeyValue *dst, KeyValue *src,
			unsigned int oldSize, unsigned int newSize) {
	// Thread Indexing: Each thread is assigned one slot from the old table.
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Pre-condition: Only process valid indices and non-empty slots.
	if (idx >= oldSize || src[idx].key == KEY_INVALID)
		return;

	unsigned int newSlot = hashFunc(src[idx].key, newSize);

	// Block Logic: Linear probing loop to insert the old element into the new table.
	while (true) {
		// Synchronization: Atomically claim an empty slot in the new table.
		unsigned int oldVal = atomicCAS(&dst[newSlot].key, KEY_INVALID, src[idx].key);

		if (oldVal == KEY_INVALID) {
			dst[newSlot].value = src[idx].value;
			break;
		}
		
		// Collision: Probe the next slot.
		++newSlot;
		newSlot %= newSize;
	}
}

/**
 * @brief Constructor for GpuHashTable.
 */
GpuHashTable::GpuHashTable(int size) {
	cudaError_t err;

	capacity = size;
	occupancy = 0;

	// Memory Allocation: Allocate the main array for key-value pairs on the device.
	err = cudaMalloc((void**) &hashtable, size * sizeof(KeyValue));
	DIE(err != cudaSuccess, "cudaMalloc");

	// Initialize all keys to KEY_INVALID (0).
	err = cudaMemset((void *) hashtable, KEY_INVALID, size * sizeof(KeyValue));
	DIE(err != cudaSuccess, "cudaMemset");
}

/**
 * @brief Destructor for GpuHashTable.
 */
GpuHashTable::~GpuHashTable() {
	cudaError_t err;
	err = cudaFree(hashtable);
	DIE(err != cudaSuccess, "cudaFree");
}

/**
 * @brief Resizes the hash table and rehashes all elements.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t err;
	KeyValue *newTable;
	unsigned int numBlocks = (capacity / BLOCK_SIZE) + 1;

	// Memory Allocation: Allocate a new, larger table on the device.
	err = cudaMalloc((void **) &newTable, numBucketsReshape * sizeof(KeyValue));
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMemset((void *) newTable, KEY_INVALID, numBucketsReshape * sizeof(KeyValue));
	DIE(err != cudaSuccess, "cudaMemset");

	// Launch the kernel to rehash elements into the new table.
	copy_and_rehash>>(newTable, hashtable, capacity, numBucketsReshape);
	cudaDeviceSynchronize();

	// Free the old table memory and update class members.
	err = cudaFree(hashtable);
	DIE(err != cudaSuccess, "cudaFree");

	hashtable = newTable;
	capacity = numBucketsReshape;
}

/**
 * @brief Inserts a batch of keys and values into the hash table.
 * @return True on success, false on failure.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	cudaError_t err;
	int *deviceKeys, *deviceValues;
	size_t numBytes = numKeys * sizeof(int);
	unsigned int numBlocks = (numKeys / BLOCK_SIZE) + 1;

	// Block Logic: Check if the load factor will exceed the max threshold and reshape if needed.
	// The new capacity is calculated to bring the load factor down to MIN_LOAD_FACTOR.
	if (((float) occupancy + (float) numKeys) / (float) capacity >= MAX_LOAD_FACTOR)
		reshape((int) (((float) occupancy + (float) numKeys) / MIN_LOAD_FACTOR));
	
	// Memory Allocation: Allocate temporary device buffers for the batch.
	err = cudaMalloc((void **) &deviceKeys, numBytes);
	DIE(err != cudaSuccess, "cudaMalloc");
	err = cudaMalloc((void **) &deviceValues, numBytes);
	DIE(err != cudaSuccess, "cudaMalloc");
	
	// Memory Transfer: Copy batch data from host to device.
	err = cudaMemcpy(deviceKeys, keys, numBytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");
	err = cudaMemcpy(deviceValues, values, numBytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	// Launch the insertion kernel.
	insert_entries>>(deviceKeys, deviceValues, numKeys, hashtable, capacity);
	cudaDeviceSynchronize();

	occupancy += numKeys;

	// Free temporary device memory.
	err = cudaFree(deviceKeys);
	DIE(err != cudaSuccess, "cudaFree");
	err = cudaFree(deviceValues);
	DIE(err != cudaSuccess, "cudaFree");
	
	return true;
}

/**
 * @brief Retrieves values for a batch of keys.
 * @return A host pointer to an array of results. Caller must free this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	cudaError_t err;
	int *deviceResult;
	int *deviceKeys;
	unsigned int numBlocks = (numKeys / BLOCK_SIZE) + 1;
	size_t numBytes = numKeys * sizeof(int);

	// Memory Management: This section uses conditional compilation to handle memory
	// allocation differently, possibly for different hardware architectures (e.g., IBM Power vs. others).
	// On non-IBM systems, it uses `cudaMallocManaged`, which creates memory accessible
	// by both the host and device, simplifying memory transfers.
	// On IBM systems, it uses separate host and device allocations.
#ifndef IBM
	err = cudaMallocManaged((void **) &deviceResult, numBytes);
#else
	int* result = new int[numKeys];
	err = cudaMalloc((void **) &deviceResult, numBytes);
#endif
	DIE(err != cudaSuccess, "cudaMallocManaged/cudaMalloc");

	// Allocate and copy keys to the device.
	err = cudaMalloc((void **) &deviceKeys, numBytes);
	DIE(err != cudaSuccess, "cudaMalloc");
	err = cudaMemcpy(deviceKeys, keys, numBytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	// Launch the get kernel.
	get_values>>(deviceKeys, numKeys, hashtable, capacity, deviceResult);
	cudaDeviceSynchronize();

	err = cudaFree(deviceKeys);
	DIE(err != cudaSuccess, "cudaFree");

#ifdef IBM
	// If not using managed memory, explicitly copy results back to the host.
	err = cudaMemcpy(result, deviceResult, numBytes, cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, "cudaMemcpy");
	err = cudaFree(deviceResult);
	DIE(err != cudaSuccess, "cudaFree");
	return result;
#else
	// With managed memory, the host can access the device pointer directly after synchronization.
	return deviceResult;
#endif
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
	return (capacity == 0)? 0.f : (float (occupancy) / (float) capacity);
}


// --- Convenience macros for external API ---
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

/*
 * @note The following line includes a .cpp file, which is highly unconventional.
 */
#include "test_map.cpp"

/*
 * @note The following block appears to be a re-definition of the header file content.
 */
#ifndef _HASHCPU_
#define _HASHCPU_

#include 

using namespace std;

// Value representing an empty slot in the hash table.
#define	KEY_INVALID		0
// Constants defining the load factor thresholds for reshaping.
#define MIN_LOAD_FACTOR	0.8f
#define MAX_LOAD_FACTOR	0.95f
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
// Default number of threads per block for kernel launches.
#define BLOCK_SIZE 512

// Macro for fatal error checking.
#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)
	
// A list of large prime numbers (unused in this specific file).
__device__ const size_t primeList[] =
{
	2llu, 3llu, 5llu, 7llu, 11llu, 13llu, 17llu, 23llu, 29llu, 37llu, 47llu,
    // ... (list truncated for brevity)
	11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu
};


/**
 * @brief Device-side hash function.
 * @note Uses bitwise operations for hashing.
 */
 __device__ int hashFunc(int data, int limit) {
	data = ((data >> 16) ^ data) * 0x45d9f3b;
	data = ((data >> 16) ^ data) * 0x45d9f3b;
	data = (data >> 16) ^ data;
	return abs(data) % limit;
 }

/**
 * @struct KeyValue
 * @brief Represents a single key-value pair in the hash table.
 */
struct KeyValue {
	unsigned int key;
	unsigned int value;
};


/**
 * @class GpuHashTable
 * @brief Host-side class for managing the GPU hash table.
 */
class GpuHashTable
{
	private:
		unsigned int capacity;
		unsigned int occupancy;
		KeyValue *hashtable;
		
	public:
		GpuHashTable(int size);

		void reshape(int sizeReshape);
		bool insertBatch(int *keys, int* values, int numKeys);

		int* getBatch(int* key, int numItems);		
		float loadFactor();

		void printHash();
	
		~GpuHashTable();
};

#endif
