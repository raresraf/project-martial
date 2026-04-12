
/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table using open addressing with linear probing.
 * @details This file provides a robust implementation of a hash table on the GPU.
 * It uses open addressing with a correct and complete linear probing strategy that
 * wraps around the table to resolve collisions. It correctly uses atomic operations
 * for thread-safe insertions and for performing a parallel resize (re-hash).
 *
 * @warning RACE CONDITION: The `insert` kernel contains a race condition. While it
 * safely claims slots using atomics, the subsequent value update is non-atomic.
 * This can lead to lost updates if multiple threads try to update the value of the
 * same existing key concurrently.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief Computes a hash value for a given key on the GPU.
 * @param data The key to hash.
 * @param limit The capacity of the hash table.
 * @return An integer hash value within the range [0, limit-1].
 */
__device__ int hash_function(int data, int limit) {
	return ((long)abs(data) * 51437llu) % 543724411781llu % limit;
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs.
 * @param hm A struct containing hash table metadata (size, element count).
 * @param device_hashmap Device pointer to the array of key-value cells.
 * @warning The value update (`device_hashmap[index].value = ...`) is not atomic,
 * creating a race condition for updates to the same key.
 */
__global__ void insert(int *keys, int *values, int numKeys, hashmap hm,
			struct cell *device_hashmap) {
	int hash_value, last_key, stop_condition, start_condition, index;
	
	// Determine the unique index for this thread to process.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= numKeys)
		return;

	hash_value = hash_function(keys[i], hm.size);

	// --- Linear Probing with Full Wrap-Around ---
	// This logic correctly probes the entire table by first checking from the
	// initial hash to the end, and then wrapping around to check from the
	// beginning to the initial hash position.
	stop_condition = hm.size;
	start_condition = hash_value;
	for (index = start_condition; index < stop_condition; index++) {
		// Atomically attempt to claim a slot if it is empty (0) or already holds our key.
		last_key = atomicCAS(&device_hashmap[index].key, 0, keys[i]);
		if (last_key == 0 || last_key == keys[i]) {
			// BUG: Non-atomic write. This is a race condition.
			device_hashmap[index].value = values[i];
			if (last_key == keys[i]) {
				// LOGIC BUG: `hm` is passed by value, so this decrement has no
				// effect on the host's count. The intent is also unclear.
				hm.numElem--;
			}
			break;
		}
		
		// If we reach the end of the array, wrap around to the beginning.
		if (index + 1 == hm.size) {
			index = -1; // Will be 0 on next loop increment.
			stop_condition = hash_value; // Stop before reaching the start again.
		}
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 */
__global__ void get(int *keys, int *values, int numKeys, hashmap hm,
			struct cell *device_hashmap) {
	int hash_value, start_condition, stop_condition, index;
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= numKeys)
		return;
	
	hash_value = hash_function(keys[i], hm.size);
	
	// --- Search with Linear Probing and Full Wrap-Around ---
	// This uses the same robust probing strategy as the insert kernel.
	stop_condition = hm.size;
	start_condition = hash_value;
	for (index = start_condition; index < stop_condition; index++) {
		// A non-atomic read is sufficient and correct for a get operation.
		if (keys[i] == device_hashmap[index].key) {
			values[i] = device_hashmap[index].value;
			break;
		}
		
		// Logic to wrap around the table for a full search.
		if (index + 1 == hm.size) {
			index = -1;
			stop_condition = hash_value;
		}
	}	
}

/**
 * @brief CUDA kernel to resize the hash table by re-hashing all elements.
 * @details Each thread is responsible for one element from the old table, which
 * it re-hashes and inserts into the new, larger table using linear probing.
 */
__global__ void reshape_hashmap(int size, int new_size, struct cell *device_hashmap,
				struct cell *device_hashmap_reshaped) {
	int hash_value, start_condition, stop_condition, index, last_key;
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= size)
		return;
	// Skip empty cells.
	if (device_hashmap[i].key == 0)
		return;
	
	// --- Re-insertion into New Table ---
	hash_value = hash_function(device_hashmap[i].key, new_size);
	stop_condition = new_size;
	start_condition = hash_value;
	
	// Perform linear probing to find an empty slot in the new table.
	for (index = start_condition; index < stop_condition; index++) {
		// Atomically claim a slot. This is safe as each thread processes a unique old element.
		last_key = atomicCAS(&device_hashmap_reshaped[index].key, 0,
				device_hashmap[i].key);
		if (last_key == 0 || last_key == device_hashmap[i].key) {
			device_hashmap_reshaped[index].value =
						device_hashmap[i].value;
			break;
		}
		
		if (index + 1 == new_size) {
			index = -1;
			stop_condition = hash_value;
		}
	}	
}

/**
 * @brief Host-side constructor. Allocates memory on the GPU.
 */
GpuHashTable::GpuHashTable(int size) {
	hm.size = size;
	hm.numElem = 0;

	cudaMalloc(&device_hashmap, size * sizeof(struct cell));
	if (device_hashmap == 0) {
		printf("Couldn't allocate memory
");
		return;
	}
	cudaMemset(device_hashmap, 0, size * sizeof(struct cell));
}

/**
 * @brief Host-side destructor. Frees GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(device_hashmap);
}

/**
 * @brief Resizes the hash table to a new, larger capacity.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int last_size = hm.size;
	const size_t block_size = 256;
	size_t blocks_no = (numBucketsReshape % block_size != 0) ?
	                   (numBucketsReshape / block_size + 1) :
	                   (numBucketsReshape / block_size);

	hm.size = numBucketsReshape;

	// 1. Allocate memory for the new table.
	cudaMalloc(&device_hashmap_reshaped, numBucketsReshape * sizeof(struct cell));
	if (device_hashmap_reshaped == 0) {
		printf("Couldn't allocate memory
");
		return;
	}
	cudaMemset(device_hashmap_reshaped, 0, numBucketsReshape * sizeof(struct cell));

	// 2. Launch the re-hashing kernel.
	reshape_hashmap>>(last_size, hm.size,
				device_hashmap, device_hashmap_reshaped);
	cudaDeviceSynchronize();

	// 3. Free the old table's memory and point to the new one.
	cudaFree(device_hashmap);
	device_hashmap = device_hashmap_reshaped;
}

/**
 * @brief Inserts a batch of key-value pairs from the host.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys;
	int *device_values;
	const size_t block_size = 256;
	size_t blocks_no = (numKeys % block_size != 0) ? (numKeys / block_size + 1) : (numKeys / block_size);

	cudaMalloc(&device_keys, numKeys * sizeof(int));
	cudaMalloc(&device_values, numKeys * sizeof(int));

	if (device_keys == 0 || device_values == 0) {
		printf("Couldn't allocate memory
");
		return false; // Changed from NULL to false to match return type.
	}

	// 1. Check load factor and resize if necessary.
	if (((hm.numElem + numKeys) * 1.0) / hm.size >= 0.9)
		reshape((hm.numElem + numKeys) / 0.8);

	// 2. Copy data from host to device.
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// 3. Launch the insertion kernel.
	insert>>(device_keys, device_values, numKeys, hm,
						device_hashmap);
	cudaDeviceSynchronize();

	hm.numElem += numKeys;

	// 4. Free temporary buffers.
	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}

/**
 * @brief Retrieves values for a batch of keys from the host.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_keys;
	int *result_values;
	int *result_values_host;
	const size_t block_size = 256;
	size_t blocks_no = (numKeys % block_size != 0) ? (numKeys / block_size + 1) : (numKeys / block_size);

	// 1. Allocate temporary device buffers and host buffer for results.
	cudaMalloc(&device_keys, numKeys * sizeof(int));
	cudaMalloc(&result_values, numKeys * sizeof(int));
	if (device_keys == 0 || result_values == 0) {
		printf("Couldn't allocate memory
");
		return NULL;
	}

	// 2. Copy keys from host to device.
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// 3. Launch the get kernel.
	get>>(device_keys, result_values, numKeys, hm,
					device_hashmap);
	cudaDeviceSynchronize();

	cudaFree(device_keys);

	// 4. Copy results from device back to host.
	result_values_host = (int *)malloc(numKeys * sizeof(int));
	if (!result_values_host) {
		printf("Couldn't allocate memory
");
		cudaFree(result_values);
		return NULL;
	}	
	cudaMemcpy(result_values_host, result_values, numKeys * sizeof(int),
						cudaMemcpyDeviceToHost);
	cudaFree(result_values);
	
	return result_values_host;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
	return (1.f * hm.numElem) / hm.size;
}


// --- Test harness and boilerplate ---
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
	// ... (rest of prime list)
};

// ... (unused hash functions)

struct hashmap
{
	int size;
	int numElem;
};

struct cell
{
	int key;
	int value;
};

class GpuHashTable
{
	struct hashmap hm;
	struct cell *device_hashmap;
	struct cell *device_hashmap_reshaped;
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
