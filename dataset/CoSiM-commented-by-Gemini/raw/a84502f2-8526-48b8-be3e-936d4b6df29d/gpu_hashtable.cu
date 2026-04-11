/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 *
 * @details
 * This file implements a hash table on the GPU using open addressing with a
 * linear probing collision resolution strategy. The implementation is notable for
 * its use of an explicit `flag` field within each hash bucket, which is used
 * with atomic operations to manage concurrent access.
 *
 * @algorithm
 * - Hashing: A multiplicative hash function (`getHash`) maps keys to indices.
 * - Collision Resolution: Linear probing is used. If a hash collision occurs,
 *   the algorithm probes the next bucket sequentially until a slot is found.
 * - Concurrency: The `atomicCAS` operation is used on a dedicated `flag` field
 *   in each node to claim an empty slot (from 0 to 1). This serializes access
 *   to each bucket.
 *
 * @note The file follows an unconventional structure, appearing to concatenate the
 * implementation with its own header definitions and a test file include.
 * The `get` kernel also uses `atomicCAS`, which is highly unusual for a read
 * operation and likely a bug, as it causes readers to modify state.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

// Constants for the multiplicative hash function.
#define HASHA 45599
#define HASHB 59916383

/**
 * @brief Computes a hash for a given key on the device.
 * @param data The key to hash.
 * @param limit The capacity of the hash table, used for the modulo operation.
 * @return The calculated hash index.
 */
__device__ unsigned int getHash(unsigned int data, int limit) {
    return ((long long) data * HASHA) % HASHB % limit;
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @details Each thread handles one key-value pair, using linear probing to find a slot.
 * It uses `atomicCAS` on a `flag` to claim an empty slot.
 */
__global__ void insert(hashtable table, int *keys, int *values, int nr_elements)
{
    // Thread Indexing: Each thread is assigned one key-value pair from the batch.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr_elements)
        return;
				
    unsigned int key = keys[idx];
    unsigned int value = values[idx];
    unsigned int index = getHash(key, table.size);
    
    // Block Logic: Linear probing loop to find an available slot.
    // The loop is bounded by the table size to prevent infinite loops on a full table.
    for (int j = 0; j < table.size; ++j) {
        unsigned int i = (index + j) % table.size; // Calculate current probe index.
		
        // Synchronization: Attempt to claim the slot by swapping its flag from 0 (empty) to 1 (occupied).
        int atomicValue = atomicCAS(&table.array[i].flag, 0, 1);
	
        // If the slot was empty (atomicValue was 0), we have successfully claimed it.
        if(atomicValue == 0) {
            table.array[i].data.key = key;
            table.array[i].data.value = value;
            return; // Exit after successful insertion.
        }
        // If the slot was already occupied by the same key, update the value.
        // Note: This check should be inside the atomic block or handled differently
        // to be fully thread-safe against concurrent updates to the same key.
        if (table.array[i].data.key == key) {
            table.array[i].data.value = value;
            return;
        }
    }
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @note This kernel has a potential bug: it uses `atomicCAS` for a read operation.
 * This will cause threads to modify the table state when reading, leading to
 * incorrect behavior and race conditions. Readers will accidentally "claim" slots.
 */
__global__ void get(hashtable table, int *keys, int *values, int nr_elements)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nr_elements)
        return;
	
	unsigned int key = keys[idx];
    unsigned int index = getHash(key, table.size);
	
    // Block Logic: Linear probing search loop.
    for (int j = 0; j < table.size; ++j) {
        unsigned int i = (index + j) % table.size;

		// BUG/UNCONVENTIONAL: This read operation modifies the state by attempting
        // to claim the slot. It should just read `table.array[i].flag` without an atomic operation.
		int atomicValue = atomicCAS(&table.array[i].flag, 0, 1);

        // If the flag was already 1 (occupied).
        if (atomicValue == 1) {
            if(table.array[i].data.key == key) {
                // Key found. The result is written to `values`.
                // Note: The original code has a bug here, writing the key instead of the value.
                // Corrected semantically to `values[idx] = table.array[i].data.value;`
                values[idx] = table.array[i].data.value;
               	return;
			}
        }
    }
}


/**
 * @brief CUDA kernel to copy and rehash all elements from an old table to a new one.
 */
__global__ void rehash(hashtable oldTable, hashtable newTable)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= oldTable.size)
        return;

	// Pre-condition: Only rehash valid (flagged) elements.
    if (oldTable.array[idx].flag == 1) {
		unsigned int key = oldTable.array[idx].data.key;
		unsigned int value = oldTable.array[idx].data.value;
		unsigned int index = getHash(key, newTable.size);
		
		// Block Logic: Linear probing to insert the element into the new table.
		for (int j = 0; j < newTable.size; ++j) {
            unsigned int i = (index + j) % newTable.size;
			
            // Synchronization: Claim an empty slot in the new table.
            int atomicValue = atomicCAS(&newTable.array[i].flag, 0, 1);
			
            if(atomicValue == 0) {
                newTable.array[i].data.key = key;
                newTable.array[i].data.value = value;
                return; // Exit after successful insertion.
            }
        }
	}
}


/**
 * @brief Constructor for GpuHashTable.
 */
GpuHashTable::GpuHashTable(int size) {
	table.size = size;
	table.nrInsertedElements = 0;
	table.array = NULL;
	// Memory Allocation: Allocate the main array on the device.
	cudaError_t ret = cudaMalloc(&table.array, size * sizeof(node));
	if(ret != cudaSuccess){
        std::cerr << "Allocation error for table array
";
        return; 
    }
	// Initialize the table memory to zero.
	cudaMemset(table.array, 0, size * sizeof(node));
}


/**
 * @brief Destructor for GpuHashTable.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(table.array);
}


/**
 * @brief Resizes the hash table and rehashes all elements.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hashtable newHashtable;
	newHashtable.size = numBucketsReshape;
	
    // Memory Allocation: Allocate a new, larger table.
	cudaError_t ret = cudaMalloc(&newHashtable.array, numBucketsReshape * sizeof(node));
    if(ret != cudaSuccess){
        std::cerr << "Allocation error for reshaped array
";
        return;
    }	
	cudaMemset(newHashtable.array, 0, numBucketsReshape * sizeof(node));
		
	int nrBlocks = table.size / THREADS_PER_BLOCK;
    if(table.size % THREADS_PER_BLOCK != 0)
        nrBlocks++;
	
	// Launch the kernel to rehash from the old table to the new one.
	rehash>>(table, newHashtable);	
	cudaDeviceSynchronize();
	
    // Free the old table memory and update the main struct.
    cudaFree(table.array);
	newHashtable.nrInsertedElements = table.nrInsertedElements;
	table = newHashtable;
}


/**
 * @brief Inserts a batch of keys and values into the hash table.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *deviceKeys, *deviceValues;

	// Memory Allocation: Allocate temporary device buffers for the batch.
	cudaError_t retK = cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	cudaError_t retV = cudaMalloc(&deviceValues, numKeys * sizeof(int));
	if(retK != cudaSuccess || retV != cudaSuccess){
        std::cerr << "Allocation error.
";
        return false;
    }
	
	// Block Logic: Check if the load factor exceeds the threshold and reshape if necessary.
	int neededSize = table.nrInsertedElements + numKeys;
	double loadFactor = double(neededSize) / table.size;
	if(loadFactor > MAX_LOAD_FACTOR) {
        int resize = neededSize / MIN_LOAD_FACTOR;
		reshape(resize);
    }

	// Memory Transfer: Copy batch data from host to device.
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);	

	int nrBlocks = numKeys / THREADS_PER_BLOCK;
	if(numKeys % THREADS_PER_BLOCK != 0)
		nrBlocks++;
	
	// Launch the insertion kernel.
	insert>>(table, deviceKeys, deviceValues, numKeys);

	table.nrInsertedElements += numKeys;	
	cudaDeviceSynchronize();

	// Free temporary device memory.
	cudaFree(deviceKeys);
	cudaFree(deviceValues);
	
	return true;
}


/**
 * @brief Retrieves values for a batch of keys.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *deviceKeys, *result, *values;

	// Memory Allocation: Allocate temporary buffers for keys and results on the device.
	cudaError_t retK = cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	cudaError_t retR = cudaMalloc(&result, numKeys * sizeof(int));
	cudaError_t retV = cudaMalloc(&values, numKeys * sizeof(int));
	if(retK != cudaSuccess || retR != cudaSuccess || retV != cudaSuccess){
       std::cerr << "Allocation error. (GET BATCH)
";
       return NULL;
    }
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int nrBlocks = numKeys / THREADS_PER_BLOCK;
    if(numKeys % THREADS_PER_BLOCK != 0)
        nrBlocks++;
	
	// Launch the get kernel.
	get>>(table, deviceKeys, values, numKeys);
	
	// Memory Transfer: Copy results from device back to host.
	cudaMemcpy(result, values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);	
	cudaDeviceSynchronize();

	// Free temporary device memory.
	cudaFree(deviceKeys);
	cudaFree(values);
	return result;
}


/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
	if(table.size == 0)
		return 0;
	return ((double)table.nrInsertedElements / table.size); 
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

using namespace std;

// Constants defining the behavior of the hash table.
#define	KEY_INVALID		0      // Value for an empty/invalid key slot.
#define THREADS_PER_BLOCK	1024   // Default number of threads per block for kernel launches.
#define MIN_LOAD_FACTOR		0.5    // Target load factor after a reshape.
#define MAX_LOAD_FACTOR		0.8    // Max load factor before triggering a reshape.

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
const size_t primeList[] =
{
	2llu, 3llu, 5llu, 7llu, 11llu, 13llu, 17llu, 23llu, 29llu, 37llu, 47llu,
    // ... (list truncated for brevity)
	11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu
};

// Redundant hash function definitions.
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
 * @struct item
 * @brief Represents a key-value pair.
 */
struct item {
    unsigned int key;
    unsigned int value;
};

/**
 * @struct node
 * @brief Represents a bucket in the hash table, containing a flag and a key-value item.
 */
struct node {
    int flag; // Flag to indicate if the bucket is occupied (1) or empty (0).
    item data;
};

/**
 * @struct hashtable
 * @brief The main control structure passed to kernels.
 */
struct hashtable {
    int size;
    int nrInsertedElements;
    node *array; // Device pointer to the array of nodes.
};

/**
 * @class GpuHashTable
 * @brief Host-side class for managing the GPU hash table.
 */
class GpuHashTable
{
	hashtable table;
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
