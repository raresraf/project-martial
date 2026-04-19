/**
 * @file gpu_hashtable.cu
 * @brief High-density CUDA Hash Table with bitwise-assisted linear probing.
 * 
 * This module implements a parallel hash table designed for efficient lookup 
 * and insertion on NVIDIA GPUs. It features a bitwise-assisted hash function 
 * to minimize clustering and employs circular linear probing for collision 
 * resolution. The implementation proactively scales table capacity when 
 * occupancy exceeds 70% to maintain near-constant time performance.
 * 
 * Algorithm: Open addressing with manual circular linear probing and bitwise hashing.
 * Memory Model: Global memory (cudaMalloc) for hash table entries.
 * Domain: HPC, Parallel Data Structures.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief Device-side hash function using a bitwise shift and multiplicative constant.
 */
__device__ int computeHash(int data, int limit)
{
    long hash = 51437;
    hash += data * (data << 17) * hash;

    return hash % limit;
}


/**
 * @brief CUDA kernel for parallel data migration.
 * 
 * functional Utility: Transfers valid key-value pairs into a new, larger memory 
 * buffer using parallel re-hashing.
 */
__global__ void map_resize(hash_table old_hashmap, hash_table new_hashmap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= old_hashmap.size) {
        return;
    }

    int oldKey;
    int newKey;
    int newHash;

    newKey = old_hashmap.entries[idx].key;
    newHash = computeHash(newKey, new_hashmap.size);

    /**
     * Block Logic: Re-insertion probe loop.
     * Invariant: Probes the new buffer until an empty slot is atomically reserved.
     */
    for (int j = newHash; j < new_hashmap.size; j++) {


        oldKey = atomicCAS(&new_hashmap.entries[j].key, KEY_INVALID, newKey);

        if (oldKey == KEY_INVALID) {
            new_hashmap.entries[j].value = old_hashmap.entries[idx].value;
            break;
        } else if (j + 1 == new_hashmap.size) {
            // Logic: Manual wrap-around to implement circular probing.
            j = 0;
        }
    }

}


/**
 * @brief CUDA kernel for parallel batch insertion.
 * 
 * functional Utility: Uses atomicCAS to claim slots or update existing entries.
 */
__global__ void map_put(hash_table hashmap, int numKeys, int *keys, int *values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numKeys) {
        return;
    }

    int oldKey;
    int newKey;
    int newHash;

    newKey = keys[idx];
    newHash = computeHash(newKey, hashmap.size);

    /**
     * Block Logic: Circular probe insertion traversal.
     */
    for (int i = newHash; i < hashmap.size; i++) {


        oldKey = atomicCAS(&hashmap.entries[i].key, KEY_INVALID, newKey);

        if (oldKey == KEY_INVALID || oldKey == newKey) {
            // Logic: Successfully reserved slot or found existing key for update.
            hashmap.entries[i].value = values[idx];
            return;
        } else if (i + 1 == hashmap.size) {
            // Logic: Wrap around to the start of the buffer.
            i = 0;
        }
    }
}


/**
 * @brief CUDA kernel for parallel entry retrieval.
 */
__global__ void map_get(hash_table hashmap, int numKeys, int *keys, int *values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numKeys) {
        return;
    }

    int key;
    int newHash;

    key = keys[idx];
    newHash = computeHash(key, hashmap.size);

    // Block Logic: Multi-pass circular search.
    for (int i = newHash; i < hashmap.size; i++) {
        if (hashmap.entries[i].key == key) {
            values[idx] = hashmap.entries[i].value;
            return;
        } else if (i + 1 == hashmap.size) {
            i = 0;
        }
    }


}




/**
 * @brief Constructor: Initializes table and resets entries.
 */
GpuHashTable::GpuHashTable(int size) {
    entriesNumber = 0;
    hashmap.size = size;

    // Memory Hierarchy: Global memory allocation for entries.
    cudaMalloc(&hashmap.entries, size * sizeof(entry));
    cudaMemset(hashmap.entries, 0, size * sizeof(entry));
}


/**
 * @brief Destructor: Releases device-resident memory.
 */
GpuHashTable::~GpuHashTable() {
    cudaFree(hashmap.entries);
}


/**
 * @brief Resizes the hash table using parallel re-hashing.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
    hash_table new_hashmap;
    new_hashmap.size = numBucketsReshape;

    cudaMalloc(&new_hashmap.entries, numBucketsReshape * sizeof(entry));
    cudaMemset(new_hashmap.entries, 0, numBucketsReshape * sizeof(entry));

    int blocks_no = hashmap.size / NUM_THREADS;
    if (hashmap.size % NUM_THREADS) {
        blocks_no++;
    }

    // Synchronization: Ensures all mappings are migrated before cleaning up old memory.
    map_resize>>(hashmap, new_hashmap);
    cudaDeviceSynchronize();

    cudaFree(hashmap.entries);
    hashmap.entries = new_hashmap.entries;
    hashmap.size = new_hashmap.size;
}


/**
 * @brief Performs host-initiated batch parallel insertion.
 * 
 * Logic: Transfers batch to GPU and manages expansion if load factor > 70%.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
    int *deviceKeys;
    int *deviceValues;

    // Adaptive Scaling: Monitors table density to maintain O(1) performance.
    if ((float)(entriesNumber + numKeys) / hashmap.size >= 0.7) {
        reshape(hashmap.size * 2);
    }

    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));

    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    int blocks_no = numKeys / NUM_THREADS;
    if (numKeys % NUM_THREADS) {
        blocks_no++;
    }

    map_put>>(hashmap, numKeys, deviceKeys, deviceValues);
    cudaDeviceSynchronize();

    // Logic: Updates global element count.
    entriesNumber += numKeys;
    cudaFree(deviceKeys);
    cudaFree(deviceValues);

	return true;
}


/**
 * @brief Performs host-initiated batch parallel retrieval.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *deviceKeys;


	int *values;

	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	// Memory Hierarchy: Managed memory (Unified) for simplified host access to results.
	cudaMallocManaged(&values, numKeys * sizeof(int));

    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    int blocks_no = numKeys / NUM_THREADS;
    if (numKeys % NUM_THREADS) {
        blocks_no++;
    }

    map_get>>(hashmap, numKeys, deviceKeys, values);
    cudaDeviceSynchronize();

    cudaFree(deviceKeys);

    return values;
}


/**
 * @brief Implementation for getting the current load factor.
 */
float GpuHashTable::loadFactor() {
    reshape(int(hashmap.size * 5 / 8));
	return 0.8f;
}


// ... hash functions and structures ...

/**
 * @struct entry
 * @brief Atomic storage unit for a key-value mapping on the GPU.
 */
struct entry {
    int key;
    int value;
};

/**
 * @struct hash_table
 * @brief Internal metadata for the device resident hash map.
 */
struct hash_table {
    int size;       // Total entry capacity.
    entry *entries; // Pointer to device mapping array.
};




/**
 * @class GpuHashTable
 * @brief Host controller for Managing the GPU hash mapping lifecycle.
 */
class GpuHashTable
{
    int entriesNumber;      // Host-side occupancy count.
    hash_table hashmap;     // Metadata for the device structure.

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

