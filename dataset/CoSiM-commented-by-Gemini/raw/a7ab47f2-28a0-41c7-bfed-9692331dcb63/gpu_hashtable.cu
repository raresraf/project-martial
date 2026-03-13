/**
 * @file gpu_hashtable.cu
 * @brief A CUDA implementation of a GPU-based hash table.
 *
 * @details
 * This file provides a parallel hash table implementation using CUDA. It uses
 * open addressing with linear probing for collision resolution. Concurrency during
 * insertions is managed using atomic Compare-and-Swap (CAS) operations. The
 * hash table can be dynamically resized when its load factor exceeds a threshold.
 *
 * The main components are:
 * - `GpuHashTable` class: The host-side interface for managing the GPU hash table.
 * - `insert_hash`, `get_hash`, `reshape_hashT`: CUDA kernels for batch operations.
 * - `getHash`: A device-side integer hash function.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "gpu_hashtable.hpp" // Assumed to contain struct/class definitions

/**
 * @brief Computes the hash for an integer key.
 * @param x The input key.
 * @param N The size of the hash table, used for the modulo operation.
 * @return The computed hash value (bucket index).
 *
 * @note This is a device function, meant to be called from CUDA kernels.
 * It uses a series of bitwise operations and multiplications (MurmurHash-like)
 * to generate a hash value.
 */
__device__ long long getHash(int x, int N) {
    if (x < 0) x = -x;
    x = ((x >> 16) ^ x) * HASH_NO;
    x = ((x >> 16) ^ x) * HASH_NO;
    x = (x >> 16) ^ x;
    x = x % N;
    return x;
}

/**
 * @brief CUDA kernel to rehash and transfer all entries from an old table to a new one.
 * @param h1 The old hash table.
 * @param siz1 The size of the old hash table.
 * @param h2 The new (and typically larger) hash table.
 * @param siz2 The size of the new hash table.
 *
 * @details
 * Each thread is responsible for moving one valid entry from the old table `h1`
 * to the new table `h2`. It uses linear probing and `atomicCAS` to find and claim
 * an empty slot in the new table.
 */
__global__ void reshape_hashT(hashTable h1, long siz1, hashTable h2, long siz2) {
    int vall, idx = blockIdx.x * blockDim.x + threadIdx.x, key1, key2;
    
    // Grid-stride loop pattern is not used; assumes grid is large enough.
    // Each thread processes one element at `idx`.
	if ((h1.pairs[idx].key == KEY_INVALID) || (siz1 <= idx))
		return;

	key2 = h1.pairs[idx].key;
	vall = getHash(key2, h2.size); // Calculate hash in the new table.

    // --- Linear Probing with Atomic Insertion ---
    // First, probe from the initial hash value to the end of the table.
	for (int i = vall; i < siz2; i++) {
        // Atomically attempt to claim an empty slot (KEY_INVALID).
		key1 = atomicCAS(&h2.pairs[i].key, KEY_INVALID, key2);
		if (key1 == KEY_INVALID) {
			h2.pairs[i].value = h1.pairs[idx].value;
			return; // Successfully inserted.
		}
	}
    // Second, if no slot was found, wrap around and probe from the beginning to the hash value.
	for (int i = 0; i < vall; i++) {
		key1 = atomicCAS(&h2.pairs[i].key, KEY_INVALID, key2);
		if (key1 == KEY_INVALID) {
			h2.pairs[i].value = h1.pairs[idx].value;
			return; // Successfully inserted.
		}
	}
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param k The number of keys to insert.
 * @param keys Device pointer to an array of keys.
 * @param values Device pointer to an array of values.
 * @param h The hash table.
 * @param siz The size of the hash table.
 *
 * @details
 * Each thread handles one key-value pair. It calculates the hash and uses linear
 * probing with `atomicCAS` to find an empty slot or update an existing key.
 */
__global__ void insert_hash(int k, int *keys, int *values, hashTable h, long siz) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x, key, vall;

	if (k <= idx) return; // Guard against out-of-bounds access.

	vall = getHash(keys[idx], siz);

    // --- Linear Probing with Atomic Insertion/Update ---
	for (int i = vall; i < siz; i++) {
		key = atomicCAS(&h.pairs[i].key, KEY_INVALID, keys[idx]);
        // If the slot was empty OR already contained our key, we can write the value.
		if (key == KEY_INVALID || key == keys[idx]) {
			atomicExch(&h.pairs[i].value, values[idx]); // Use atomicExch for thread-safe update.
			return;
		}
	}
    // Wrap-around probe.
	for (int i = 0; i < vall; i++) {
		key = atomicCAS(&h.pairs[i].key, KEY_INVALID, keys[idx]);
		if (key == KEY_INVALID || key == keys[idx]) {
			atomicExch(&h.pairs[i].value, values[idx]);
			return;
		}
	}
}

/**
 * @brief CUDA kernel to retrieve a batch of values corresponding to given keys.
 * @param k The number of keys to retrieve.
 * @param keys Device pointer to an array of keys to look up.
 * @param values Device pointer to an array where the found values will be written.
 * @param h The hash table.
 * @param siz The size of the hash table.
 *
 * @details
 * Each thread handles one key lookup. It uses linear probing to find the key.
 * Since this is a read-only operation on the keys, no atomics are needed for the lookup itself.
 */
__global__ void get_hash(int k, int *keys, int *values, hashTable h, long siz) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x, vall;

	if (k <= idx) return;

	vall = getHash(keys[idx], siz);

    // --- Linear Probing Search ---
	for (int i = vall; i < siz; i++) {
		if (h.pairs[i].key == keys[idx]) {
			values[idx] = h.pairs[i].value;
			return;
		}
	}
    // Wrap-around search.
	for (int i = 0; i < vall; i++) {
		if (h.pairs[i].key == keys[idx]) {
			values[idx] = h.pairs[i].value;
			return;
		}
	}
}

/**
 * @brief Host-side constructor for the GpuHashTable.
 */
GpuHashTable::GpuHashTable(int size) {
	hashT.size = size;
	cntPairs = 0;
	hashT.pairs = nullptr;
	// Allocate memory on the GPU for the hash table array.
	cudaMalloc(&hashT.pairs, size * sizeof(pair));
	// Initialize all keys to KEY_INVALID (0).
	cudaMemset(hashT.pairs, 0, size * sizeof(pair));
}

/**
 * @brief Host-side destructor.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashT.pairs);
}

/**
 * @brief Resizes the hash table on the GPU.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	// Calculate grid size for the reshape kernel launch.
	int k = (hashT.size + THREADS_NO - 1) / THREADS_NO;

	hashTable newH;
	newH.size = numBucketsReshape;

	// Allocate memory for the new, larger table.
	cudaMalloc(&newH.pairs, numBucketsReshape * sizeof(pair));
	cudaMemset(newH.pairs, 0, numBucketsReshape * sizeof(pair));

	// Launch the kernel to rehash and move data to the new table.
	reshape_hashT<<>>(hashT, hashT.size, newH, newH.size);

	cudaDeviceSynchronize(); // Wait for the reshape to complete.
	cudaFree(hashT.pairs); // Free the old table memory.
	hashT = newH; // Replace the old table with the new one.
}

/**
 * @brief Inserts a batch of key-value pairs from the host.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *aKeys, *aValues;
    // Calculate grid size for the insert kernel.
	int k = (numKeys + THREADS_NO - 1) / THREADS_NO;
	long nr = cntPairs + numKeys;

    // Check if the load factor will exceed the maximum threshold.
	if (nr / (float)hashT.size >= LOADFACTOR_MAX) {
		reshape((int)(nr / LOADFACTOR_MIN)); // Resize if necessary.
    }

	// Allocate temporary device memory for the input keys and values.
	cudaMalloc(&aKeys, numKeys * sizeof(int));
	cudaMalloc(&aValues, numKeys * sizeof(int));

	// Copy keys and values from host to device.
	cudaMemcpy(aKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(aValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Launch the insertion kernel.
	insert_hash<<>>(numKeys, aKeys, aValues, hashT, hashT.size);

	cudaDeviceSynchronize(); // Wait for insertions to complete.
	cudaFree(aKeys);
	cudaFree(aValues);
	cntPairs += numKeys;
	return true;
}

/**
 * @brief Retrieves a batch of values for given keys.
 * @return A host-accessible pointer to an array of values.
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
	int *aKeys, *valls;
	// Calculate grid size for the get kernel.
	int k = (numKeys + THREADS_NO - 1) / THREADS_NO;

	cudaMalloc(&aKeys, numKeys * sizeof(int));
	// Use managed memory for the results so the host can access it directly after synchronization.
	cudaMallocManaged(&valls, numKeys * sizeof(int));

	cudaMemcpy(aKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Launch the get kernel.
	get_hash<<>>(numKeys, aKeys, valls, hashT, hashT.size);
	cudaDeviceSynchronize(); // Wait for the lookup to complete.
	cudaFree(aKeys);
	return valls;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
	if (hashT.size == 0) return 0;
	return (float(cntPairs) / hashT.size);
}
