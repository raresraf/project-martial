/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-based hash table using CUDA.
 *
 * This file contains the CUDA device code and host-side wrapper class for a
 * hash table that operates on the GPU. It supports batch insertion and retrieval
 * of key-value pairs and includes dynamic resizing to manage load factor.
 * The hashing strategy uses linear probing for collision resolution.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_hashtable.hpp"

// Prime numbers for hashing to reduce collisions.
#define PRIME_1 13169977
#define PRIME_2 5351951779

// Load factor thresholds for resizing the hash table.
#define MIN_LOAD_FACTOR 0.81
#define MAX_LOAD_FACTOR 0.9

#define MAX_THREADS 1024

/**
 * @brief Computes a hash value for a given key.
 * @param data The input key to be hashed.
 * @param limit The capacity of the hash table, used for the modulo operation.
 * @return The calculated hash value.
 *
 * This is a device function, intended to be called from within CUDA kernels.
 */
__device__ int get_hash(int data, int limit) {
	return ((long long)abs(data) * PRIME_1) % PRIME_2 % limit;
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param keys Device pointer to an array of keys.
 * @param values Device pointer to an array of values.
 * @param numKeys The number of keys to insert.
 * @param ht The hash table structure.
 *
 * Each thread handles one key-value pair. It uses linear probing and atomic
 * compare-and-swap (atomicCAS) to handle collisions in a thread-safe manner.
 */
__global__ void insert_batch(int* keys, int* values, int numKeys, HashTable ht){
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numKeys) {
		return;
	}

	int key = keys[idx];
	int hashed_key = get_hash(keys[idx], ht.capacity);
	bool inserted = false;
	int current_key;
	
	// Linear probing from the hashed key to the end of the table.
	for(int i = hashed_key; i < ht.capacity; i++){
		// Atomically insert the key if the slot is empty (KEY_INVALID).
		current_key = atomicCAS(&ht.entries[i].key, KEY_INVALID, key);
		
		if(current_key == key || current_key == KEY_INVALID){
			inserted = true;
			ht.entries[i].value = values[idx];
			return;
		}
	}

	// If not inserted, wrap around and probe from the beginning of the table.
	if (!inserted) {
		for (int i = 0; i < hashed_key; i++) {
			current_key = atomicCAS(&ht.entries[i].key, KEY_INVALID, key);
			
			if (current_key == key || current_key == KEY_INVALID) {
				ht.entries[i].value = values[idx];
				return;
			}
		}
	}
	
}

/**
 * @brief CUDA kernel to resize the hash table and rehash all existing elements.
 * @param old_ht The old hash table.
 * @param new_ht The new, larger hash table.
 *
 * Each thread is responsible for one entry from the old table. It reads the
 * key-value pair and re-inserts it into the new table using linear probing.
 */
__global__ void resize_ht(HashTable old_ht, HashTable new_ht){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Ignore empty slots in the old table.
	if (idx >= old_ht.capacity || old_ht.entries[idx].key == KEY_INVALID){
		return;
	}

	int key = old_ht.entries[idx].key;
	int value = old_ht.entries[idx].value;

	// Re-hash the key for the new table capacity.
	int hashed_key = get_hash(key, new_ht.capacity);
	bool inserted = false;
	int current_key;
	
	// Linear probing to find an empty slot in the new table.
	for(int i = hashed_key; i < new_ht.capacity; i++){
		current_key = atomicCAS(&new_ht.entries[i].key, KEY_INVALID, key);
		
		if(current_key == KEY_INVALID){
			inserted = true;
			new_ht.entries[i].value = value;
			return;
		}
	}

	// Wrap-around probing if the end of the table is reached.
	if (!inserted) {
		for (int i = 0; i < hashed_key; i++) {
			current_key = atomicCAS(&new_ht.entries[i].key, KEY_INVALID, key);
			
			if(current_key == KEY_INVALID){
				new_ht.entries[i].value = value;
				return;
			}
		}
	}
}

/**
 * @brief CUDA kernel to retrieve a batch of values corresponding to a batch of keys.
 * @param keys Device pointer to an array of keys to look up.
 * @param numKeys The number of keys to look up.
 * @param result Device pointer to store the retrieved values.
 * @param ht The hash table structure.
 *
 * Each thread searches for one key. It uses linear probing, starting from the
 * key's hash, to find the matching key.
 */
__global__ void get_batch(int* keys, int numKeys, int* result, HashTable ht){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx >= numKeys || idx > ht.capacity){
		return;
	}

	bool found = false;
	int hashed_key = get_hash(keys[idx], ht.capacity);
	int key = keys[idx];

	// Linear probing from the hashed key to find the target key.
	for(int i = hashed_key; i < ht.capacity; i++){
		if(ht.entries[i].key == key){
			found = true;
			result[idx] = ht.entries[i].value;
			return;
		}
	}

	// Wrap-around probing if not found in the first scan.
	if (!found) {
		for (int i = 0; i < hashed_key; i++) {
			if(ht.entries[i].key == key){
				result[idx] = ht.entries[i].value;
				return;
			}
		}
	}
}

/**
 * @brief Constructor for the GpuHashTable class.
 * @param size The initial capacity of the hash table.
 *
 * Allocates GPU memory for the hash table entries and initializes them.
 */
GpuHashTable::GpuHashTable(int size) {
	ht.capacity = size;
	ht.inserted_items = 0;
	
	cudaError_t err = cudaMalloc(&ht.entries, size * sizeof(HashTableItem));
	DIE(err != cudaSuccess, cudaGetErrorString(err));

	err = cudaMemset(ht.entries, 0, size * sizeof(HashTableItem));
	DIE(err != cudaSuccess, cudaGetErrorString(err));
}

/**
 * @brief Destructor for the GpuHashTable class.
 *
 * Frees the GPU memory allocated for the hash table entries.
 */
GpuHashTable::~GpuHashTable() {
	cudaError_t err = cudaFree(ht.entries);
	DIE(err != cudaSuccess, cudaGetErrorString(err));
}

/**
 * @brief Resizes the hash table to a new capacity.
 * @param new_capacity The new capacity for the hash table.
 *
 * This function allocates a new, larger hash table on the GPU, launches a
 * kernel to rehash and insert all elements from the old table into the new
 * one, and then frees the old table.
 */
void GpuHashTable::reshape(int new_capacity) {
	HashTable new_ht;
	new_ht.capacity = new_capacity;
	new_ht.inserted_items = ht.inserted_items;
	
	cudaError_t err = cudaMalloc(&new_ht.entries, new_ht.capacity * sizeof(HashTableItem));
	DIE(err != cudaSuccess, cudaGetErrorString(err));
	
	err = cudaMemset(new_ht.entries, 0, new_ht.capacity * sizeof(HashTableItem));
	DIE(err != cudaSuccess, cudaGetErrorString(err));

	int blocks = (new_ht.capacity % MAX_THREADS == 0) ? new_ht.capacity / MAX_THREADS
		: new_ht.capacity / MAX_THREADS + 1;

	// Launch the resize kernel.
	resize_ht<<<blocks, MAX_THREADS>>>(ht, new_ht);

	cudaDeviceSynchronize();

	err = cudaFree(ht.entries);
	DIE(err != cudaSuccess, cudaGetErrorString(err));

	ht = new_ht;
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys Pointer to a host array of keys.
 * @param values Pointer to a host array of values.
 * @param numKeys The number of pairs to insert.
 * @return True on success.
 *
 * This method handles memory transfers between host and device, checks if a
 * resize is needed based on the load factor, and launches the insertion kernel.
 */
bool GpuHashTable::insertBatch(int* keys, int* values, int numKeys) {
	
	int *dkeys;
	int *dvalues;
	int bytes_size = numKeys * sizeof(int);

	cudaError_t err = cudaMalloc(&dkeys, bytes_size);
	DIE(err != cudaSuccess, cudaGetErrorString(err));

	err = cudaMalloc(&dvalues, bytes_size);
	DIE(err != cudaSuccess, cudaGetErrorString(err));


	err = cudaMemcpy(dkeys, keys, bytes_size, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, cudaGetErrorString(err));

	err = cudaMemcpy(dvalues, values, bytes_size, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, cudaGetErrorString(err));

	// Resize if the new keys would exceed the maximum load factor.
	if (ht.inserted_items + numKeys > MAX_LOAD_FACTOR * ht.capacity) {
		reshape(int(ht.inserted_items + numKeys) / MIN_LOAD_FACTOR);
	}

	// Calculate the number of blocks needed for the kernel launch.
	int blocks = (numKeys % MAX_THREADS == 0) ? numKeys / MAX_THREADS
												: numKeys / MAX_THREADS + 1;

	insert_batch<<<blocks, MAX_THREADS>>>(dkeys, dvalues, numKeys, ht);

	cudaDeviceSynchronize();

	ht.inserted_items += numKeys;

	err = cudaFree(dkeys);
	DIE(err != cudaSuccess, cudaGetErrorString(err));

	err = cudaFree(dvalues);
	DIE(err != cudaSuccess, cudaGetErrorString(err));

    return true;
}

/**
 * @brief Retrieves a batch of values for a given set of keys.
 * @param keys Pointer to a host array of keys.
 * @param numKeys The number of keys to retrieve.
 * @return A pointer to a host array containing the retrieved values. The caller
 *         is responsible for freeing this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {

	if (keys == NULL || numKeys == 0) {
		return NULL;
	}

	int* dValues;
	int* dKeys;
	int bytes_size = sizeof(int) * numKeys;
	int* hValues;

	cudaError_t err = cudaMalloc(&dValues, bytes_size);
	DIE(err != cudaSuccess, cudaGetErrorString(err));

	err = cudaMalloc(&dKeys, bytes_size);
	DIE(err != cudaSuccess, cudaGetErrorString(err));

	hValues = (int*) malloc(bytes_size);
	DIE(hValues == NULL, "malloc");

	err = cudaMemcpy(dKeys, keys, bytes_size, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, cudaGetErrorString(err));

	int blocks = (numKeys % MAX_THREADS == 0) ? numKeys / MAX_THREADS
												: numKeys / MAX_THREADS + 1;

	get_batch<<<blocks, MAX_THREADS>>>(dKeys, numKeys, dValues, ht);
	
	cudaDeviceSynchronize();

	err = cudaMemcpy(hValues, dValues, bytes_size, cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, cudaGetErrorString(err));

    return hValues;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float.
 */
float GpuHashTable::loadFactor() {
	
	if (ht.capacity == 0) {
		return 0;
	}

	return float(ht.inserted_items) / float(ht.capacity);
}