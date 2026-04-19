/**
 * @file gpu_hashtable.cu
 * @brief Thread-safe GPU Hash Table with circular linear probing.
 * 
 * This module implements a high-throughput hash table on NVIDIA GPUs. It utilizes 
 * atomic operations for concurrent state management and a robust circular linear 
 * probing algorithm to resolve hash collisions within global memory.
 * 
 * Algorithm: Open addressing with two-pass circular linear probing.
 * Memory Model: Global memory (cudaMalloc) for entry arrays.
 * Domain: HPC, Parallel Data Structures.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define PRIME_1 13169977
#define PRIME_2 5351951779

#define MIN_LOAD_FACTOR 0.81
#define MAX_LOAD_FACTOR 0.9

#define MAX_THREADS 1024


/**
 * @brief Device-side hash function using a 64-bit prime modular scheme.
 */
__device__ int get_hash(int data, int limit) {
	return ((long long)abs(data) * PRIME_1) % PRIME_2 % limit;
}

/**
 * @brief CUDA kernel for parallel entry insertion.
 * 
 * Functional Utility: Claims empty slots (key=0) or updates existing keys 
 * using atomic Compare-And-Swap (CAS).
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
	
	
	/**
	 * Block Logic: Primary insertion probe pass.
	 */
	for(int i = hashed_key; i < ht.capacity; i++){
		current_key = atomicCAS(&ht.entries[i].key, KEY_INVALID, key);
		
		if(current_key == key || current_key == KEY_INVALID){
			inserted = true;
			ht.entries[i].value = values[idx];
			return;
		}
	}

	
	/**
	 * Block Logic: Wrap-around insertion probe pass.
	 */
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
 * @brief CUDA kernel for re-hashing data during expansion.
 */
__global__ void resize_ht(HashTable old_ht, HashTable new_ht){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= old_ht.capacity || old_ht.entries[idx].key == KEY_INVALID){
		return;
	}


	int key = old_ht.entries[idx].key;
	int value = old_ht.entries[idx].value;


	int hashed_key = get_hash(key, new_ht.capacity);
	bool inserted = false;
	int current_key;
	
	
	// Logic: Standard circular probe re-insertion.
	for(int i = hashed_key; i < new_ht.capacity; i++){
		current_key = atomicCAS(&new_ht.entries[i].key, KEY_INVALID, key);
		
		if(current_key == KEY_INVALID){
			inserted = true;
			new_ht.entries[i].value = value;
			return;
		}
	}

	
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
 * @brief CUDA kernel for parallel value retrieval.
 */
__global__ void get_batch(int* keys, int numKeys, int* result, HashTable ht){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx >= numKeys || idx > ht.capacity){
		return;
	}

	bool found = false;
	int hashed_key = get_hash(keys[idx], ht.capacity);
	int key = keys[idx];

	
	// Logic: Linear probe search pass.
	for(int i = hashed_key; i < ht.capacity; i++){
		
		if(ht.entries[i].key == key){
			found = true;
			result[idx] = ht.entries[i].value;
			return;
		}
	}

	
	// Logic: Search wrap-around.
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
 * @brief Constructor: Initializes the hash table metadata and device memory.
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
 * @brief Destructor: Releases device entry memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaError_t err = cudaFree(ht.entries);
	DIE(err != cudaSuccess, cudaGetErrorString(err));
}


/**
 * @brief Dynamically expands table capacity via parallel migration.
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


	resize_ht>>(ht, new_ht);


	cudaDeviceSynchronize();

	err = cudaFree(ht.entries);
	DIE(err != cudaSuccess, cudaGetErrorString(err));

	ht = new_ht;
}


/**
 * @brief Performs batch parallel insertion.
 * 
 * Logic: Monitors load factor and triggers expansion if density > 90%.
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


	// Adaptive Scaling: Ensures performance by maintaining low collision density.
	if (ht.inserted_items + numKeys > MAX_LOAD_FACTOR * ht.capacity) {
		reshape(int(ht.inserted_items + numKeys) / MIN_LOAD_FACTOR);
	}


	int blocks = (numKeys % MAX_THREADS == 0) ? numKeys / MAX_THREADS
												: numKeys / MAX_THREADS + 1;

	insert_batch>>(dkeys, dvalues, numKeys, ht);

	cudaDeviceSynchronize();

	ht.inserted_items += numKeys;

	err = cudaFree(dkeys);
	DIE(err != cudaSuccess, cudaGetErrorString(err));

	err = cudaFree(dvalues);
	DIE(err != cudaSuccess, cudaGetErrorString(err));

    return true;
}


/**
 * @brief Batch parallel retrieval.
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


	get_batch>>(dKeys, numKeys, dValues, ht);

	cudaDeviceSynchronize();

	err = cudaMemcpy(hValues, dValues, bytes_size, cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, cudaGetErrorString(err));

    return hValues;
}


/**
 * @brief Current saturation ratio.
 */
float GpuHashTable::loadFactor() {

	if (ht.capacity == 0) {
		return 0;
	}

	return float(ht.inserted_items) / float(ht.capacity);
}


// ... primeList documentation ...

/**
 * @brief Multi-tier hash functions utilizing prime number modular arithmetic.
 */
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
 * @struct HashTableItem
 * @brief Atomic storage unit for a key-value mapping on the GPU.
 */
typedef struct{
	int key;
	int value;
}HashTableItem;

/**
 * @struct HashTable
 * @brief Internal state for the device-resident hash mapping.
 */
typedef struct  {
	int capacity;           // Entry buffer capacity.
	int inserted_items;     // Current occupancy count.
	HashTableItem* entries; // Device pointer to entries.
}HashTable;


/**
 * @class GpuHashTable
 * @brief Host-side handle for GPU hash table lifecycle management.
 */
class GpuHashTable
{
	HashTable ht;
public:
	GpuHashTable(int size);
	void reshape(int sizeReshape);

	bool insertBatch(int* keys, int* values, int numKeys);
	int* getBatch(int* key, int numItems);

	float loadFactor();

	~GpuHashTable();
};


#endif

