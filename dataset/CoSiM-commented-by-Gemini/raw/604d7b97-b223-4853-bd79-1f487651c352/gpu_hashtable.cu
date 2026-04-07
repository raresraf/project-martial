
/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 * @details This file defines a hash table that resides in GPU memory and is manipulated
 * through CUDA kernels. It uses linear probing with wrap-around for collision resolution
 * and atomic operations for thread safety.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

const size_t HASH1 = 10703903591llu;
const size_t HASH2 = 21921594616111llu;


/**
 * @brief Computes a hash for a given integer key on the device.
 * @param data The integer key to hash.
 * @param limit The size of the hash table, used to scale the hash value.
 * @return The computed hash value.
 */
__device__ int hash4(int data, int limit) {
	return ((long long)abs(data) * HASH1) % HASH2 % limit;
}


/**
 * @brief CUDA kernel to copy and rehash elements from an old hash table to a new one.
 * @param old_vec Pointer to the old hash table data.
 * @param new_vec Pointer to the new hash table data.
 * @param old_n The size of the old hash table.
 * @param new_n The size of the new hash table.
 * @details Each thread takes an element from the old table and inserts it into the new table
 * using linear probing with wrap-around.
 */
__global__ void kernel_copyVec(struct hash_pair *old_vec, struct hash_pair *new_vec, int old_n, int new_n) {
	
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	
	if (i < old_n) {
		int newKey = old_vec[i].key;
		int hash = hash4(newKey, new_n);
		int start = hash;
		int end = new_n;

		int flag = 0;	
		// Probe from the initial hash to the end of the new table.
		for (int k = start; k < end; k++) {
			int oldKey = atomicCAS(&new_vec[k].key, KEY_INVALID, newKey);
			if (oldKey == KEY_INVALID) {
				new_vec[k].value = old_vec[i].value;
				flag = 1;
				break;
			}
		}
		// If not inserted, wrap around and probe from the beginning.
		if (flag == 0) {
			for (int k = 0; k < start; k++) {
				int oldKey = atomicCAS(&new_vec[k].key, KEY_INVALID, newKey);
				if (oldKey == KEY_INVALID) {
					new_vec[k].value = old_vec[i].value;
					break;
				}
			}
		}
  	}
}


/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param keys Pointer to an array of keys to insert.
 * @param values Pointer to an array of values to insert.
 * @param numKeys The number of keys to insert.
 * @param hashtable Pointer to the hash table data.
 * @param full_capacity The total capacity of the hash table.
 * @param capacity Pointer to the current number of elements in the hash table.
 * @details Each thread handles one key-value pair, using linear probing with wrap-around
 * and atomicCAS to find an empty slot or update an existing key.
 */
__global__ void kernel_insertVec(int *keys, int *values, int numKeys, struct hash_pair *hashtable, int full_capacity, int *capacity) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < numKeys) {
		int newKey = keys[i];
		int hash = hash4(newKey, full_capacity);
		int start = hash;
		int end = full_capacity;

		int flag = 0;
		// Probe from the initial hash to the end of the table.
		for(int k = start; k < end; k++) {
			int oldKey = atomicCAS(&hashtable[k].key, KEY_INVALID, newKey);

			if (oldKey == KEY_INVALID) {
				
				hashtable[k].value = values[i];
				flag = 1;
				
				atomicAdd(&(*capacity), 1);
				break;
			}
			if (oldKey == newKey) {
				
				hashtable[k].value = values[i];
				flag = 1;
				break;
			}
			
		}
		// If not inserted, wrap around and probe from the beginning.
		if (flag == 0) {
			
			for (int k = 0; k < start; k++) {
				int oldKey = atomicCAS(&hashtable[k].key, KEY_INVALID, newKey);

				if (oldKey == KEY_INVALID) {
					hashtable[k].value = values[i];
					atomicAdd(&(*capacity), 1);
					break;
				}
				if (oldKey == newKey) {
					hashtable[k].value = values[i];
					break;
				}
			}
		}
  	}
}


/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param keys Pointer to an array of keys to look up.
 * @param values Pointer to an array to store the retrieved values.
 * @param numKeys The number of keys to look up.
 * @param hashtable Pointer to the hash table data.
 * @param full_capacity The total capacity of the hash table.
 * @details Each thread searches for a key using linear probing with wrap-around.
 */
__global__ void kernel_getVec(int *keys, int *values, int numKeys, struct hash_pair *hashtable, int full_capacity) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < numKeys) {
		int key = keys[i];
		int hash = hash4(key, full_capacity);
		int start = hash;
		int end = full_capacity;

		int flag = 0;
		// Probe from the initial hash to the end of the table.
		for(int k = start; k < end; k++) {
			if (hashtable[k].key == key) {
				
				values[i] = hashtable[k].value;
				flag = 1;
				break;
			}
		}
		// If not found, wrap around and probe from the beginning.
		if (flag == 0) {
			for (int k = 0; k < start; k++) {
				if (hashtable[k].key == key) {
					values[i] = hashtable[k].value;
					break;
				}
			}
		}
  	}
}



/**
 * @brief Constructs a GpuHashTable object.
 * @param size The initial capacity of the hash table.
 * @details Allocates managed memory for the capacity counter and device memory for the hash table itself.
 */
GpuHashTable::GpuHashTable(int size) {
	cudaError_t err;

	
	err = cudaMallocManaged(&capacity, sizeof(int));
	DIE(err != cudaSuccess, "cudaMallocManaged");

	*capacity = 0;

	full_capacity = size;

	
	err = cudaMalloc(&hashtable, sizeof(struct hash_pair) * full_capacity);
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMemset(hashtable, 0, full_capacity * sizeof(struct hash_pair));
	DIE(err != cudaSuccess, "cudaMemset");
}


/**
 * @brief Destroys the GpuHashTable object.
 * @details Frees the GPU memory allocated for the hash table and its capacity counter.
 */
GpuHashTable::~GpuHashTable() {
	cudaError_t err;

	
	err = cudaFree(hashtable);
	DIE(err != cudaSuccess, "cudaFree hashtable");

	err = cudaFree(capacity);
	DIE(err != cudaSuccess, "cudaFree capacity");
}


/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The new capacity of the hash table.
 * @details Allocates a new hash table on the GPU and launches a kernel to re-hash
 * all elements from the old table into the new one.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t err;
	struct hash_pair *copy_hashtable;

	
	err = cudaMalloc(&copy_hashtable, sizeof(struct hash_pair) * numBucketsReshape);
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMemset(copy_hashtable, 0, numBucketsReshape * sizeof(hash_pair));
	DIE(err != cudaSuccess, "cudaMemset");

	
	int blocks_no = full_capacity / BLOCKSIZE;
	if (full_capacity % BLOCKSIZE)
  		++blocks_no;

  	
	kernel_copyVec>>(hashtable, copy_hashtable, full_capacity, numBucketsReshape);
	err = cudaDeviceSynchronize();
	DIE(err != cudaSuccess, "cudaDeviceSynchronize");

	err = cudaFree(hashtable);
	DIE(err != cudaSuccess, "cudaFree");

	full_capacity = numBucketsReshape;
	hashtable = copy_hashtable;
}


/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys An array of keys to insert.
 * @param values An array of values to insert.
 * @param numKeys The number of keys and values in the batch.
 * @return True if the insertion was successful.
 * @details If the load factor exceeds a threshold, the table is resized. Then, keys and values
 * are copied to the GPU, and the insertion kernel is launched.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	cudaError_t err;
	int *deviceKeys, *deviceValues;

	err = cudaMalloc(&deviceKeys, (size_t) (numKeys * sizeof(int)));
	DIE(err != cudaSuccess, "cudaMalloc");



	err = cudaMalloc(&deviceValues, (size_t) (numKeys * sizeof(int)));
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	err = cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	
	float new_factor = ( (float) (*capacity + numKeys) / full_capacity );
	if (new_factor > 0.9f)
		reshape((int)(((float) *capacity + numKeys) / 0.8f));

	int blocks_no = numKeys / BLOCKSIZE;
	if (numKeys % BLOCKSIZE)
		++blocks_no;

	kernel_insertVec>>(deviceKeys, deviceValues, numKeys, hashtable, full_capacity, capacity);
	err = cudaDeviceSynchronize();
	DIE(err != cudaSuccess, "cudaDeviceSynchronize");



	err = cudaFree(deviceKeys);
	DIE(err != cudaSuccess, "cudaFree");

	err = cudaFree(deviceValues);
	DIE(err != cudaSuccess, "cudaFree");

	return true;
}


/**
 * @brief Retrieves values for a batch of keys.
 * @param keys An array of keys to look up.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to a host array containing the retrieved values. The caller must free this memory.
 * @details This function copies the keys to the GPU, launches a retrieval kernel, and copies
 * the results back to the host.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	cudaError_t err;


	int *deviceKeys, *deviceValues, *hostValues;

	err = cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMalloc(&deviceValues, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");

	hostValues = (int*) calloc(numKeys, sizeof(int));

	err = cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	int blocks_no = numKeys / BLOCKSIZE;
	if (numKeys % BLOCKSIZE)
		++blocks_no;

	kernel_getVec>>(deviceKeys, deviceValues, numKeys, hashtable, full_capacity);
	cudaDeviceSynchronize();

	err = cudaMemcpy(hostValues, deviceValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, "cudaMemcpy");

	err = cudaFree(deviceKeys);
	DIE(err != cudaSuccess, "cudaFree");

	err = cudaFree(deviceValues);
	DIE(err != cudaSuccess, "cudaFree");
	
	return hostValues;
}


/**
 * @brief Calculates the load factor of the hash table.
 * @return The current load factor (ratio of number of elements to total capacity).
 */
float GpuHashTable::loadFactor() {
	float factor = ((float) *capacity / full_capacity);
	return factor;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
