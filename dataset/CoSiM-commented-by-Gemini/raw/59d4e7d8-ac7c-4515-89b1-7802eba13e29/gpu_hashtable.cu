

/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 * @details This file defines a hash table that resides in GPU memory and is manipulated
 * through CUDA kernels. It supports batch insertion and retrieval operations. The collision
 * resolution strategy is a form of linear probing with wrap-around, and thread safety
 * is ensured using atomic operations. The hash table can also be dynamically resized.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief Computes a hash for a given integer key on the device.
 * @param data The integer key to hash.
 * @param limit The size of the hash table, used to scale the hash value.
 * @return The computed hash value.
 */
__device__ int myhashKernel(int data, int limit) {
	return ((long long)abs(data) * PRIMENO) % PRIMENO2 % limit;
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param keys Pointer to an array of keys to insert.
 * @param values Pointer to an array of values to insert.
 * @param limit The number of keys to insert.
 * @param myTable The hash table structure.
 * @details Each thread handles one key-value pair. It computes the hash of the key and
 * uses atomic compare-and-swap (atomicCAS) to find an empty slot, implementing a
 * linear probing strategy with wrap-around for collision resolution.
 */
__global__ void insert_kerFunc(int *keys, int *values, int limit, hashtable myTable) {
	int newIndex;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int old, i, j;
	int low, high;

	if (index < limit) {
		newIndex = myhashKernel(keys[index], myTable.size);
		low = newIndex;
		high = myTable.size;
		// Probes in two stages: from initial hash to end, then from start to initial hash.
		for(j = 0; j < 2; j++) {
			for (i = low; i < high; i++) {
				// Atomically try to insert the key if the slot is empty.
				old = atomicCAS(&myTable.list[i].key, 0, keys[index]);
				if (old == keys[index] || old == 0) {
					myTable.list[i].value = values[index];
					return;
				}
			}
			low = 0;
			high = newIndex;
		}

	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param myTable The hash table structure.
 * @param keys Pointer to an array of keys to look up.
 * @param values Pointer to an array to store the retrieved values.
 * @param limit The number of keys to look up.
 * @details Each thread handles one key. It computes the hash and uses linear probing
 * with wrap-around to find the key in the table. If found, the corresponding value
 * is written to the result array.
 */
__global__ void get_kerFunc(hashtable myTable, int *keys, int *values, int limit) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j;
	int newIndex;
	int low, high;

	if (index < limit) {
		newIndex = myhashKernel(keys[index], myTable.size);
		low = newIndex;
		high = myTable.size;
		// Probes in two stages: from initial hash to end, then from start to initial hash.
		for(j = 0; j < 2; j++) {
			for (i = low; i < high; i++) { 
				if (myTable.list[i].key == keys[index]) {
					values[index] = myTable.list[i].value;
					return;
				}
			}
			low = 0;
			high = newIndex;	
		}
	}
}

/**
 * @brief CUDA kernel to reshape the hash table by re-hashing all elements.
 * @param oldTable The old hash table.
 * @param newTable The new, larger hash table.
 * @param size The size of the new hash table.
 * @details Each thread takes an element from the old table and inserts it into the new table
 * using the same probing strategy as `insert_kerFunc`.
 */
__global__ void reshapeKerFunc(hashtable oldTable, hashtable newTable, int size) {
	int newIndex, i, j;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int low, high;

	if (index >= newTable.size)
		return;

	newIndex = myhashKernel(oldTable.list[index].key, size);
	low = newIndex;
	high = size;
	// Probes in two stages to re-insert the element.
	for(j = 0; j < 2; j++) {
		for (i = low; i < high; i++) {
			if (atomicCAS(&newTable.list[i].key, 0,
				oldTable.list[index].key) == 0) {
				newTable.list[i].value = oldTable.list[index].value;
				return;
			}
		}
		low = 0;
		high = newIndex;
	}
}


/**
 * @brief Constructs a GpuHashTable object.
 * @param size The initial size of the hash table.
 * @details Allocates memory on the GPU for the hash table and initializes it to zero.
 */
GpuHashTable::GpuHashTable(int size) {
	myTable.size = size;
	myTable.slotsTaken = 0;

	DIE(cudaMalloc(&myTable.list, size * sizeof(node)) != cudaSuccess, "BOO COULDNT ALLOC MYTABLE LIST INIT\n"); 
	cudaMemset(myTable.list, 0, size * sizeof(node));
}


/**
 * @brief Destroys the GpuHashTable object.
 * @details Frees the GPU memory allocated for the hash table.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(myTable.list);
}


/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The new size of the hash table.
 * @details This function allocates a new, larger hash table on the GPU and launches a kernel
 * to re-hash all the elements from the old table into the new one.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hashtable newTable;
	int noBlocks;
	int slots = myTable.slotsTaken;
	newTable.size = numBucketsReshape;
	newTable.slotsTaken = myTable.slotsTaken;

	DIE(cudaMalloc(&newTable.list, numBucketsReshape * sizeof(node)) != cudaSuccess, "BOO COULDNT ALLOC MYTABLE LIST INIT\n"); 	
	cudaMemset(newTable.list, 0, numBucketsReshape * sizeof(node));



	noBlocks = (myTable.size % BLOCK_SIZE != 0) ?
		(myTable.size / BLOCK_SIZE + 1) : (myTable.size / BLOCK_SIZE);
	reshapeKerFunc>>(myTable, newTable, numBucketsReshape);
	
	cudaDeviceSynchronize();
	cudaFree(myTable.list);
	myTable = newTable;
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
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *device_keys, *device_values;
	int noBlocks;
	myTable.slotsTaken += numKeys;

	size_t memSize = numKeys * sizeof(int);
	DIE(cudaMalloc(&device_keys, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE KEYS\n");
	DIE(cudaMalloc(&device_values, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE VALUES\n");

	if ((float(myTable.slotsTaken) / myTable.size) >= 0.95)
		reshape(int((myTable.slotsTaken) / 0.8));

	cudaMemcpy(device_keys, keys, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, memSize, cudaMemcpyHostToDevice);



	noBlocks = (numKeys % BLOCK_SIZE != 0) ?
		(numKeys / BLOCK_SIZE + 1) : (numKeys / BLOCK_SIZE);
	insert_kerFunc>>(device_keys, device_values, numKeys, myTable);

	cudaDeviceSynchronize();

	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}


/**
 * @brief Retrieves values for a batch of keys.
 * @param keys An array of keys to look up.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to an array in GPU memory containing the retrieved values.
 * @details This function copies the keys to the GPU and launches a kernel to retrieve the values.
 * The result is a pointer to GPU memory.
 */
 int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_values;
    int *device_keys;
    int noBlocks;

	DIE(cudaMallocManaged(&device_values, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE VALUES\n");
	DIE(cudaMalloc(&device_keys, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE KEYS\n");

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	


	noBlocks = (numKeys % BLOCK_SIZE != 0) ?
		(numKeys / BLOCK_SIZE + 1) : (numKeys / BLOCK_SIZE);

	get_kerFunc >> (myTable, device_keys, device_values, numKeys);
	
	cudaDeviceSynchronize();
	
	cudaFree(device_keys);
	return device_values;
}


/**
 * @brief Calculates the load factor of the hash table.
 * @return The load factor (ratio of occupied slots to total size).
 */
 float GpuHashTable::loadFactor() {
	return (myTable.size > 0) ? float(myTable.slotsTaken) / myTable.size : 0;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
