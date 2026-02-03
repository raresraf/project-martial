/**
 * @file gpu_hashtable.cu
 * @brief CUDA implementation of a hash table with linear probing.
 *
 * This file provides a GPU-accelerated hash table that uses linear probing for
 * collision resolution. It is designed for batch insertion and retrieval of
 * key-value pairs. The implementation includes CUDA kernels for insertion,
 * retrieval, and rehashing, along with a host-side class to manage the hash
 * table's lifecycle and operations.
 */

#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


#define LOAD_FACTOR                     0.8f
#define FAIL                            false
#define SUCCESS                         true


/**
 * @brief Constructor for the GpuHashTable class.
 * @param size The initial size of the hash table.
 */
 GpuHashTable::GpuHashTable(int size) {
    limitSize = size;
    currentSize = 0;
    cudaMalloc(&hashTableBuckets, limitSize * sizeof(HashTableEntry));

    if (hashTableBuckets == 0) {
        cerr << "[HOST] Couldn't allocate memory for GpuHashTable!\n";
    }

    cudaMemset(hashTableBuckets, 0, limitSize * sizeof(HashTableEntry));

}


/**
 * @brief Destructor for the GpuHashTable class.
 */
GpuHashTable::~GpuHashTable() {
    cudaFree(hashTableBuckets);
}


/**
 * @brief Computes the hash for a given key.
 * @param data The key to hash.
 * @param limit The size of the hash table.
 * @return The computed hash value.
 *
 * This function uses a series of bitwise operations and multiplications to
 * generate a hash value, which is then mapped to the hash table size.
 */
__device__ int getHash(int data, int limit) {
    
    data ^= data >> 16;
    data *= 0x85ebca6b;
    data ^= data >> 13;
    data *= 0xc2b2ae35;
    data ^= data >> 16;
    return data % limit;

}


/**
 * @brief CUDA kernel to insert a batch of key-value pairs.
 * @param keys A device pointer to the array of keys.
 * @param values A device pointer to the array of values.
 * @param numKeys The number of pairs to insert.
 * @param hashTableBuckets A device pointer to the hash table buckets.
 * @param limitSize The size of the hash table.
 *
 * This kernel uses linear probing to resolve collisions. Each thread handles
 * one key-value pair and uses atomic compare-and-swap (atomicCAS) to find an
 * empty slot or update an existing key.
 */
__global__ void kernelInsertEntry(int *keys, int *values, int numKeys, HashTableEntry *hashTableBuckets, int limitSize) {

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  
    // Pre-condition: Ensure the thread index is within the bounds of the input arrays.
    if (threadId >= numKeys)
        return;
    
    int currentKey = keys[threadId];
    int currentValue = values[threadId];
    int hash = getHash(currentKey, limitSize);
    
    // Invariant: The loop continues until an empty slot is found or the key is
    // updated.
    while(true) {


        int inplaceKey = atomicCAS(&hashTableBuckets[hash].HashTableEntryKey, KEY_INVALID, currentKey);
        if (inplaceKey == currentKey || inplaceKey == KEY_INVALID) {
            hashTableBuckets[hash].HashTableEntryValue = currentValue;
            return;
        }
        hash = (hash + 1) % limitSize;
    }
}


/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys A host pointer to the array of keys.
 * @param values A host pointer to the array of values.
 * @param numKeys The number of pairs to insert.
 * @return True on success, false on failure.
 */
 bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
    int futureLoadFactor = (float) (currentSize + numKeys) / limitSize;
   
    // Pre-condition: If inserting the new keys would exceed the load factor,
    // the hash table is resized.
    if (futureLoadFactor >= LOAD_FACTOR) {
        reshape(limitSize + numKeys);
    }

    int *deviceKeys;
    int *deviceValues;

    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));

    if (deviceValues == 0 || deviceKeys == 0) {
        cerr << "[HOST] Couldn't allocate memory for device keys or values arrays!\n";
        return FAIL;
    }

    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    int threadBlockSize = 1024;
    int gridSize = numKeys/ threadBlockSize;
    if (numKeys % threadBlockSize)
        gridSize++;

    kernelInsertEntry>>(
            deviceKeys,
            deviceValues,
            numKeys,
            hashTableBuckets,
            limitSize
            );

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    currentSize += numKeys;

    cudaFree(deviceKeys);
    cudaFree(deviceValues);
   
	return SUCCESS;
}


/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param keys A device pointer to the array of keys.
 * @param values A device pointer to the array where values will be stored.
 * @param numKeys The number of keys to look up.
 * @param limitSize The size of the hash table.
 * @param hashTableBuckets A device pointer to the hash table buckets.
 *
 * This kernel performs a parallel lookup using linear probing to find the keys.
 */
__global__ void kernelGetEntry( int *keys, int *values, int numKeys, int limitSize, HashTableEntry *hashTableBuckets) {

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= numKeys)
        return;

    int currentKey = keys[threadId];
    int hash = getHash(currentKey, limitSize);
    
    // Invariant: The loop continues until the key is found or an empty slot is
    // encountered, which indicates the key is not in the table.
    while(true) {
        if (hashTableBuckets[hash].HashTableEntryKey == currentKey) {
            values[threadId] =  hashTableBuckets[hash].HashTableEntryValue;
            return;
        }

        if (hashTableBuckets[hash].HashTableEntryKey == KEY_INVALID) {
            values[threadId] = 0;
            return;
        }

        hash = (hash + 1) % limitSize;
    }
}


/**
 * @brief Retrieves values for a batch of keys.
 * @param keys A host pointer to the array of keys.
 * @param numKeys The number of keys to look up.
 * @return A host pointer to an array of the retrieved values.
 */
 int* GpuHashTable::getBatch(int* keys, int numKeys) {
    int *deviceKeys;
    int *values;
    int *deviceValues;

    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));

    if (deviceKeys == 0 ||  deviceValues == 0) {
        cerr << "[HOST] Couldn't allocate memory for device keys GPU arrays!\n";
        return NULL;
    }

    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    int threadBlockSize = 1024;
    int gridSize = numKeys/ threadBlockSize;
    if (numKeys % threadBlockSize)
        gridSize++;

    kernelGetEntry>>(
            deviceKeys,
            deviceValues,
            numKeys,
            limitSize,
            hashTableBuckets
            );

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
        
    values = (int *) calloc(numKeys, sizeof(int));
    if (values == NULL) {
        cerr << "[HOST] Couldn't allocate memory for thre return values array!\n";
        return NULL;
    }

    cudaMemcpy(values, deviceValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceValues);
    cudaFree(deviceKeys);

    return values;
}


/**
 * @brief CUDA kernel to copy entries from an old to a new hash table.
 * @param hashTableBucketsOrig The old hash table buckets.
 * @param limitSizeOrig The size of the old hash table.
 * @param hashTableBuckets The new hash table buckets.
 * @param limitSize The size of the new hash table.
 *
 * This kernel is used during rehashing. Each thread is responsible for one
 * entry from the old table, which it re-inserts into the new table.
 */
__global__ void kernelCopyTable(
        HashTableEntry *hashTableBucketsOrig,
        int limitSizeOrig,
        HashTableEntry *hashTableBuckets,
        int limitSize) {

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= limitSizeOrig)
        return;

    if (hashTableBucketsOrig[threadId].HashTableEntryKey == KEY_INVALID)
        return;

    int currentKey = hashTableBucketsOrig[threadId].HashTableEntryKey;
    int currentValue = hashTableBucketsOrig[threadId].HashTableEntryValue;
    int hash = getHash(currentKey, limitSize);
    
    // Invariant: This loop uses linear probing and atomicCAS to insert the entry
    // into the new hash table.
    while (true) {
        int inplaceKey = atomicCAS(&hashTableBuckets[hash].HashTableEntryKey, KEY_INVALID, currentKey);
        if (inplaceKey == currentKey || inplaceKey == KEY_INVALID) {
            hashTableBuckets[hash].HashTableEntryValue = currentValue;
            return;
        }
        hash = (hash + 1) % limitSize;
    }
}


/**
 * @brief Resizes and rehashes the hash table.
 * @param numBucketsReshape The new size of the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
    cout << "pennis"<< endl;
    HashTableEntry *hashTableBucketsReshaped;
    int newLimitSize = numBucketsReshape;

    cudaMalloc(&hashTableBucketsReshaped, newLimitSize * sizeof(HashTableEntry));

    if (hashTableBucketsReshaped == 0)
        cerr << "[HOST] Couldn't allocate memory for GpuHashTable Reshape!\n";

    cudaMemset(hashTableBucketsReshaped, 0, newLimitSize * sizeof(HashTableEntry));

    int threadBlockSize = 1024;
    int gridSize = limitSize / threadBlockSize;
    if (limitSize % threadBlockSize)
        gridSize++;
    
    kernelCopyTable>>(
            hashTableBuckets,
            limitSize,
            hashTableBucketsReshaped,
            newLimitSize
            );

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    
    cudaFree(hashTableBuckets);
    
    hashTableBuckets = hashTableBucketsReshaped;
    limitSize = newLimitSize;
}


float GpuHashTable::loadFactor() {
    if (currentSize == 0)
        return 0.f;
    else
        return (float) currentSize / limitSize;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"