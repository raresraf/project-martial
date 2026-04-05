/**
 * @file gpu_hashtable.cu
 * @brief CUDA implementation of a GPU-resident hash table.
 *
 * This file defines a hash table data structure that is stored and manipulated
 * entirely on the GPU. It uses an open addressing scheme with linear probing for
 * collision resolution. Operations like insertion and retrieval are performed in
 * parallel batches by CUDA kernels.
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

// Defines the maximum load factor before the hash table is resized.
#define LOAD_FACTOR                     0.8f
// Constants for operation success/failure.
#define FAIL                            false
#define SUCCESS                         true


/**
 * @brief Host-side constructor for the GpuHashTable.
 * @param size The initial number of buckets to allocate for the hash table.
 *
 * Allocates memory on the GPU for the hash table buckets and initializes
 * all entries to zero, marking them as empty.
 */
 GpuHashTable::GpuHashTable(int size) {
    limitSize = size;
    currentSize = 0;
    // Allocate device memory for the hash buckets.
    cudaMalloc(&hashTableBuckets, limitSize * sizeof(HashTableEntry));

    if (hashTableBuckets == 0) {
        cerr << "[HOST] Couldn't allocate memory for GpuHashTable!\n";
    }

    // Initialize all bucket keys to KEY_INVALID (0).
    cudaMemset(hashTableBuckets, 0, limitSize * sizeof(HashTableEntry));

}

/**
 * @brief Host-side destructor for the GpuHashTable.
 *
 * Frees the GPU memory allocated for the hash table buckets.
 */
GpuHashTable::~GpuHashTable() {
    cudaFree(hashTableBuckets);
}


/**
 * @brief A device function to compute the hash of an integer key.
 * @param data The input key to hash.
 * @param limit The size of the hash table, used for the modulo operation.
 * @return The computed hash value, an index into the hash table.
 *
 * This function uses a series of bitwise operations and multiplications
 * (a variant of MurmurHash) to create a well-distributed hash value.
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
 * @brief CUDA kernel for inserting a batch of key-value pairs into the hash table.
 * @param keys Pointer to the device array of keys.
 * @param values Pointer to the device array of values.
 * @param numKeys The number of key-value pairs to insert.
 * @param hashTableBuckets Pointer to the hash table buckets on the device.
 * @param limitSize The total number of buckets in the hash table.
 *
 * Each thread in the grid processes one key-value pair. It calculates the hash,
 * then uses linear probing and an atomic Compare-and-Swap (CAS) operation to
 * find an empty bucket or update an existing key. The atomic CAS is critical
 * for ensuring thread safety during concurrent insertions.
 */
__global__ void kernelInsertEntry(int *keys, int *values, int numKeys, HashTableEntry *hashTableBuckets, int limitSize) {

    // Determine the unique ID for the current thread.
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  
    // Boundary check: ensure the thread is not processing out of bounds.
    if (threadId >= numKeys)
        return;
    
    int currentKey = keys[threadId];
    int currentValue = values[threadId];
    int hash = getHash(currentKey, limitSize);
    
    // Begin linear probing loop.
    while(true) {
        // Atomically attempt to write the currentKey into an empty bucket (marked by KEY_INVALID).
        // `atomicCAS` returns the old value that was in the bucket.
        int inplaceKey = atomicCAS(&hashTableBuckets[hash].HashTableEntryKey, KEY_INVALID, currentKey);
        
        // If the old value was KEY_INVALID, this thread successfully claimed the bucket.
        // If the old value was the same as currentKey, we are updating an existing entry.
        if (inplaceKey == currentKey || inplaceKey == KEY_INVALID) {
            hashTableBuckets[hash].HashTableEntryValue = currentValue;
            return;
        }
        // If the bucket was occupied by another key, move to the next bucket.
        hash = (hash + 1) % limitSize;
    }
}


/**
 * @brief Host-side function to insert a batch of key-value pairs.
 * @param keys Pointer to the host array of keys.
 * @param values Pointer to the host array of values.
 * @param numKeys The number of pairs to insert.
 * @return `true` on success, `false` on failure.
 *
 * This function orchestrates the batch insertion process, including checking the
 * load factor, resizing if necessary, transferring data to the GPU, launching
 * the kernel, and cleaning up.
 */
 bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
    float futureLoadFactor = (float) (currentSize + numKeys) / limitSize;
   
    // If adding the new keys exceeds the load factor, resize the table first.
    if (futureLoadFactor >= LOAD_FACTOR) {
        reshape(limitSize + numKeys);
    }

    int *deviceKeys;
    int *deviceValues;

    // Allocate memory on the device for the keys and values.
    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));

    if (deviceValues == 0 || deviceKeys == 0) {
        cerr << "[HOST] Couldn't allocate memory for device keys or values arrays!\n";
        return FAIL;
    }

    // Copy the key and value data from host to device.
    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    // Configure the kernel launch parameters (grid and block size).
    int threadBlockSize = 1024;
    int gridSize = numKeys/ threadBlockSize;
    if (numKeys % threadBlockSize)
        gridSize++;

    // Launch the insertion kernel.
    kernelInsertEntry>>(
            deviceKeys,
            deviceValues,
            numKeys,
            hashTableBuckets,
            limitSize
            );

    // Wait for the kernel to complete and check for errors.
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    currentSize += numKeys;

    // Free the temporary device memory.
    cudaFree(deviceKeys);
    cudaFree(deviceValues);
   
	return SUCCESS;
}

/**
 * @brief CUDA kernel for retrieving a batch of values corresponding to a batch of keys.
 * @param keys Pointer to the device array of keys to look up.
 * @param values Pointer to the device array where the results will be stored.
 * @param numKeys The number of keys to look up.
 * @param limitSize The size of the hash table.
 * @param hashTableBuckets Pointer to the hash table buckets on the device.
 *
 * Each thread processes one key. It uses the same linear probing logic as the
 * insert kernel to find the key. If found, the corresponding value is written
 * to the output array; otherwise, a sentinel value (0) is written.
 */
__global__ void kernelGetEntry( int *keys, int *values, int numKeys, int limitSize, HashTableEntry *hashTableBuckets) {

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= numKeys)
        return;

    int currentKey = keys[threadId];
    int hash = getHash(currentKey, limitSize);
    
    // Linear probing loop to find the key.
    while(true) {
        // If the key is found, write the value to the output array and terminate.
        if (hashTableBuckets[hash].HashTableEntryKey == currentKey) {
            values[threadId] =  hashTableBuckets[hash].HashTableEntryValue;
            return;
        }

        // If an empty bucket is encountered, the key does not exist in the table.
        if (hashTableBuckets[hash].HashTableEntryKey == KEY_INVALID) {
            values[threadId] = 0; // Sentinel for 'not found'.
            return;
        }

        // Move to the next bucket.
        hash = (hash + 1) % limitSize;
    }
}


/**
 * @brief Host-side function to retrieve a batch of values.
 * @param keys Pointer to the host array of keys to look up.
 * @param numKeys The number of keys to retrieve.
 * @return A pointer to a host array containing the corresponding values.
 *         The caller is responsible for freeing this memory. Returns NULL on failure.
 */
 int* GpuHashTable::getBatch(int* keys, int numKeys) {
    int *deviceKeys;
    int *values;
    int *deviceValues;

    // Allocate memory on the device for the input keys and output values.
    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));

    if (deviceKeys == 0 ||  deviceValues == 0) {
        cerr << "[HOST] Couldn't allocate memory for device keys GPU arrays!\n";
        return NULL;
    }

    // Copy keys from host to device.
    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    // Configure and launch the get kernel.
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

    // Wait for the kernel to finish.
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    
    // Allocate host memory for the results.
    values = (int *) calloc(numKeys, sizeof(int));
    if (values == NULL) {
        cerr << "[HOST] Couldn't allocate memory for thre return values array!\n";
        return NULL;
    }

    // Copy the results from device to host.
    cudaMemcpy(values, deviceValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up device memory.
    cudaFree(deviceValues);
    cudaFree(deviceKeys);

    return values;
}

/**
 * @brief CUDA kernel to copy elements from an old hash table to a new, larger one.
 * @param hashTableBucketsOrig Pointer to the original (old) hash table.
 * @param limitSizeOrig The size of the original hash table.
 * @param hashTableBuckets Pointer to the new, resized hash table.
 * @param limitSize The size of the new hash table.
 *
 * This kernel is used for rehashing. Each thread takes an element from the old
 * table and re-inserts it into the new table using the same atomic linear
 * probing logic as the main insertion kernel.
 */
__global__ void kernelCopyTable(
        HashTableEntry *hashTableBucketsOrig,
        int limitSizeOrig,
        HashTableEntry *hashTableBuckets,
        int limitSize) {

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= limitSizeOrig)
        return;

    // Skip empty buckets from the old table.
    if (hashTableBucketsOrig[threadId].HashTableEntryKey == KEY_INVALID)
        return;

    int currentKey = hashTableBucketsOrig[threadId].HashTableEntryKey;
    int currentValue = hashTableBucketsOrig[threadId].HashTableEntryValue;
    // Re-hash the key for the new table size.
    int hash = getHash(currentKey, limitSize);
    
    // Use atomic linear probing to insert into the new table.
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
 * @brief Resizes the hash table to a new capacity and rehashes all existing elements.
 * @param numBucketsReshape The new number of buckets for the resized table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
    cout << "pennis"<< endl; // Note: Unprofessional debug output.
    HashTableEntry *hashTableBucketsReshaped;
    int newLimitSize = numBucketsReshape;

    // Allocate a new, larger table on the device.
    cudaMalloc(&hashTableBucketsReshaped, newLimitSize * sizeof(HashTableEntry));

    if (hashTableBucketsReshaped == 0)
        cerr << "[HOST] Couldn't allocate memory for GpuHashTable Reshape!\n";

    cudaMemset(hashTableBucketsReshaped, 0, newLimitSize * sizeof(HashTableEntry));

    // Configure and launch the kernel to copy/rehash elements.
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

    // Wait for the copy to complete.
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    
    // Free the old table memory.
    cudaFree(hashTableBuckets);
    
    // Point the class members to the new table and update the size.
    hashTableBuckets = hashTableBucketsReshaped;
    limitSize = newLimitSize;
}

/**
 * @brief Calculates and returns the current load factor of the hash table.
 * @return The load factor as a float.
 */
float GpuHashTable::loadFactor() {
    if (currentSize == 0)
        return 0.f;
    else
        return (float) currentSize / limitSize;
}


// Convenience macros for using the hash table in the test file.
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
 >>>> file: gpu_hashtable.hpp
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		                 0

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)
	
const size_t primeList[] =
{
	2llu, 3llu, 5llu, 7llu, 11llu, 13llu, 17llu, 23llu, 29llu, 37llu, 47llu,
	59llu, 73llu, 97llu, 127llu, 151llu, 197llu, 251llu, 313llu, 397llu,
	499llu, 631llu, 797llu, 1009llu, 1259llu, 1597llu, 2011llu, 2539llu,
	3203llu, 4027llu, 5087llu, 6421llu, 8089llu, 10193llu, 12853llu, 16193llu,
	20399llu, 25717llu, 32401llu, 40823llu, 51437llu, 64811llu, 81649llu,
	102877llu, 129607llu, 163307llu, 205759llu, 259229llu, 326617llu,
	411527llu, 518509llu, 653267llu, 823117llu, 1037059llu, 1306601llu,
	1646237llu, 2074129llu, 2613229llu, 3292489llu, 4148279llu, 5226491llu,
	6584983llu, 8296553llu, 10453007llu, 13169977llu, 16593127llu, 20906033llu,
	26339969llu, 33186281llu, 41812097llu, 52679969llu, 66372617llu,
	83624237llu, 105359939llu, 132745199llu, 167248483llu, 210719881llu,
	265490441llu, 334496971llu, 421439783llu, 530980861llu, 668993977llu,
	842879579llu, 1061961721llu, 1337987929llu, 1685759167llu, 2123923447llu,
	2675975881llu, 3371518343llu, 4247846927llu, 5351951779llu, 6743036717llu,
	8495693897llu, 10703903591llu, 13486073473llu, 16991387857llu,
	21407807219llu, 26972146961llu, 33982775741llu, 42815614441llu,
	53944293929llu, 67965551447llu, 85631228929llu, 107888587883llu,
	135931102921llu, 171262457903llu, 215777175787llu, 271862205833llu,
	342524915839llu, 431554351609llu, 543724411781llu, 685049831731llu,
	863108703229llu, 1087448823553llu, 1370099663459llu, 1726217406467llu,
	2174897647073llu, 2740199326961llu, 3452434812973llu, 4349795294267llu,
	5480398654009llu, 6904869625999llu, 8699590588571llu, 10960797308051llu,
	13809739252051llu, 17399181177241llu, 21921594616111llu, 27619478504183llu,
	34798362354533llu, 43843189232363llu, 55238957008387llu, 69596724709081llu,
	87686378464759llu, 110477914016779llu, 139193449418173llu,
	175372756929481llu, 220955828033581llu, 278386898836457llu,
	350745513859007llu, 441911656067171llu, 556773797672909llu,
	701491027718027llu, 883823312134381llu, 1113547595345903llu,
	1402982055436147llu, 1767646624268779llu, 2227095190691797llu,
	2805964110872297llu, 3535293248537579llu, 4454190381383713llu,
	5611928221744609llu, 7070586497075177llu, 8908380762767489llu,
	11223856443489329llu, 14141172994150357llu, 17816761525534927llu,
	22447712886978529llu, 28282345988300791llu, 35633523051069991llu,
	44895425773957261llu, 56564691976601587llu, 71267046102139967llu,
	89790851547914507llu, 113129383953203213llu, 142534092204280003llu,
	179581703095829107llu, 226258767906406483llu, 285068184408560057llu,
	359163406191658253llu, 452517535812813007llu, 570136368817120201llu,
	718326812383316683llu, 905035071625626043llu, 1140272737634240411llu,
	1436653624766633509llu, 1810070143251252131llu, 2280545475268481167llu,
	2873307249533267101llu, 3620140286502504283llu, 4561090950536962147llu,
	5746614499066534157llu, 7240280573005008577llu, 9122181901073924329llu,
	11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu
};




int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}


int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}




struct HashTableEntry {
    int HashTableEntryKey;
    int HashTableEntryValue;
};



class GpuHashTable
{
    private:
        int currentSize;
        int limitSize;
        HashTableEntry *hashTableBuckets;

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

