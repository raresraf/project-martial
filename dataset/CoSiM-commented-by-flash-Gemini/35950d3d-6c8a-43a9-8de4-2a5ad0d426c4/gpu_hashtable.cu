/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA for parallel operations.
 *
 * This file provides a high-performance hash table designed to run on NVIDIA GPUs.
 * It supports fundamental hash table operations such as reshaping (resizing and rehashing),
 * batch insertion of key-value pairs, and batch retrieval of values for given keys.
 * The implementation leverages CUDA kernels, global memory for the hash table data,
 * and atomic operations for thread-safe collision resolution (linear probing).
 * It is optimized for parallel execution on many-core architectures.
 */

#include <iostream>   // @brief Standard input/output stream library for debugging and general I/O.
#include <string>     // @brief Standard string manipulation functions.
#include <sstream>    // @brief String stream manipulation for data formatting.
#include <vector>     // @brief Dynamic array (vector) container.
#include <algorithm>  // @brief Standard algorithms like std::min/max.
#include <cmath>      // @brief Mathematical functions, e.g., for ceil.
#include <cerrno>     // @brief Defines integer variable `errno` for error indications.

#include "gpu_hashtable.hpp" // @brief Custom header for HashMapData structure and GpuHashTable class definition.


// Global constants and utility macros defined outside the main class implementation.
// They are likely included from gpu_hashtable.hpp or implicitly defined.

#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID	0 // @brief Defines the value representing an invalid or empty key in the hash table.

#define BLOCK_SIZE 1024 // @brief Defines the number of threads per block for CUDA kernel launches.

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0) // @brief A utility macro for error checking and abnormal program termination.




/**
 * @brief Auxiliary hash function 1.
 * @param data The input integer data to hash.
 * @param limit The upper bound for the hash value (size of the hash table).
 * @return An integer hash value within the range [0, limit-1].
 */
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
/**
 * @brief Auxiliary hash function 2.
 * @param data The input integer data to hash.
 * @param limit The upper bound for the hash value (size of the hash table).
 * @return An integer hash value within the range [0, limit-1].
 */
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
/**
 * @brief Auxiliary hash function 3.
 * @param data The input integer data to hash.
 * @param limit The upper bound for the hash value (size of the hash table).
 * @return An integer hash value within the range [0, limit-1].
 */
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
} 
{
  	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x; // @brief Calculates a unique global index for the current CUDA thread.

	// Block Logic: Process elements from the old hash table.
	// Pre-condition: `index` must be within the bounds of the `oldData` array.
	// Invariant: Each thread processes one element from `oldData`.
	if(index < oldSize)
	{
		int currentKey = oldData[index].key; // @brief Key of the current element from the old hash table.
		int currentValue = oldData[index].value; // @brief Value of the current element from the old hash table.

		// Block Logic: Skip invalid (empty) key-value pairs.
		// Invariant: Only valid entries from `oldData` are rehashed.
		if(currentKey == KEY_INVALID)
			return;

		// Block Logic: Calculate the initial hash for the current key in the new hash table.
		// Invariant: `keyHash` will be a valid index within `newData`'s bounds.
		int keyHash = myHash(currentKey, newSize);

		// Block Logic: Perform linear probing to find an empty slot or an existing key in the new hash table.
		// Invariant: The loop continues until the key is successfully inserted or found.
		while(1)
		{
			// Functional Utility: Atomically compare and swap (CAS) `KEY_INVALID` with `currentKey`.
			// This attempts to claim an empty slot without race conditions.
			int oldKey = atomicCAS(&(newData[keyHash].key), KEY_INVALID, currentKey);
			if(oldKey == KEY_INVALID) // Block Logic: If the slot was empty, the key is successfully placed.
			{
				newData[keyHash].value = currentValue; // Functional Utility: Assign the value to the newly claimed slot.
				break;
			}

			// Block Logic: If the slot was occupied, move to the next slot (linear probing).
			// Invariant: `keyHash` wraps around `newSize` to stay within table boundaries.
			keyHash = (keyHash + 1) % newSize;
		}
	}
}




__global__ void hashMapInsertKernel(HashMapData *data, int maxSize, int *keys, int *values, int numKeys, int *addedKeys)
{
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x; // @brief Calculates a unique global index for the current CUDA thread.

	// Block Logic: Process keys for insertion.
	// Pre-condition: `index` must be within the bounds of the `keys` and `values` arrays.
	// Invariant: Each thread processes one key-value pair for insertion.
	if(index < numKeys)
	{
		int currentKey = keys[index]; // @brief Key to be inserted.
		int currentValue = values[index]; // @brief Value to be associated with the key.
		int keyHash = myHash(currentKey, maxSize); // @brief Initial hash for the current key in the hash table.

		// Block Logic: Perform linear probing to find an empty slot or an existing key.
		// Invariant: The loop continues until the key is successfully inserted or an existing entry is updated.
		while(1)
		{
			// Functional Utility: Atomically compare and swap (CAS) `KEY_INVALID` with `currentKey`.
			// This attempts to claim an empty slot or check if the slot is already occupied by `currentKey`.
			int oldKey = atomicCAS(&(data[keyHash].key), KEY_INVALID, currentKey);
			if(oldKey == KEY_INVALID || oldKey == currentKey) // Block Logic: If the slot was empty or already contains `currentKey`.
			{
				if(oldKey == KEY_INVALID) // Block Logic: If the key was newly added.
				{
					// Functional Utility: Atomically increment the count of newly added keys.
					atomicAdd(addedKeys, 1);
				}

				data[keyHash].value = currentValue; // Functional Utility: Assign or update the value in the hash table.
				break;
			}

			// Block Logic: If the slot was occupied by a different key, move to the next slot (linear probing).
			// Invariant: `keyHash` wraps around `maxSize` to stay within table boundaries.
			keyHash = (keyHash + 1) % maxSize;
		}
	}
}



__global__ void hashMapGetBatchKernel(HashMapData *data, int maxSize, int *keys, int *values, int numKeys)
{
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x; // @brief Calculates a unique global index for the current CUDA thread.

	// Block Logic: Process keys for batch retrieval.
	// Pre-condition: `index` must be within the bounds of the `keys` array.
	// Invariant: Each thread processes one key for lookup.
	if(index < numKeys)
	{
		int currentKey = keys[index]; // @brief Key to be looked up.
		int keyHash = myHash(currentKey, maxSize); // @brief Initial hash for the current key in the hash table.

		// Block Logic: Perform linear probing to find the key or an empty slot.
		// Invariant: The loop continues until the key is found, an empty slot is encountered, or the table is fully traversed.
		while(1)
		{
			// Block Logic: Check if the current slot contains the target key.
			// Functional Utility: If found, store the corresponding value and break the loop.
			if(data[keyHash].key == currentKey)
			{
				values[index] = data[keyHash].value; // Functional Utility: Store the found value in the output array.
				break;
			}

			// Block Logic: If an invalid (empty) key is encountered, the key is not in the table.
			// Functional Utility: Store the value associated with `KEY_INVALID` (usually 0) and break.
			if(data[keyHash].key == KEY_INVALID)
			{
				values[index] = data[keyHash].value; // Functional Utility: Store the default value for a missing key.
				break;
			}
			
			// Block Logic: If the slot contains a different key, move to the next slot (linear probing).
			// Invariant: `keyHash` wraps around `maxSize` to stay within table boundaries.
			keyHash = (keyHash + 1) % maxSize;
		}
		
	}
}



GpuHashTable::GpuHashTable(int size) 
{
	this->currentSize = 0; // Functional Utility: Initialize the current number of elements to zero.
	this->maxSize = size; // Functional Utility: Set the maximum capacity of the hash table.

	// Block Logic: Allocate global GPU memory for the hash table data.
	// Pre-condition: `size` is a positive integer.
	// Invariant: `this->data` points to a newly allocated block of GPU memory for `size` HashMapData entries.
	cudaMalloc(&(this->data), sizeof(HashMapData) * size);
	DIE(this->data == NULL, "cudaMalloc failed in constructor"); // Functional Utility: Error check for CUDA memory allocation.

	// Functional Utility: Initialize the allocated GPU memory to zeros, effectively setting all keys to `KEY_INVALID`.
	cudaMemset(this->data, 0, sizeof(HashMapData) * size);
}


GpuHashTable::~GpuHashTable() 
{
	// Functional Utility: Free the global GPU memory allocated for the hash table data.
	// This prevents memory leaks on the device when a GpuHashTable object is destroyed.
	cudaFree(this->data);
}


void GpuHashTable::reshape(int numBucketsReshape) 
{
	// Functional Utility: Store the old hash table data pointer and size for rehashing.
	HashMapData *oldData = this->data;
	int oldSize = this->maxSize;

	// Block Logic: Allocate new GPU memory for the reshaped hash table.
	// Invariant: `newData` points to a newly allocated block of GPU memory for `numBucketsReshape` HashMapData entries.
	HashMapData *newData = NULL;
	cudaMalloc((void **) &newData, numBucketsReshape * sizeof(HashMapData));
	DIE(newData == NULL, "cudaMalloc failed in reshape"); // Functional Utility: Error check for CUDA memory allocation.

	// Functional Utility: Initialize the new GPU memory to zeros, marking all new entries as invalid.
	cudaMemset(newData, 0, sizeof(HashMapData) * numBucketsReshape);

	// Functional Utility: Update the GpuHashTable object's members to point to the new hash table.
	this->data = newData; 
	this->maxSize = numBucketsReshape;

	// Block Logic: Calculate the number of blocks needed for the reshape kernel launch.
	// Invariant: `numberOfBlocks` ensures all elements of `oldData` are processed by the kernel.
	size_t numberOfBlocks = oldSize / BLOCK_SIZE;
	if (oldSize % BLOCK_SIZE) 
	{
		numberOfBlocks = numberOfBlocks + 1;
	}

	// Functional Utility: Launch the CUDA kernel to rehash elements from the old table to the new one.
	hashmapReshapeKernel<<<numberOfBlocks, BLOCK_SIZE>>>(newData, numBucketsReshape, oldData, oldSize);
	cudaDeviceSynchronize(); // Functional Utility: Synchronize the device to ensure the kernel completes before freeing `oldData`.

	// Functional Utility: Free the old GPU memory after all elements have been rehashed.
	cudaFree(oldData);
}

bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) 
{
	int *deviceKeys = NULL; // @brief Pointer to GPU memory for batch keys.
	int *deviceValues = NULL; // @brief Pointer to GPU memory for batch values.

	// Block Logic: Allocate GPU memory for input keys and values.
	cudaMalloc((void **) &deviceKeys, numKeys * sizeof(int));
	DIE(deviceKeys == NULL, "cudaMalloc failed in insertBatch 1"); // Functional Utility: Error check.

	cudaMalloc((void **) &deviceValues, numKeys * sizeof(int));
	DIE(deviceValues == NULL, "cudaMalloc failed in insertBatch 2"); // Functional Utility: Error check.

	// Functional Utility: Copy keys and values from host memory to device memory.
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Block Logic: Check if resizing is necessary based on the load factor.
	// Invariant: If the estimated new load factor exceeds 0.8, the hash table is reshaped.
	float newLoadFactor = (float)(this->currentSize + numKeys) / this->maxSize;
	if(newLoadFactor >= 0.8)
	{
		// Block Logic: Calculate new size and reshape the hash table.
		// Invariant: The new size is chosen to keep the load factor below a desired threshold (0.85).
		int newSize = ceil((100.0 / 85.0) * (this->currentSize + numKeys));
		this->reshape(newSize);
	}

	// Block Logic: Calculate the number of blocks needed for the insert kernel launch.
	size_t numberOfBlocks = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE) 
	{
		numberOfBlocks = numberOfBlocks + 1;
	}

	// Block Logic: Allocate GPU memory for `newKeys` counter and initialize to zero.
	// `newKeys` will store the count of truly new key insertions.
	int *newKeys;
	cudaMalloc((void **) &newKeys, sizeof(int));
	cudaMemset(newKeys, 0, sizeof(int));
	
	// Block Logic: Allocate host memory for `hostNewKeys` to retrieve the count of new keys from device.
	int *hostNewKeys = (int *)malloc(sizeof(int));
	// Functional Utility: Copy initial value (0) of `newKeys` from device to host (redundant here, as it was memset to 0).
	cudaMemcpy(hostNewKeys, newKeys, sizeof(int), cudaMemcpyDeviceToHost);

	// Functional Utility: Launch the CUDA kernel to insert the batch of key-value pairs.
	hashMapInsertKernel<<<numberOfBlocks, BLOCK_SIZE>>>(this->data, this->maxSize, deviceKeys, deviceValues, numKeys, newKeys);
	cudaDeviceSynchronize(); // Functional Utility: Synchronize the device to ensure the kernel completes before reading `newKeys`.

	// Functional Utility: Copy the final count of new keys from device to host.
	cudaMemcpy(hostNewKeys, newKeys, sizeof(int), cudaMemcpyDeviceToHost);
	// Functional Utility: Update the total `currentSize` of the hash table.
	this->currentSize = this->currentSize + *hostNewKeys;

	// Functional Utility: Free allocated GPU and host memory.
	cudaFree(deviceKeys);
	cudaFree(deviceValues);
	cudaFree(newKeys);
	free(hostNewKeys);

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) 
{
	// Block Logic: Declare host and device pointers for keys and values.
	int *deviceKeys; // @brief Pointer to GPU memory for batch lookup keys.
	int *deviceValues; // @brief Pointer to GPU memory to store retrieved batch values.

	// Block Logic: Allocate GPU memory for input keys.
	cudaMalloc((void **) &deviceKeys, numKeys * sizeof(int));
	DIE(deviceKeys == NULL, "cudaMalloc failed in getBatch 1"); // Functional Utility: Error check.

	// Block Logic: Allocate GPU memory for output values.
	cudaMalloc((void **) &deviceValues, numKeys * sizeof(int));
	DIE(deviceValues == NULL, "cudaMalloc failed in getBatch 2"); // Functional Utility: Error check.

	// Functional Utility: Copy lookup keys from host memory to device memory.
	cudaMemcpy(deviceKeys, keys, sizeof(int) * numKeys, cudaMemcpyHostToDevice);

	// Block Logic: Allocate host memory for retrieved values.
	int *hostValues = (int *)malloc(sizeof(int) * numKeys);
	DIE(hostValues == NULL, "malloc failed in getBatch"); // Functional Utility: Error check.

	// Block Logic: Calculate the number of blocks needed for the get batch kernel launch.
	size_t numberOfBlocks = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE) 
	{
		numberOfBlocks = numberOfBlocks + 1;
	}

	// Functional Utility: Launch the CUDA kernel to retrieve the batch of values.
	hashMapGetBatchKernel<<<numberOfBlocks, BLOCK_SIZE>>>(this->data, this->maxSize, deviceKeys, deviceValues, numKeys);
	cudaDeviceSynchronize(); // Functional Utility: Synchronize the device to ensure the kernel completes before retrieving values.

	// Functional Utility: Copy retrieved values from device memory to host memory.
	cudaMemcpy(hostValues, deviceValues, sizeof(int) * numKeys, cudaMemcpyDeviceToHost);

	// Functional Utility: Free allocated GPU memory.
	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	return hostValues; // Functional Utility: Return pointer to host array of retrieved values. Caller is responsible for freeing this memory.
}


float GpuHashTable::loadFactor() 
{
	float loadFactor = ((float)this->currentSize) / this->maxSize; // Functional Utility: Calculate the load factor as the ratio of current elements to total capacity.
	return loadFactor; // Functional Utility: Return the calculated load factor.
}



#define HASH_INIT GpuHashTable GpuHashTable(1); // @brief Initializes a GpuHashTable instance with an initial capacity of 1.
#define HASH_RESERVE(size) GpuHashTable.reshape(size); // @brief Reshapes the GpuHashTable to a specified new size.

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys) // @brief Inserts a batch of key-value pairs into the GpuHashTable.
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys) // @brief Retrieves a batch of values for given keys from the GpuHashTable.

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor() // @brief Retrieves the current load factor of the GpuHashTable.

#include "test_map.cpp"
/**
 * @brief Conditional compilation block for CPU-specific hash table definitions.
 *
 * This section defines global constants, utility macros, and data structures
 * that are essential for the hash table implementation, particularly when
 * differentiating between GPU and CPU contexts, though in this file, it primarily
 * serves as the definition for these core components.
 */
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID	0 // @brief Defines the value representing an invalid or empty key in the hash table.

#define BLOCK_SIZE 1024 // @brief Defines the number of threads per block for CUDA kernel launches.

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0) // @brief A utility macro for error checking and abnormal program termination.

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



__device__ int myHash(int data, int limit) {
	/**
	 * @brief Primary hash function for CUDA kernels.
	 * Functional Utility: Computes a hash value for `data` specifically for use
	 *                     within CUDA device kernels. It uses a large prime constant
	 *                     and a modulo operation with `limit` to map keys to
	 *                     indices within the hash table.
	 * @param data The input integer data to hash.
	 * @param limit The upper bound for the hash value (size of the hash table).
	 * @return An integer hash value within the range [0, limit-1].
	 */
	return ((long)abs(data) * 1037059llu) % 53944293929llu % limit;
}

struct HashMapData
{
	int key; // @brief The key stored in the hash table entry.
	int value; // @brief The value associated with the key in the hash table entry.
};

/**
 * @class GpuHashTable
 * @brief Manages a dynamically resizable hash table on the GPU.
 *
 * This class encapsulates the GPU memory management and CUDA kernel launches
 * required to implement a high-performance hash table. It provides an interface
 * for creating, reshaping, and performing batch insert and retrieve operations
 * on the GPU hash table.
 */
class GpuHashTable
{
public:
	int maxSize; // @brief The current maximum capacity of the hash table (number of buckets).
	int currentSize; // @brief The current number of active key-value pairs in the hash table.
	HashMapData* data; // @brief Pointer to GPU memory where the hash table data (array of HashMapData) is stored.

public:
    /**
     * @brief Constructor for GpuHashTable.
     * @param size Initial capacity of the hash table.
     */
	GpuHashTable(int size);
    /**
     * @brief Reshapes the hash table to a new size.
     * @param sizeReshape New capacity for the hash table.
     */
	void reshape(int sizeReshape);

    /**
     * @brief Inserts a batch of key-value pairs into the hash table.
     * @param keys Host array of keys to insert.
     * @param values Host array of values to insert.
     * @param numKeys Number of key-value pairs to insert.
     * @return True if insertion is successful.
     */
	bool insertBatch(int* keys, int* values, int numKeys);
    // bool insertBatch2(int* keys, int* values, int numKeys); // @brief Unused/alternative batch insertion method.
    /**
     * @brief Retrieves a batch of values for given keys from the hash table.
     * @param key Host array of keys to lookup.
     * @param numItems Number of keys to lookup.
     * @return Host array of retrieved values. Caller is responsible for freeing this memory.
     */
	int* getBatch(int* key, int numItems);

    /**
     * @brief Calculates the current load factor of the hash table.
     * @return The load factor as a float (currentSize / maxSize).
     */
	float loadFactor();
    // void occupancy(); // @brief Unused/debugging method for hash table occupancy.
    // void print(string info); // @brief Unused/debugging method for printing hash table contents.

    /**
     * @brief Destructor for GpuHashTable.
     * Releases GPU memory allocated for the hash table.
     */
	~GpuHashTable();
};

#endif

