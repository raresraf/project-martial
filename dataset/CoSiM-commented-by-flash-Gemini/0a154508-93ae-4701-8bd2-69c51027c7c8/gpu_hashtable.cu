/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA.
 *
 * This file provides a CUDA C++ implementation of a hash table designed to operate
 * efficiently on a GPU. It utilizes a strategy similar to Cuckoo hashing with linear
 * probing and atomic operations to handle collisions and ensure thread-safe concurrent
 * access during insertion and rehashing. The hash table supports batch insertions
 * and retrievals, making it suitable for high-throughput data processing in parallel environments.
 * It also includes dynamic resizing (rehashing) to maintain performance under varying load.
 *
 * Functional Utility: Provides a high-performance key-value store optimized for
 * parallel processing on NVIDIA GPUs, particularly useful in applications requiring
 * rapid lookups and insertions in large datasets.
 *
 * Algorithm:
 * - Hash function: Uses multiplication and modulo with large prime numbers for key distribution.
 * - Linear probing: If the directly hashed location is occupied, a linear scan is performed.
 * - Atomic operations: `atomicCAS` (Compare-And-Swap) is used for thread-safe insertion and rehashing,
 *   preventing race conditions.
 * - Dynamic Resizing (`reshape`): When the load factor exceeds `MAX_LFACTOR`, the hash table
 *   is resized to a larger capacity (determined by `MIN_LFACTOR`), and elements are rehashed
 *   into the new table in parallel using `kernel_reshape`.
 * - Batch Operations: Insertion and retrieval operations (`insertBatch`, `getBatch`) are
 *   designed to process multiple key-value pairs in parallel using CUDA kernels.
 *
 * CUDA Architecture Details:
 * - `__global__` kernels: `kernel_insert`, `kernel_get`, `kernel_reshape` are executed
 *   on the GPU by multiple threads in parallel.
 * - `__device__` functions: `hash_function` is a device-side helper function.
 * - Grid and Block organization: Threads are organized into blocks and grids, with
 *   `index = blockIdx.x * blockDim.x + threadIdx.x` calculating a unique global thread ID.
 * - Device memory management: `cudaMalloc`, `cudaMemset`, `cudaFree` manage memory on the GPU.
 * - Synchronization: `cudaDeviceSynchronize` ensures that all GPU operations complete
 *   before proceeding with host-side code.
 *
 * Time Complexity:
 * - `hash_function`: O(1).
 * - Host-side `GpuHashTable` methods: Dominated by CUDA kernel launch overhead and `cudaMemcpy` (O(N) for N items in batch),
 *   plus kernel execution time.
 * - `kernel_insert`, `kernel_get`, `kernel_reshape`: In the best case, O(1) for each thread (direct hash hit).
 *   In the worst case, performance can degrade to O(bucketSize) or O(linear_probe_length) per thread if many collisions occur.
 *   With good hash functions and appropriate load factors, average case is closer to O(1) per thread.
 * - `reshape`: O(oldBucketSize) due to parallel rehash kernel.
 * Space Complexity:
 * - `GpuHashTable`: O(bucketSize) for the hash map on device memory.
 * - Kernel arguments/temporary device arrays: O(numKeys) for batch operations.
 */
#include <iostream> ///< For standard C++ input/output operations, e.g., `std::cerr`, `printf`.
#include <vector> ///< For dynamic arrays, not directly used in this specific implementation but a common C++ utility.
#include <string> ///< For string manipulation, e.g., `std::string` in `print` method signature.
#include <cmath> ///< For mathematical functions, e.g., `abs`.
#include <algorithm> ///< For algorithms like `min`, `max`, not directly used in the provided kernels.
#include <cstdio> ///< For `fprintf`, `perror`.
#include <cstdlib> ///< For `exit`.
#include <cerrno> ///< For `errno`.

#include "gpu_hashtable.hpp" ///< Header file defining `GpuHashTable` class structures.

#define PRIME_A 653267llu ///< A large prime number used in hash function calculations.
#define PRIME_B 107888587883llu ///< Another large prime number used in hash function calculations.
#define MIN_LFACTOR 0.8f ///< Minimum load factor for the hash table to trigger resizing to a smaller size (not explicitly used for shrinking in current reshape logic, but for target size).
#define MAX_LFACTOR 0.9f ///< Maximum load factor before triggering a hash table reshape (growth).
#define THREADS_PER_BLOCK 512 ///< Number of CUDA threads to launch per block.
#define KEY_INVALID 0 ///< Sentinel value representing an invalid or empty key in the hash table.

/**
 * @brief CUDA device function to compute a hash value for a given integer data.
 * Functional Utility: Provides a fast hash function executed on the GPU, used by kernels
 * to map keys to bucket indices. The hash function uses multiplication and modulo
 * operations with large prime numbers (`PRIME_A`, `PRIME_B`) to distribute keys evenly.
 *
 * @param data The integer key to hash.
 * @param limit The upper bound (size of the hash table) for the hash value.
 * @return An integer hash value within the range [0, limit-1].
 */
__device__
int hash_function(int data, int limit) {
    return ((long)abs(data) * PRIME_A) % PRIME_B % limit;
}

/**
 * @brief CUDA kernel for parallel rehashing of existing entries into a new hash table.
 * Functional Utility: This kernel is launched during a `reshape` operation. Each thread
 * processes an entry from the `oldHashTable` (if valid) and attempts to insert it into
 * the `newHashTable`. It uses `hash_function` to determine the target location and
 * `atomicCAS` for thread-safe insertion, resolving collisions by linear probing.
 *
 * @param oldHashTable The structure representing the old hash table on the device.
 * @param newHashTable The structure representing the new (larger) hash table on the device.
 */
__global__
void kernel_reshape(hashTable oldHashTable, hashTable newHashTable) {
    // Functional Utility: Calculate unique global thread ID.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int oldKey, newKey;

    if (index >= oldHashTable.size) { // Guard against out-of-bounds access.
        return;
    }
    // Precondition: `oldHashTable.map[index].key` must be a valid key or KEY_INVALID.
    newKey = oldHashTable.map[index].key;

    // Block Logic: Only rehash if the old slot contains a valid key.
    if (newKey == KEY_INVALID) {
        return;
    }

    bool found = false; // Flag to indicate if insertion was successful.

    // Functional Utility: Calculate hash for the new hash table size.
    int hashValue = hash_function(newKey, newHashTable.size);

    
    // Block Logic: Linear probing from `hashValue` to the end of the `newHashTable`.
    for (int i = hashValue; i < newHashTable.size; i++) {
        
        // Functional Utility: Atomically attempt to insert `newKey` if the slot `newHashTable.map[i].key` is empty (0).
        // Returns the old value of `newHashTable.map[i].key`.
        oldKey = atomicCAS(&newHashTable.map[i].key, KEY_INVALID, newKey);
        if (oldKey == KEY_INVALID) { // If the slot was empty.
            newHashTable.map[i].value = oldHashTable.map[index].value; // Insert the value.
            found = true;
            break; // Insertion successful, exit loop.
        }
    }

    // Block Logic: If not inserted yet, continue linear probing from the beginning up to `hashValue-1`.
    for (int i = 0; i < hashValue && !found; i++) {
        
        oldKey = atomicCAS(&newHashTable.map[i].key, KEY_INVALID, newKey);
        if (oldKey == KEY_INVALID) { // If the slot was empty.
            newHashTable.map[i].value = oldHashTable.map[index].value; // Insert the value.
            break; // Insertion successful, exit loop.
        }
    }
}

/**
 * @brief CUDA kernel for parallel insertion of key-value pairs into the hash table.
 * Functional Utility: Each thread attempts to insert a single key-value pair. It
 * uses `hash_function` to determine potential locations and `atomicCAS` for thread-safe
 * insertion, resolving collisions via linear probing. It also handles cases where
 * the key already exists (updating the value).
 *
 * @param hashMap The structure representing the hash table on the device.
 * @param device_keys Pointer to device memory containing an array of keys to insert.
 * @param device_values Pointer to device memory containing an array of values.
 * @param numKeys The number of key-value pairs to insert.
 */
__global__
void kernel_insert(hashTable hashMap, int *device_keys, int *device_values, int numKeys) {
    // Functional Utility: Calculate unique global thread ID.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int oldKey, newKey;

    if (index >= numKeys) { // Guard against out-of-bounds access.
        return;
    }
    newKey = device_keys[index]; // Key to insert by current thread.

    bool found = false; // Flag to indicate if insertion was successful.

    // Functional Utility: Calculate hash for the key.
    int hashValue = hash_function(newKey, hashMap.size);

    
    // Block Logic: Linear probing from `hashValue` to the end of the `hashMap`.
    for (int i = hashValue; i < hashMap.size; i++) {
        
        // Functional Utility: Atomically attempt to insert `newKey` if the slot is empty or contains the same key.
        oldKey = atomicCAS(&hashMap.map[i].key, KEY_INVALID, newKey);
        if (newKey == oldKey || oldKey == KEY_INVALID) { // If slot was empty or contained the same key.
            hashMap.map[i].value = device_values[index]; // Insert or update the value.
            found = true;
            return; // Insertion successful, exit loop.
        }
    }

    // Block Logic: If not inserted yet, continue linear probing from the beginning up to `hashValue-1`.
    for (int i = 0; i < hashValue && !found; i++) {
        
        oldKey = atomicCAS(&hashMap.map[i].key, KEY_INVALID, newKey);
        if (newKey == oldKey || oldKey == KEY_INVALID) { // If slot was empty or contained the same key.
            hashMap.map[i].value = device_values[index]; // Insert or update the value.
            return; // Insertion successful, exit loop.
        }
    }
    // Precondition: If this point is reached, the insertion failed (e.g., table is effectively full or extreme collisions).
    // In a production hash table, this might indicate a need for resizing or a more sophisticated collision resolution.
}

/**
 * @brief CUDA kernel for parallel retrieval of values for a batch of keys.
 * Functional Utility: Each thread attempts to find the value associated with a single key.
 * It searches the `hashMap` using linear probing from the hashed location.
 *
 * @param hashMap The structure representing the hash table on the device.
 * @param device_keys Pointer to device memory containing an array of keys to search for.
 * @param values Pointer to device memory where retrieved values will be stored.
 * @param numKeys The number of keys to retrieve.
 */
__global__
void kernel_get(hashTable hashMap, int *device_keys, int *values, int numKeys) {
    // Functional Utility: Calculate unique global thread ID.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int keyToGet;

    
    if (index >= numKeys) { // Guard against out-of-bounds access.
        return;
    }

    keyToGet = device_keys[index]; // Key to search for by current thread.

    bool found = false; // Flag to indicate if key was found.

    // Functional Utility: Calculate hash for the key.
    int hashValue = hash_function(device_keys[index], hashMap.size);

    
    // Block Logic: Linear probing from `hashValue` to the end of the `hashMap`.
    for (int i = hashValue; i < hashMap.size; i++) {
        if (hashMap.map[i].key == keyToGet) { // If key found.
            values[index] = hashMap.map[i].value; // Store the value.
            found = true;
            return; // Retrieval successful, exit loop.
        }
    }

    // Block Logic: If not found yet, continue linear probing from the beginning up to `hashValue-1`.
    for (int i = 0; i < hashValue && !found; i++) {
        if (hashMap.map[i].key == keyToGet) { // If key found.
            values[index] = hashMap.map[i].value; // Store the value.
            return; // Retrieval successful, exit loop.
        }
    }
    // Functional Utility: If the key is not found after exhaustive search, `values[index]` will
    // retain its initial (potentially garbage or zeroed) value.
}


/**
 * @brief Constructor for `GpuHashTable`.
 * Functional Utility: Initializes the host-side `GpuHashTable` object and allocates
 * device memory for the underlying hash map (`hashMap.map`). The map is initialized
 * with `KEY_INVALID` (0) values.
 *
 * @param size The initial size (number of entries) for the hash map on the GPU.
 */
GpuHashTable::GpuHashTable(int size) {
    
    hashMap.size = size; // Set the size of the hash map.
    nrEntries = 0; // Initialize number of entries to zero.

    
    // Block Logic: Allocate device memory for the hash map.
    if (cudaMalloc((void **) &hashMap.map, size * sizeof(hashItem)) != cudaSuccess) {
        printf("[HOST] Couldn't allocate memory [INIT]\n"); // Print error to host console.
        return;
    }

    
    // Functional Utility: Initialize all allocated device memory to 0 (KEY_INVALID).
    cudaMemset(hashMap.map, 0, size * sizeof (hashItem));
}


/**
 * @brief Destructor for `GpuHashTable`.
 * Functional Utility: Frees the device memory allocated for the hash map,
 * preventing memory leaks on the GPU when the `GpuHashTable` object is destroyed.
 */
GpuHashTable::~GpuHashTable() {
    
    cudaFree(hashMap.map); // Free device memory.
}


/**
 * @brief Resizes the hash table to a new capacity and rehashes all existing elements.
 * Functional Utility: Dynamically adjusts the capacity of the hash table to maintain
 * performance (e.g., lower collision rates) as the number of inserted elements changes.
 * This involves allocating new device memory, launching `kernel_reshape` to rehash
 * elements in parallel, and then freeing the old device memory.
 *
 * @param numBucketsReshape The new desired size for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
    int block_size = THREADS_PER_BLOCK; // Use predefined block size.
    hashTable newHashMap; // Declare a new hash table structure.
    newHashMap.size = numBucketsReshape; // Set the new size.

    
    // Block Logic: Allocate device memory for the new hash map.
    if (cudaMalloc((void **) &newHashMap.map, numBucketsReshape * sizeof (hashItem)) != cudaSuccess) {
        printf("[HOST] Couldn't allocate memory [RESHAPE]\n"); // Print error to host console.
        return;
    }

    
    // Functional Utility: Initialize all allocated new device memory to 0 (KEY_INVALID).
    cudaMemset(newHashMap.map, 0, numBucketsReshape * sizeof (hashItem));

    // Functional Utility: Calculate the number of blocks needed for the `kernel_reshape` launch.
    // This ensures all elements of the old hash table are processed.
    unsigned int blocks_no = hashMap.size / block_size;
    
    if (hashMap.size % block_size) {
        ++blocks_no; // Add an extra block if there's a remainder.
    }

    
    // Functional Utility: Launch the `kernel_reshape` to rehash elements from the old table to the new one.
    kernel_reshape<<<blocks_no, block_size>>>(hashMap, newHashMap);
    
    cudaDeviceSynchronize(); // Ensure kernel completes before proceeding with host-side memory operations.

    
    cudaFree(hashMap.map); // Free the old hash map device memory.
    
    hashMap = newHashMap; // Update the hash map structure to point to the new, resized map.
}


/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * Functional Utility: Orchestrates the parallel insertion of multiple key-value pairs.
 * It manages host-to-device memory transfers, triggers resizing (`reshape`) if the
 * load factor (`currLFACTOR`) becomes too high, launches the `kernel_insert`, and updates
 * the internal count of inserted pairs (`nrEntries`).
 *
 * @param keys Pointer to an array of keys on the host.
 * @param values Pointer to an array of values on the host.
 * @param numKeys The number of key-value pairs in the batch.
 * @return `true` if insertion was successful, `false` otherwise (e.g., memory allocation failure).
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys_array, *device_values_array; // Pointers for device memory.
	int numBytes = numKeys * sizeof (int); // Total bytes needed for key/value arrays.
	int block_size = THREADS_PER_BLOCK; // Use predefined block size.

	
	// Block Logic: Allocate device memory for keys and values for transfer.
	cudaMalloc((void **) &device_keys_array, numBytes);
	cudaMalloc((void **) &device_values_array, numBytes);
	
	if (device_keys_array == 0 || device_values_array == 0) { // Error handling for device memory allocation.
	    printf("[HOST] Couldn't allocate memory [INSERT_BATCH]\n");
	    return false;
	}

	
	// Functional Utility: Check current load factor and trigger `reshape` if it exceeds `MAX_LFACTOR`.
	float currLFACTOR = (float) ((nrEntries + numKeys) / hashMap.size);
	if (currLFACTOR >= MAX_LFACTOR) {
	    // Functional Utility: Reshape to a size that ensures the load factor is below `MIN_LFACTOR` after insertion.
	    reshape((int) ((nrEntries + numKeys) / MIN_LFACTOR));
	}

	
	// Functional Utility: Copy keys and values from host memory to device memory.
	cudaMemcpy(device_keys_array, keys, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_values_array, values, numBytes, cudaMemcpyHostToDevice);

	// Functional Utility: Calculate the number of blocks needed for the `kernel_insert` launch.
	unsigned int blocks_no = numKeys / block_size;
	
	if (numKeys % block_size) {
	    ++blocks_no; // Add an extra block if there's a remainder.
	}

	
	// Functional Utility: Launch the `kernel_insert` to perform parallel insertion.
	kernel_insert<<<blocks_no, block_size>>> (hashMap, device_keys_array, device_values_array, numKeys);
	
	cudaDeviceSynchronize(); // Ensure kernel completes before proceeding.

	
    nrEntries += numKeys; // Update the total count of entries.

    
    // Functional Utility: Free temporary device memory used for input arrays.
    cudaFree(device_keys_array);
    cudaFree(device_values_array);
    return true;
}


/**
 * @brief Retrieves a batch of values for given keys from the hash table.
 * Functional Utility: Orchestrates the parallel retrieval of multiple values.
 * It manages host-to-device memory transfer for keys, launches the `kernel_get`,
 * and transfers the retrieved values from device memory back to newly allocated
 * host memory, returning a pointer to these values.
 *
 * @param keys Pointer to an array of keys on the host to search for.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to host memory containing the retrieved values. The caller
 *         is responsible for `free`ing this memory. Returns `nullptr` on allocation failure.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_keys_array, *device_values_array; // Pointers for device memory.
    size_t numBytes = numKeys * sizeof (int); // Total bytes needed for key/value arrays.
    int block_size = THREADS_PER_BLOCK; // Use predefined block size.

    
	// Block Logic: Allocate device memory for keys and values.
	cudaMalloc ((void **) &device_keys_array, numBytes);
	cudaMalloc((void **) &device_values_array, numBytes);

	
	// Block Logic: Allocate host memory for the results to be returned.
	int *valuesToReturn = (int *) malloc (numBytes);

	
	if (device_keys_array == 0 || device_values_array == 0 || valuesToReturn == 0) { // Error handling for memory allocation.
	    printf("[HOST]  Couldn't allocate memory [GET_BATCH]\n");
	    return nullptr;
	}

	
	// Functional Utility: Copy keys from host memory to device memory.
	cudaMemcpy(device_keys_array, keys, numBytes, cudaMemcpyHostToDevice);

	// Functional Utility: Calculate the number of blocks needed for the `kernel_get` launch.
	unsigned int blocks_no = numKeys / block_size;
	
	if (numKeys % block_size) {
	    ++blocks_no; // Add an extra block if there's a remainder.
	}

	
	// Functional Utility: Launch the `kernel_get` to perform parallel retrieval.
	kernel_get<<<blocks_no, block_size>>>(hashMap, device_keys_array, device_values_array, numKeys);
	
	cudaDeviceSynchronize(); // Ensure kernel completes before proceeding.

	
	// Functional Utility: Copy the retrieved values from device memory back to host memory.
	cudaMemcpy(valuesToReturn, device_values_array, numBytes, cudaMemcpyDeviceToHost);

	
	// Functional Utility: Free temporary device memory used for input/output arrays on the device.
	cudaFree(device_keys_array);
	cudaFree(device_values_array);

	
	return valuesToReturn; // Return pointer to the host array containing the results.
}


/**
 * @brief Calculates the current load factor of the hash table.
 * Functional Utility: Provides a metric for the density of the hash table,
 * indicating how full it is. This is used by `insertBatch` to decide when
 * to trigger a `reshape` operation to maintain efficient performance.
 *
 * @return The load factor (number of entries / hash map size). Returns 0.f if `hashMap.size` is 0.
 */
float GpuHashTable::loadFactor() {
	return (hashMap.size == 0)? 0.f : (float (nrEntries) / hashMap.size);
	
}

// Unimplemented methods in this file.
// void GpuHashTable::occupancy() {
// // Placeholder for future implementation to check bucket occupancy.
// }
//
// void GpuHashTable::print(std::string info) {
// // Placeholder for future implementation to print hash table contents for debugging.
// }


#define HASH_INIT GpuHashTable GpuHashTable(1); ///< Macro for initializing a `GpuHashTable` object.
#define HASH_RESERVE(size) GpuHashTable.reshape(size); ///< Macro for reshaping the `GpuHashTable`.

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys) ///< Macro for batch insertion into `GpuHashTable`.
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys) ///< Macro for batch retrieval from `GpuHashTable`.

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor() ///< Macro for getting the load factor of `GpuHashTable`.

// #include "test_map.cpp" ///< Include for testing purposes. (Typically commented out in production code)
// Functional Utility: The content below this point is typically from a .hpp file (e.g., gpu_hashtable.hpp)
// or a common utility header, and it defines constants, helper hash functions, and the class structure.
// It is duplicated here, likely for self-containment or specific compilation environments.
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std; // Brings all identifiers from the std namespace into the current scope.

#define	KEY_INVALID		0 ///< Constant: Represents an invalid or empty key in the hash table.

/**
 * @brief Macro for error handling and exiting the program.
 * Functional Utility: Provides a convenient way to check assertions and
 * terminate the program with an error message if an assertion fails, printing
 * file and line number information (`__FILE__`, `__LINE__`) and a system error (`perror`).
 *
 * @param assertion The boolean condition to check. If true, the program exits.
 * @param call_description A string describing the function call or context where the error occurred.
 */
#define DIE(assertion, call_description) 
	do { \
		if (assertion) { \
		fprintf(stderr, "(%s, %d): ", \
		__FILE__, __LINE__); \
		perror(call_description); \
		exit(errno); \
	}	
} while (0)
	
/**
 * @brief Array of large prime numbers.
 * Functional Utility: This list provides a set of large prime numbers used in the
 * custom hash functions (`hash1`, `hash2`, `hash3`). Using prime numbers for modulo
 * operations in hashing helps to achieve a more uniform distribution of hash values,
 * which in turn reduces collisions and improves hash table performance.
 */
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


/**
 * @brief Hash function 1.
 * Functional Utility: Computes a hash value for a given integer data using a specific
 * prime from `primeList` and applies modulo operations to keep the result within `limit`.
 * This is used as one of the candidate hash functions for the hash table.
 *
 * @param data The integer key to hash.
 * @param limit The upper bound (size of the hash table) for the hash value.
 * @return An integer hash value within the range [0, limit-1].
 */
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
/**
 * @brief Hash function 2.
 * Functional Utility: Computes a hash value for a given integer data using a specific
 * prime from `primeList` and applies modulo operations to keep the result within `limit`.
 * This is used as one of the candidate hash functions for the hash table.
 *
 * @param data The integer key to hash.
 * @param limit The upper bound (size of the hash table) for the hash value.
 * @return An integer hash value within the range [0, limit-1].
 */
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}


/**
 * @brief Hash function 3.
 * Functional Utility: Computes a hash value for a given integer data using a specific
 * prime from `primeList` and applies modulo operations to keep the result within `limit`.
 * This is used as one of the candidate hash functions for the hash table.
 *
 * @param data The integer key to hash.
 * @param limit The upper bound (size of the hash table) for the hash value.
 * @return An integer hash value within the range [0, limit-1].
 */
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}


/**
 * @struct hashItem
 * @brief Represents a single key-value pair entry in the hash table.
 * Functional Utility: This structure defines the fundamental unit of data storage
 * within the hash table's buckets.
 */
struct hashItem {
    int key, value; ///< Key and value pair.
};

/**
 * @struct hashTable
 * @brief Represents the structure of the hash table itself on the device.
 * Functional Utility: This structure holds the metadata for the hash table
 * (its size) and a pointer to the actual array of `hashItem`s stored in device memory.
 * It's passed to CUDA kernels to operate on the hash table.
 */
struct hashTable {
    int size; ///< The current allocated size of the hash map (number of entries).
    hashItem *map; ///< Pointer to the array of `hashItem`s in device memory.
};

/**
 * @class GpuHashTable
 * @brief (Redeclaration from gpu_hashtable.hpp and defined in this .cu file)
 * Functional Utility: This class provides the host-side interface to manage and
 * interact with the GPU-resident hash table. It encapsulates the device memory
 * pointers, manages operations like resizing and batch transfers, and launches
 * CUDA kernels to perform the actual hash table operations on the GPU.
 */
class GpuHashTable
{
	public:
        hashTable hashMap; ///< The underlying hash table structure on the device.
        int nrEntries; ///< Counter for the number of key-value pairs currently in the hash table.

		GpuHashTable(int size); ///< Constructor: Initializes the hash table.
		void reshape(int sizeReshape); ///< Resizes and rehashes the hash table.
		
		bool insertBatch(int *keys, int* values, int numKeys); ///< Inserts a batch of key-value pairs.
		int* getBatch(int* key, int numItems); ///< Retrieves a batch of values for given keys.
		
		float loadFactor(); ///< Calculates the current load factor.
		void occupancy(); ///< Placeholder for a method to calculate bucket occupancy.
		void print(string info); ///< Placeholder for a method to print hash table contents.
	
		~GpuHashTable(); ///< Destructor: Frees allocated device memory.
};

#endif // _HASHCPU_