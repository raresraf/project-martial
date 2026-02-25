/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA.
 *
 * This file provides a high-performance hash table designed for parallel
 * execution on NVIDIA GPUs. It supports batch insertion and retrieval
 * operations, and dynamic resizing (reshaping). Collision resolution
 * is handled using linear probing with atomic operations to ensure correctness
 * in a concurrent environment.
 *
 * Key features include:
 * - GPU kernels for parallel insert, copy (for reshape), and get operations.
 * - Atomic operations (`atomicCAS`, `atomicInc`) for thread-safe updates.
 * - Custom hash functions.
 * - Host-side wrapper class `GpuHashTable` for managing GPU memory and kernel launches.
 * - Dynamic resizing of the hash table to maintain a healthy load factor.
 */

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <cstdio>
#include <cstdlib>
#include <cerrno> // For ENOMEM

#include "gpu_hashtable.hpp" // Assumed to contain HashTableEntry and GpuHashTable declarations

/**
 * @brief CUDA kernel for inserting key-value pairs into the hash table.
 *
 * This kernel processes a batch of key-value pairs, computing a hash index
 * for each key. It uses linear probing to resolve collisions and atomic
 * operations to ensure thread-safe insertion and update of entries.
 *
 * @param hashtable_vec Pointer to the hash table array in global device memory.
 * @param size The current size (number of buckets) of the hash table.
 * @param keys Pointer to an array of keys to insert (in device memory).
 * @param values Pointer to an array of values to insert (in device memory).
 * @param numKeys The number of keys to process in this batch.
 * @param entries_added Device pointer to a counter tracking the number of
 *                      new entries actually added (not just updated).
 *                      Updated using `atomicInc`.
 *
 * Thread Indexing Logic:
 * Each thread processes one key-value pair from the input batch.
 * `index = threadIdx.x + blockDim.x * blockIdx.x` maps each thread to a unique
 * element in the input `keys` and `values` arrays.
 *
 * Memory Hierarchy Usage:
 * - `hashtable_vec`, `keys`, `values`: Global device memory.
 * - Atomic operations are used to manage concurrent access to `hashtable_vec`
 *   and `entries_added`.
 *
 * Synchronization Points:
 * - `atomicCAS(&(hashtable_vec[hash_index].occupied), 0, 1)`: Atomically checks
 *   if a bucket is free (occupied == 0) and sets it to occupied (1) if it is.
 *   This prevents multiple threads from trying to claim the same empty slot.
 * - `atomicInc(entries_added, numKeys + 1)`: Atomically increments a global
 *   counter to track the total number of unique entries added to the hash table.
 *   The second argument `numKeys + 1` is used as a wrap-around value, effectively
 *   making it an unbounded counter for this purpose, assuming `numKeys + 1`
 *   is greater than the expected maximum number of increments.
 */
__global__ void kernel_insert(HashTableEntry *hashtable_vec, int size,
		int *keys, int *values, int numKeys, unsigned int *entries_added)
{
	// Calculate a global index for the current thread.
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int hash_index;

	// Only process keys within the bounds of the input batch.
	if (index >= numKeys)
		return;
	
	// Compute the initial hash index for the current key.
	hash_index = hash_func(keys[index], size);

	// Block Logic: Linear probing for collision resolution.
	// Invariant: The loop continues until an empty slot is found or the key
	//            is already present at the current hash_index.
	// Synchronization: `atomicCAS` ensures that only one thread can claim a free slot.
	while(hashtable_vec[hash_index].key != keys[index] &&
		atomicCAS(&(hashtable_vec[hash_index].occupied), 0, 1)) {
		// Move to the next bucket (linear probing).
		hash_index++;
		// Wrap around if the end of the table is reached.
		hash_index %= size;
	}

	// Block Logic: Insert or update the entry.
	// If the current slot's key is not the same as the key to insert, it's a new entry.
	if (hashtable_vec[hash_index].key != keys[index]) {
		// Set the key for the new entry.
		hashtable_vec[hash_index].key = keys[index];
		// Atomically increment the count of new entries added.
		atomicInc(entries_added, numKeys + 1); // numKeys + 1 as wrap-around for atomicInc
	}
	// Always set the value (either new insertion or update).
	hashtable_vec[hash_index].value = values[index];
	
}

/**
 * @brief CUDA kernel for copying/rehashing entries during a hash table reshape operation.
 *
 * This kernel iterates through the old hash table and rehashes each occupied
 * entry into a new, larger hash table. It uses linear probing with atomic
 * compare-and-swap (`atomicCAS`) to handle potential collisions during the copy.
 *
 * @param hashtable_vec Pointer to the old hash table array (source) in global device memory.
 * @param size The size of the old hash table.
 * @param new_hashtable_vec Pointer to the new hash table array (destination) in global device memory.
 * @param new_size The size of the new hash table.
 *
 * Thread Indexing Logic:
 * Each thread processes one potential bucket from the old hash table.
 * `index = threadIdx.x + blockDim.x * blockIdx.x` maps each thread to a unique
 * bucket in the `hashtable_vec`.
 *
 * Memory Hierarchy Usage:
 * - `hashtable_vec`, `new_hashtable_vec`: Global device memory.
 * - `current_entry`: Register memory (private to each thread).
 *
 * Synchronization Points:
 * - `atomicCAS(&(new_hashtable_vec[hash_index].occupied), 0, 1)`: Atomically checks
 *   if a bucket in the new hash table is free and sets it to occupied if it is.
 *   This is essential for correct rehashing in parallel.
 */
__global__ void kernel_copy(HashTableEntry *hashtable_vec, int size,
		HashTableEntry *new_hashtable_vec, int new_size)
{
	// Calculate a global index for the current thread, corresponding to an old hash table bucket.
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	HashTableEntry current_entry;

	// Only process buckets within the bounds of the old hash table.
	if (index >= size)
		return;
	
	// Read the entry from the old hash table.
	current_entry = hashtable_vec[index];

	// Block Logic: Rehash and copy only occupied entries from the old table to the new one.
	// Synchronization: `atomicCAS` handles potential collisions in the new table.
	if (current_entry.occupied == 1) {
		// Compute the new hash index for the entry's key in the new table.
		int hash_index = hash_func(current_entry.key, new_size);

		// Linear probing with atomic compare-and-swap for collision resolution in the new table.
		while (atomicCAS(&(new_hashtable_vec[hash_index].occupied), 0, 1)) {
			hash_index++;
			hash_index %= new_size;
		}

		// Once a slot is claimed (or found already claimed by the same key, though not explicitly checked here),
		// copy the key and value to the new hash table.
		new_hashtable_vec[hash_index].key = current_entry.key;
		new_hashtable_vec[hash_index].value = current_entry.value;
	}
}

/**
 * @brief CUDA kernel for retrieving values associated with a batch of keys.
 *
 * This kernel processes a batch of keys, computing a hash index for each.
 * It uses linear probing to find the key in the hash table. If the key is
 * found, its corresponding value is written to the output `values` array.
 * If not found, the corresponding `values` entry remains unchanged (or 0 if initialized).
 *
 * @param hashtable_vec Pointer to the hash table array in global device memory.
 * @param size The current size (number of buckets) of the hash table.
 * @param keys Pointer to an array of keys to retrieve (in device memory).
 * @param values Pointer to an array where retrieved values will be stored (in device memory).
 * @param numKeys The number of keys to process in this batch.
 *
 * Thread Indexing Logic:
 * Each thread processes one key from the input batch.
 * `index = threadIdx.x + blockDim.x * blockIdx.x` maps each thread to a unique
 * element in the input `keys` array.
 *
 * Memory Hierarchy Usage:
 * - `hashtable_vec`, `keys`, `values`: Global device memory.
 */
__global__ void kernel_get(HashTableEntry *hashtable_vec, int size, int *keys,
		int *values, int numKeys)
{
	// Calculate a global index for the current thread.
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int hash_index;

	// Only process keys within the bounds of the input batch.
	if (index >= numKeys)
		return;
	
	// Compute the initial hash index for the current key.
	hash_index = hash_func(keys[index], size);

	// Block Logic: Linear probing to find the key.
	// Invariant: The loop continues as long as the current bucket is occupied
	//            and its key does not match the target key.
	while (hashtable_vec[hash_index].occupied == 1 &&
		keys[index] != hashtable_vec[hash_index].key) {
		// Move to the next bucket (linear probing).
		hash_index++;
		// Wrap around if the end of the table is reached.
		hash_index %= size;
	}

	// If the key is found (bucket is occupied and keys match), write the value.
	// If the loop terminated because `occupied` was 0, the key is not in the table,
	// and `values[index]` would retain its initial value (likely 0 from calloc).
	if (hashtable_vec[hash_index].occupied == 1)
		values[index] = hashtable_vec[hash_index].value;
}


/**
 * @brief Constructor for the GpuHashTable class.
 *
 * Initializes the hash table on the GPU, allocating memory for the table
 * entries and setting initial size and element count.
 *
 * @param size The initial number of buckets for the hash table.
 */
GpuHashTable::GpuHashTable(int size)
{
	// Set the CUDA device (device 1 is hardcoded here).
	HANDLE_ERROR(cudaSetDevice(1));

	// Allocate memory for the hash table entries in device global memory.
	HANDLE_ERROR(
		cudaMalloc(
			&(this->hashtable_vec),
			size * sizeof(HashTableEntry)
		)
	);

	this->size = size;        // Store the allocated size.
	this->num_elems = 0;      // Initialize the number of elements to 0.
}


/**
 * @brief Destructor for the GpuHashTable class.
 *
 * Frees the dynamically allocated device memory used by the hash table.
 */
GpuHashTable::~GpuHashTable()
{
	// Free the device memory for the hash table vector.
	HANDLE_ERROR(cudaFree(this->hashtable_vec));
}


/**
 * @brief Reshapes (resizes) the hash table on the GPU.
 *
 * This function allocates a new, larger hash table, copies all existing
 * elements from the old table to the new one using `kernel_copy`,
 * and then frees the old table.
 *
 * @param numBucketsReshape The target number of elements for the new table.
 *                          The actual new_size is calculated based on a load factor.
 */
void GpuHashTable::reshape(int numBucketsReshape)
{
	HashTableEntry *new_hashtable_vec;
	// Calculate the new size based on desired number of buckets, assuming a load factor of 0.8.
	unsigned long new_size = numBucketsReshape / 0.8;
	
	// Allocate memory for the new hash table in device global memory.
	HANDLE_ERROR(
		cudaMalloc(
			&new_hashtable_vec,
			new_size * sizeof(HashTableEntry)
		)
	);

	// Define block size and calculate grid size for the kernel launch.
	const size_t block_size = 16;
	size_t blocks_no = this->size/16 + (this->size % 16 == 0 ? 0 : 1);

	// Launch the kernel to copy elements from the old table to the new table.
	// The `<<<blocks_no, block_size>>>` syntax specifies the grid and block dimensions.
	kernel_copy<<<(int)blocks_no, (int)block_size>>>( // Explicit cast for kernel launch parameters
		this->hashtable_vec,
		this->size,
		new_hashtable_vec,
		new_size
	);

	// Ensure the kernel completes before proceeding with host-side memory operations.
	HANDLE_ERROR(cudaDeviceSynchronize());

	// Free the memory of the old hash table.
	HANDLE_ERROR(cudaFree(this->hashtable_vec));
	// Update the GpuHashTable object to point to the new hash table.
	this->hashtable_vec = new_hashtable_vec;
	this->size = new_size;
}


/**
 * @brief Inserts a batch of key-value pairs into the GPU hash table.
 *
 * This function handles memory allocation for keys and values on the device,
 * transfers data from host to device, launches the `kernel_insert` kernel,
 * and updates the element count. It also triggers a reshape if the load factor
 * becomes too high.
 *
 * @param keys Pointer to an array of keys on the host.
 * @param values Pointer to an array of values on the host.
 * @param numKeys The number of key-value pairs in the batch.
 * @return True indicating the batch insertion process was initiated.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys)
{
	// Define block size and calculate grid size for the kernel launch.
	const size_t block_size = 16;
	size_t blocks_no = numKeys/16 + (numKeys % 16 == 0 ? 0 : 1);
	unsigned int *new_entries_no;
	int *device_keys, *device_values;

	// Check if reshaping is needed to accommodate new entries.
	if (this->num_elems + numKeys >= this->size)
		reshape((this->num_elems + numKeys)); // Reshape to fit new elements.

	// Allocate device memory for the keys and values to be inserted.
	HANDLE_ERROR(
		cudaMalloc(
			&device_keys,
			numKeys * sizeof(int)
		)
	);
	HANDLE_ERROR(
		cudaMalloc(
			&device_values,
			numKeys * sizeof(int)
		)
	);

	// Allocate managed memory for the new entries counter.
	// Managed memory simplifies data sharing between host and device.
	HANDLE_ERROR(
		cudaMallocManaged(
			&new_entries_no,
			sizeof(unsigned int)
		)
	);

	// Initialize the counter on the host (which will be visible on device due to managed memory).
	*new_entries_no = 0;

	// Copy keys from host memory to device memory.
	HANDLE_ERROR(
		cudaMemcpy(
			device_keys,
			keys,
			numKeys * sizeof(int),
			cudaMemcpyHostToDevice
		)
	);
	// Copy values from host memory to device memory.
	HANDLE_ERROR(
		cudaMemcpy(
			device_values,
			values,
			numKeys * sizeof(int),
			cudaMemcpyHostToDevice
		)
	);
	
	// Launch the kernel to insert the key-value pairs.
	kernel_insert<<<(int)blocks_no, (int)block_size>>>( // Explicit cast for kernel launch parameters
		this->hashtable_vec,
		this->size,
		device_keys,
		device_values,
		numKeys,
		new_entries_no
	);
	// Wait for the kernel to complete execution.
	HANDLE_ERROR(cudaDeviceSynchronize());

	// Update the host-side element count with the number of new entries added by the kernel.
	this->num_elems += *new_entries_no;

	// Free device memory allocated for the input keys and values.
	HANDLE_ERROR(cudaFree(device_keys));
	HANDLE_ERROR(cudaFree(device_values));
	// Free managed memory for the counter.
	HANDLE_ERROR(cudaFree(new_entries_no)); // Added to prevent leak of managed memory

	return true;
}


/**
 * @brief Retrieves a batch of values for given keys from the GPU hash table.
 *
 * This function allocates device memory for keys and results, transfers keys
 * from host to device, launches the `kernel_get` kernel, and then transfers
 * the retrieved values back to host memory.
 *
 * @param keys Pointer to an array of keys on the host for which to retrieve values.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to a newly allocated host array containing the retrieved values.
 *         The caller is responsible for freeing this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys)
{
	// Define block size and calculate grid size for the kernel launch.
	const size_t block_size = 16;
	size_t blocks_no = numKeys/16 + (numKeys % 16 == 0 ? 0 : 1);
	int *device_keys, *device_values, *values;

	// Allocate device memory for the keys to be searched.
	HANDLE_ERROR(
		cudaMalloc(
			&device_keys,
			numKeys * sizeof(int)
		)
	);
	// Allocate device memory to store the retrieved values.
	HANDLE_ERROR(
		cudaMalloc(
			&device_values,
			numKeys * sizeof(int)
		)
	);

	// Copy keys from host memory to device memory.
	HANDLE_ERROR(
		cudaMemcpy(
			device_keys,
			keys,
			numKeys * sizeof(int),
			cudaMemcpyHostToDevice
		)
	);

	// Allocate host memory for the result values and initialize to zero.
	values = (int *)calloc(numKeys, sizeof(int));
	// Error handling for host memory allocation.
	if(values == NULL) {
		fprintf(stderr, "values get calloc\n");
		exit(ENOMEM);
	}

	// Launch the kernel to retrieve values for the given keys.
	kernel_get<<<(int)blocks_no, (int)block_size>>>( // Explicit cast for kernel launch parameters
		this->hashtable_vec,
		this->size,
		device_keys,
		device_values,
		numKeys
	);
	// Wait for the kernel to complete execution.
	HANDLE_ERROR(cudaDeviceSynchronize());

	// Copy the retrieved values from device memory back to host memory.
	HANDLE_ERROR(
		cudaMemcpy(
			values,
			device_values,
			numKeys * sizeof(int),
			cudaMemcpyDeviceToHost
		)
	);

	// Free device memory allocated for keys and values.
	HANDLE_ERROR(cudaFree(device_keys));
	HANDLE_ERROR(cudaFree(device_values));
	
	return values; // Return pointer to host array with results.
}


/**
 * @brief Calculates the current load factor of the hash table.
 *
 * The load factor is defined as the number of elements divided by the
 * total number of buckets (size).
 *
 * @return The current load factor as a float.
 */
float GpuHashTable::loadFactor()
{
	return (float)this->num_elems / this->size;
}

// These two methods are declared in gpu_hashtable.hpp but not defined in .cu file.
// void GpuHashTable::occupancy() { /* ... */ }
// void GpuHashTable::print(string info) { /* ... */ }


// Macros for convenient interaction with the GpuHashTable object.
// These are likely used for testing or a simplified interface.

#define HASH_INIT GpuHashTable GpuHashTable(1); /**< @brief Initializes a GpuHashTable instance with an initial size of 1. */
#define HASH_RESERVE(size) GpuHashTable.reshape(size); /**< @brief Resizes the GpuHashTable to accommodate 'size' elements. */

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys) /**< @brief Inserts a batch of key-value pairs into the hash table. */
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys) /**< @brief Retrieves a batch of values from the hash table. */

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor() /**< @brief Retrieves the current load factor of the hash table. */

#include "test_map.cpp" // Include for testing purposes.

#ifndef _HASHCPU_
#define _HASHCPU_

// Error codes, might be duplicated from system headers but defined here.
#define ENOMEM			12
#define	KEY_INVALID		0 // Represents an invalid key or a default key value.
#define FNV_OFFSET		2166136261llu // FNV-1a hash initial offset basis.
#define FNV_PRIME		16777619llu // FNV-1a hash prime.
#define PRIME_NUM		14480561146010017169llu // A large prime number used in hash function calculations.

/**
 * @brief Macro for robust CUDA error checking.
 *
 * This macro checks the return value of CUDA API calls and exits the program
 * with an error message if an error occurred.
 *
 * @param assertion The CUDA error code to check.
 * @param call_description A string describing the CUDA call.
 */
#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
			fprintf(stderr, "(%s, %d): ",	\
			__FILE__, __LINE__);	\
			perror(call_description);	\
			exit(errno);	\
		}	\
	} while (0)


/**
 * @brief Helper function to handle CUDA runtime errors.
 *
 * This function prints the CUDA error message and the file/line number
 * where the error occurred, then exits the program.
 *
 * @param err The CUDA error code.
 * @param file The name of the file where the error occurred.
 * @param line The line number where the error occurred.
 */
static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        cerr << cudaGetErrorString(err) << " in " 
            << file << " at line " << line << endl;
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) /**< @brief Macro for simplifying CUDA error handling calls. */

/**
 * @brief Basic device hash function for integer data.
 *
 * This hash function uses modular arithmetic with a large prime to
 * compute a hash index within a given limit.
 *
 * @param data The integer key to hash.
 * @param limit The upper bound for the hash index (e.g., table size).
 * @return The computed hash index.
 */
__device__ int hash_func_basic(int data, int limit) {
	// Uses a multiplicative hash approach with PRIME_NUM and FNV_PRIME.
	return ((long)abs(data) * PRIME_NUM) % FNV_PRIME % limit;
}


/**
 * @brief Device FNV-1a hash function for integer data.
 *
 * Implements the FNV-1a hash algorithm to generate a hash for an integer key.
 * This hash function processes the integer byte by byte.
 *
 * @param data The integer key to hash.
 * @param limit The upper bound for the hash index (e.g., table size).
 * @return The computed hash index.
 */
__device__ int hash_func(int data, int limit) {
	unsigned long long hash = FNV_OFFSET;

	// Block Logic: Iterate through each byte of the integer data.
	// Invariant: 'hash' is updated based on the current byte, following FNV-1a algorithm.
	for (size_t i = 0; i < sizeof(int); i++) {
		// Extract one byte from the integer.
		char byte_data = data >> (i * 8) & 0xFF;
		// XOR with the current byte.
		hash = hash ^ byte_data;
		// Multiply by the FNV prime.
		hash *= FNV_PRIME;
	}

	// Return the hash index modulo the limit (table size).
	return hash % limit;
}

/**
 * @brief Structure representing an entry in the hash table.
 *
 * Each entry stores a key, its associated value, and an 'occupied' flag
 * to indicate whether the bucket contains a valid entry (0: free, 1: occupied).
 */
typedef struct {
	int key = 0;       /**< @brief The key stored in this hash table entry. */
	int value = 0;     /**< @brief The value associated with the key. */
	int occupied = 0;  /**< @brief Flag indicating if the entry is occupied (1) or free (0). */
} HashTableEntry;


/**
 * @brief GpuHashTable class (Host-side definition).
 *
 * This class provides the host-side interface for managing a hash table
 * that resides in GPU global memory. It handles memory allocation/deallocation,
 * kernel launches, and table resizing logic.
 */
class GpuHashTable
{
public:
	unsigned long size;            /**< @brief The total number of buckets in the hash table. */
	unsigned long num_elems;       /**< @brief The current number of elements stored in the hash table. */
	HashTableEntry *hashtable_vec; /**< @brief Pointer to the hash table array in GPU global memory. */

	/**
	 * @brief Constructor for GpuHashTable.
	 * @param size Initial size of the hash table.
	 */
	GpuHashTable(int size);
	
	/**
	 * @brief Reshapes (resizes) the hash table.
	 * @param sizeReshape The desired new capacity for the hash table.
	 */
	void reshape(int sizeReshape);
	
	/**
	 * @brief Inserts a batch of key-value pairs into the hash table.
	 * @param keys Host pointer to the array of keys.
	 * @param values Host pointer to the array of values.
	 * @param numKeys Number of key-value pairs in the batch.
	 * @return True if the batch insertion process was successfully initiated.
	 */
	bool insertBatch(int *keys, int* values, int numKeys);
	
	/**
	 * @brief Retrieves a batch of values for given keys from the hash table.
	 * @param key Host pointer to the array of keys to retrieve.
	 * @param numItems Number of keys in the batch.
	 * @return Host pointer to an array containing the retrieved values.
	 *         The caller is responsible for freeing this memory.
	 */
	int* getBatch(int* key, int numItems);
	
	/**
	 * @brief Calculates the current load factor of the hash table.
	 * @return The load factor (num_elems / size).
	 */
	float loadFactor();
	
	// These methods are declared but not defined in the provided .cu file.
	// void occupancy();
	// void print(string info);

	/**
	 * @brief Destructor for GpuHashTable.
	 */
	~GpuHashTable();
};

#endif

