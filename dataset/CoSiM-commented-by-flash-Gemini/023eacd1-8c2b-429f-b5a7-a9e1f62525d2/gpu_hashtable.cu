/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table (`GpuHashTable`) leveraging CUDA.
 * This file provides a parallel key-value storage and retrieval mechanism designed for NVIDIA GPUs.
 * It utilizes CUDA Unified Memory for efficient data sharing between host and device, employs
 * linear probing for collision resolution, and supports dynamic resizing of the hash table.
 *
 * Memory Hierarchy Usage:
 *   - Utilizes CUDA Unified Memory (`cudaMallocManaged`) for `hashtable` and `hashtable_current_size`.
 *     This allows both host (CPU) and device (GPU) to access and modify the same memory region,
 *     simplifying memory management and data transfer.
 *
 * Thread Indexing Logic:
 *   - CUDA kernels (`kernel_insert`, `kernel_get`) use `idx = blockIdx.x * blockDim.x + threadIdx.x`
 *     to calculate a unique global thread index. This allows each thread to process a distinct key
 *     in parallel across the GPU grid.
 *
 * Synchronization Points:
 *   - `atomicCAS`, `atomicExch`, `atomicAdd`: Used within kernels to ensure atomic, safe,
 *     and concurrent updates to shared hash table memory from multiple GPU threads, crucial
 *     for preventing race conditions during insertions and value updates.
 *   - `__syncthreads()`: Employed within kernels to synchronize all threads within a block.
 *     Its usage within loops for collision resolution ensures that memory writes are visible
 *     across threads before proceeding, maintaining data consistency.
 *   - `cudaDeviceSynchronize()`: Used on the host side to ensure all GPU kernel operations
 *     have completed before CPU continues, guaranteeing that results are ready or memory
 *     is safe to deallocate/re-access by the host.
 *
 * Algorithm:
 *   - Linear Probing Hash Table: Collisions are resolved by sequentially searching for the next
 *     available slot in the hash table.
 *   - Custom Hash Function: A multiplicative hash function (or similar) is used to map keys
 *     to initial table indices.
 *   - Dynamic Resizing (`reshape`): When the hash table's load factor exceeds a threshold
 *     (implicitly handled by checking `hashtable_current_size + numKeys > hashtable_size`),
 *     the table is resized to a larger capacity, and all existing elements are rehashed.
 *
 * Optimization:
 *   - Parallel Execution: Key-value operations are parallelized across GPU threads,
 *     significantly accelerating batch insertions and retrievals.
 *   - Atomic Operations: Critical sections involving hash table updates are protected
 *     by CUDA atomic operations to ensure correctness in a concurrent environment.
 *
 * Time Complexity:
 *   - `insertBatch`, `getBatch`: In an ideal scenario with minimal collisions, operations are
 *     effectively O(1) per key in parallel (O(numKeys / num_threads)). In worst-case scenarios
 *     (high collision rates, dense table), performance can degrade towards O(numKeys * hashtable_size)
 *     for individual operations due to linear probing.
 *   - `reshape`: O(hashtable_size_old + hashtable_size_new) due to copying and rehashing elements.
 * Space Complexity:
 *   - O(hashtable_size) for the hash table structure itself.
 *   - O(numKeys) for temporary host and device buffers during batch operations.
 */


#include "gpu_hashtable.hpp"

// Functional Utility: Constructor for the GpuHashTable.
// Initializes the hash table structure, allocating Unified Memory and setting initial states.
// @param size The initial desired size for the hash table.
GpuHashTable::GpuHashTable(int size) {
	// Block Logic: Initialize the hashtable_size member with the provided size.
	hashtable_size = size;

	// Block Logic: Allocate Unified Memory for `hashtable_current_size`.
	// This integer tracks the number of elements currently in the hash table.
	cudaMallocManaged((void **) &hashtable_current_size, sizeof(int));
	// Block Logic: Initialize the current size to 0.
	*hashtable_current_size = 0;

	// Block Logic: Allocate Unified Memory for the hash table array itself.
	// This array will store `node` structures (key-value pairs).
	cudaMallocManaged((void **) &hashtable, size * sizeof(struct node));
	// Block Logic: Initialize all keys in the newly allocated hash table to KEY_INVALID.
	// This marks all slots as empty and available for insertion.
	for (int i = 0; i < size; i++)
		hashtable[i].key = KEY_INVALID;
}

// Functional Utility: Destructor for the GpuHashTable.
// Frees the Unified Memory allocated for the hash table and its size tracker.
GpuHashTable::~GpuHashTable() {
	// Block Logic: Free the memory allocated for the hash table array.
	cudaFree(hashtable);
	// Block Logic: Free the memory allocated for the current size tracker.
	cudaFree(hashtable_current_size);
}

// Functional Utility: Reshapes (resizes) the hash table to a new capacity.
// This involves creating a new, larger hash table, rehashing all existing elements
// from the old table into the new one, and then freeing the old table.
// @param numBucketsReshape The desired base number of buckets for the new hash table.
// This is typically the current number of elements to be stored, used to calculate a new size.
void GpuHashTable::reshape(int numBucketsReshape) {
	// Block Logic: Calculate the new size for the hash table.
	// A common heuristic is to use a prime number or a factor slightly larger than the expected number of elements
	// to maintain a good load factor and reduce collisions. Here, it's approximately 1.07 times `numBucketsReshape`.
	int new_size = numBucketsReshape * 1.07f;

	// Inline: Store the current number of elements before resizing.
	int old_size = *hashtable_current_size;

	// Block Logic: Allocate host memory to temporarily store old keys and values.
	// This is necessary because the `hashtable` pointer will be freed and reallocated.
	// Precondition: `hashtable_current_size` correctly reflects the number of active elements.
	int *old_keys = (int *)malloc(*(hashtable_current_size) * sizeof(int));
	int *old_values = (int *)malloc(*(hashtable_current_size) *sizeof(int));
	
	// Inline: Index to track insertion into `old_keys` and `old_values`.
	int index = 0;

	// Block Logic: Iterate through the old hash table to extract existing key-value pairs.
	// Invariant: Only valid (non-KEY_INVALID) entries are copied to the temporary host arrays.
	for (int i = 0; i < hashtable_size; i++) {
		// Conditional Logic: Check if the current slot contains a valid key.
		if (hashtable[i].key != KEY_INVALID) {
			old_keys[index] = hashtable[i].key;
			old_values[index] = hashtable[i].value;
			index++;
		}
	}

	// Block Logic: Free the Unified Memory of the old hash table and its size tracker.
	cudaFree(hashtable);
	cudaFree(hashtable_current_size);

	// Block Logic: Update the hash table's size to the newly calculated capacity.
	hashtable_size = new_size;

	// Block Logic: Allocate new Unified Memory for `hashtable_current_size` and initialize it to 0.
	cudaMallocManaged((void **) &hashtable_current_size, sizeof(int));
	*hashtable_current_size = 0;

	// Block Logic: Allocate new Unified Memory for the resized hash table.
	cudaMallocManaged((void **) &hashtable, new_size * sizeof(struct node));
	// Block Logic: Initialize all slots in the new hash table to KEY_INVALID, marking them as empty.
	for (int i = 0; i < new_size; i++)
		hashtable[i].key = KEY_INVALID;

	// Functional Utility: Re-insert all previously stored key-value pairs into the new hash table.
	// This step effectively rehashes all elements according to the new table size.
	insertBatch(old_keys, old_values, old_size);

	// Block Logic: Free the host memory used for temporary storage of old keys and values.
	free(old_keys);
	free(old_values);
}

// Functional Utility: CUDA kernel for inserting a batch of key-value pairs into the hash table.
// Each thread processes one key-value pair, attempting to insert it into the hash table
// using linear probing and atomic operations for concurrency control.
// @param keys Device pointer to an array of keys to insert.
// @param values Device pointer to an array of values corresponding to the keys.
// @param numKeys The total number of keys to insert in this batch.
// @param hashtable Device pointer to the hash table structure (array of nodes).
// @param hashtable_size The current total capacity of the hash table.
// @param hashtable_current_size Device pointer to an integer tracking the current number of elements in the hash table.
__global__ void kernel_insert(int *keys, int *values, int numKeys,
struct node *hashtable, int hashtable_size,
int *hashtable_current_size) {
	// Block Logic: Calculate the global thread index.
	// Invariant: Each thread gets a unique 'idx' for processing its assigned key.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// Conditional Logic: Ensure thread does not process out-of-bounds keys.
	if (idx >= numKeys)
		return;

	int my_key = keys[idx];
	int my_value = values[idx];

	// Block Logic: Compute the initial hash for the current key.
	int hash = hash_function(my_key, hashtable_size);
	// Block Logic: Loop indefinitely until the key is successfully inserted or updated.
	// This implements linear probing for collision resolution.
	while (1) {
		// Inline: Atomically compare `hashtable[hash].key` with `KEY_INVALID`.
		// If `KEY_INVALID`, it means the slot is empty, so store `my_key`.
		// Returns the original value of `hashtable[hash].key`.
		int init_val = atomicCAS(&(hashtable[hash].key), KEY_INVALID,
			my_key);

		// Conditional Logic: Check if the slot was empty and the key was successfully inserted.
		if (init_val == KEY_INVALID) {
			// Inline: Atomically exchange `hashtable[hash].value` with `my_value`.
			atomicExch(&hashtable[hash].value, my_value);
			// Inline: Atomically increment the total count of elements in the hash table.
			atomicAdd(hashtable_current_size, 1);

			// Inline: Synchronize all threads within the block to ensure memory visibility before breaking.
			__syncthreads();
			break;
		}
		// Conditional Logic: Check if the key already exists in the current slot (update scenario).
		else if (init_val == my_key) {
			// Inline: Atomically exchange `hashtable[hash].value` with `my_value` (update its value).
			atomicExch(&hashtable[hash].value, my_value);

			// Inline: Synchronize all threads within the block to ensure memory visibility before breaking.
			__syncthreads();
                        break;
		}

		// Block Logic: If collision occurred (slot not empty and key not matched), move to the next slot (linear probing).
		// Invariant: The hash is wrapped around the hashtable_size to stay within bounds.
		hash = (hash + 1) % hashtable_size;
	}
}

// Functional Utility: Host method to insert a batch of key-value pairs into the GPU hash table.
// It manages memory allocation on the device, copies data from host to device, launches the kernel,
// and frees device memory.
// @param keys Host pointer to an array of keys.
// @param values Host pointer to an array of values.
// @param numKeys The number of key-value pairs to insert.
// @return True if the batch insertion was successful, false otherwise (though currently always returns true).
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	// Conditional Logic: If there are no keys to insert, return true immediately.
	if (numKeys <= 0)
		return true;

	// Conditional Logic: Check if resizing is necessary before insertion.
	// Precondition: If adding `numKeys` elements would exceed the current hash table capacity, reshape it.
	if (*hashtable_current_size + numKeys > hashtable_size)
		reshape(hashtable_size + numKeys);

	// Block Logic: Allocate device memory for keys and values.
	int *device_keys, *device_values;

	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));

	// Block Logic: Copy key and value data from host to device.
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int),
		cudaMemcpyHostToDevice);
	
	// Functional Utility: Launch the kernel_insert on the GPU.
	// The number of blocks and threads per block are dynamically determined to process all `numKeys`.
	// Inline: The kernel is launched with `(numKeys + 255) / 256` blocks and `256` threads per block.
	kernel_insert<<<(numKeys + 255) / 256, 256>>>(device_keys, device_values, numKeys,
		hashtable, hashtable_size, 
		hashtable_current_size);
	// Functional Utility: Synchronize the device to ensure all kernel operations are complete before proceeding on the host.
	cudaDeviceSynchronize();

	// Block Logic: Free the device memory allocated for temporary keys and values.
	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}

// Functional Utility: CUDA kernel for retrieving values associated with a batch of keys from the hash table.
// Each thread processes one key, attempts to find its corresponding value using linear probing,
// and stores the result in the `values` array.
// @param keys Device pointer to an array of keys to retrieve.
// @param values Device pointer to an array where retrieved values will be stored.
// @param numKeys The total number of keys to retrieve in this batch.
// @param hashtable Device pointer to the hash table structure (array of nodes).
// @param hashtable_size The current total capacity of the hash table.
__global__ void kernel_get(int *keys, int *values, int numKeys,
struct node *hashtable, int hashtable_size) {
	// Block Logic: Calculate the global thread index.
	// Invariant: Each thread gets a unique 'idx' for processing its assigned key.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// Conditional Logic: Ensure thread does not process out-of-bounds keys.
	if (idx >= numKeys)
		return;

	// Inline: Retrieve the key for the current thread.
	int my_key = keys[idx];
	// Block Logic: Compute the initial hash for the current key.
	int hash = hash_function(my_key, hashtable_size);

	// Block Logic: Loop indefinitely until the key is found.
	// This implements linear probing for finding the key.
	while (1) {
		// Conditional Logic: Check if the current slot contains the desired key.
		if (hashtable[hash].key == my_key) {
			values[idx] = hashtable[hash].value; // Inline: Store the found value.

			// Inline: Synchronize all threads within the block to ensure memory visibility before breaking.
			__syncthreads();
			break;;
		}

		// Block Logic: If the key is not found in the current slot, move to the next slot (linear probing).
        // Invariant: The hash is wrapped around the hashtable_size to stay within bounds.
        hash = (hash + 1) % hashtable_size;
	}
}

// Functional Utility: Host method to retrieve values for a batch of keys from the GPU hash table.
// It manages memory allocation on the device, copies data from host to device, launches the kernel,
// copies results back from device to host, and frees device memory.
// @param keys Host pointer to an array of keys for which to retrieve values.
// @param numKeys The number of keys to retrieve.
// @return A host pointer to an array of retrieved values. The caller is responsible for freeing this memory.
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	// Block Logic: Allocate host memory for the result values.
	int *values = (int *)calloc(numKeys, sizeof(int));

	// Block Logic: Allocate device memory for keys and values.
	int *device_keys, *device_values;

	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));

	// Block Logic: Copy key data from host to device.
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int),
		cudaMemcpyHostToDevice);
	// Block Logic: Copy initial (zeroed) value data from host to device.
	cudaMemcpy(device_values, values, numKeys * sizeof(int),
		cudaMemcpyHostToDevice);

	// Functional Utility: Launch the kernel_get on the GPU.
	// The number of blocks and threads per block are dynamically determined to process all `numKeys`.
	// Inline: The kernel is launched with `(numKeys + 255) / 256` blocks and `256` threads per block.
	kernel_get<<<(numKeys + 255) / 256, 256>>>(device_keys, device_values, numKeys,
		hashtable, hashtable_size);
	// Functional Utility: Synchronize the device to ensure all kernel operations are complete before proceeding on the host.
	cudaDeviceSynchronize();

	// Block Logic: Copy retrieved values from device back to host memory.
	cudaMemcpy(values, device_values, numKeys * sizeof(int),
		cudaMemcpyDeviceToHost);

	// Block Logic: Free the device memory allocated for temporary keys and values.
	cudaFree(device_keys);
	cudaFree(device_values);

	// Functional Utility: Return the host pointer to the array of retrieved values.
	return values;
}

// Functional Utility: Calculates the current load factor of the hash table.
// The load factor is the ratio of the number of elements to the total capacity of the hash table.
// It's a key metric for hash table performance, indicating how full the table is.
// @return The current load factor as a float.
float GpuHashTable::loadFactor() {
	// Inline: Cast `hashtable_current_size` to float for floating-point division.
	return *(hashtable_current_size) / (float)hashtable_size;
}

// Functional Utility: (Declared but not implemented) Likely intended to measure and potentially report
// the occupancy of GPU resources (e.g., SMs, registers, shared memory) when kernels are executing.
// Architectural Intent: Useful for performance tuning and ensuring efficient GPU utilization.
void occupancy();
// Functional Utility: (Declared but not implemented) Likely intended for debugging purposes,
// to print the current state or contents of the hash table to the console or log.
void print(string info);


// Functional Utility: Macro to initialize a GpuHashTable instance.
// This macro simplifies the creation and initial setup of the hash table.
#define HASH_INIT GpuHashTable GpuHashTable(1);
// Functional Utility: Macro to reserve (resize) the GpuHashTable.
// This macro provides a convenient way to trigger the reshape operation for dynamic sizing.
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

// Functional Utility: Macro to insert a batch of key-value pairs into the GpuHashTable.
// This wraps the GpuHashTable::insertBatch method for easier use.
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
// Functional Utility: Macro to retrieve a batch of values from the GpuHashTable.
// This wraps the GpuHashTable::getBatch method for easier use.
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

// Functional Utility: Macro to get the current load factor of the GpuHashTable.
// This wraps the GpuHashTable::loadFactor method.
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"


// Block Logic: Conditional compilation block for CPU hash table definitions.
// This block ensures that the following definitions are only included if _HASHCPU_ is not defined,
// likely to prevent redefinition conflicts if a CPU-specific hash table is also present.
#ifndef _HASHCPU_
#define _HASHCPU_

#include <iostream> // Functional Utility: Includes the C++ Standard Library for input/output operations.
#include <map> // Functional Utility: Includes the C++ Standard Library map, often used for comparison or as a reference.
#include <vector> // Functional Utility: Includes the C++ Standard Library vector for dynamic arrays.
#include <string> // Functional Utility: Includes the C++ Standard Library string for string manipulation.
#include <algorithm> // Functional Utility: Includes various algorithms from the C++ Standard Library.
#include <chrono> // Functional Utility: Includes chrono for high-resolution timing operations, useful for benchmarking.
#include <random> // Functional Utility: Includes random for random number generation, often used in test data generation.

using namespace std; // Functional Utility: Brings the entire std namespace into scope for convenience.

// Functional Utility: Constant representing an invalid or empty key in the hash table.
// It's crucial for distinguishing empty slots from slots containing valid data.
#define KEY_INVALID 0

// Functional Utility: Macro for robust error handling and process termination.
// If the assertion is true, it prints an error message to stderr, including file and line number,
// then exits the program with the last recorded error number.
// Architectural Intent: Provides a standardized way to handle critical, unrecoverable errors.
#define DIE(assertion, call_description)              \
	do                                            \
	{                                             \
		if (assertion)                        \
		{                                     \
			fprintf(stderr, "(%s, %d): ", \
				__FILE__, __LINE__);  \
			perror(call_description);     \
			exit(errno);                  \
		}                                     \
	} while (0)

// Functional Utility: A pre-defined list of prime numbers.
// These primes are often used in hash table implementations for determining table sizes,
// particularly for ensuring good distribution in modulo operations, or in hash function design.
const size_t primeList[] = {
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
	11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu};

__device__ int hash_function(int data, int limit) {
        return ((long)abs(data) * 21407807219llu) % 139193449418173llu % limit;
}

// Functional Utility: Structure representing a node (key-value pair) in the hash table.
// This structure is stored within the `hashtable` array.
struct node {
	int key;   // Inline: The key of the hash table entry. `KEY_INVALID` indicates an empty slot.
	int value; // Inline: The value associated with the key.
};


// Functional Utility: Declaration of the GpuHashTable class.
// This class encapsulates the GPU hash table's data structures and methods.
class GpuHashTable
{
public:
	struct node *hashtable;             // Inline: Pointer to the hash table array in Unified Memory.
	int hashtable_size;                 // Inline: The current total capacity of the hash table.
	int *hashtable_current_size;        // Inline: Pointer to an integer in Unified Memory tracking the number of elements.
	
	GpuHashTable(int size);             // Functional Utility: Constructor to initialize the hash table.
	void reshape(int sizeReshape);      // Functional Utility: Resizes the hash table and rehashes elements.

	bool insertBatch(int *keys, int *values, int numKeys); // Functional Utility: Inserts a batch of key-value pairs.
	int *getBatch(int *key, int numItems);                 // Functional Utility: Retrieves values for a batch of keys.

	float loadFactor();                                     // Functional Utility: Calculates the current load factor.
	void occupancy();                                       // Functional Utility: (Declared but not implemented) Likely for measuring occupancy.
	void print(string info);                                // Functional Utility: (Declared but not implemented) Likely for debugging/printing hash table contents.

	~GpuHashTable();                                        // Functional Utility: Destructor to free allocated memory.
};

// Block Logic: End of conditional compilation for _HASHCPU_.
#endif