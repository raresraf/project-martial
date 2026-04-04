/**
 * @file gpu_functions.cu
 * @brief Implements device-side CUDA kernels for a parallel hash table.
 * @details This file contains the core logic for insertion and retrieval on the GPU.
 * The hash table uses open addressing with linear probing for collision resolution.
 * All operations are designed to be safe for concurrent execution by many threads.
 *
 * @note This file is intended to be included by a host-side C++ file that manages
 * the hash table lifecycle, memory, and kernel launches.
 */

/**
 * @brief Computes a hash value for a given integer key on the device.
 * @param key The integer key to hash.
 * @param HMAX A device pointer to an integer holding the maximum size of the hash table (the capacity).
 * @return An integer hash value within the range [0, HMAX-1].
 *
 * @details The hash function uses a multiplicative hashing scheme with large prime numbers
 * to distribute keys. The result is taken modulo the table capacity.
 */
__device__ int hash_func(int key, int *HMAX) {
	// Hash Logic: Multiplicative hashing with large primes, followed by modulo table size.
	return ((long)abs(key) * 26339969) % 6743036717 % *HMAX;
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param hash_keys Device pointer to the array storing hash table keys.
 * @param hash_values Device pointer to the array storing hash table values.
 * @param empty_slots Device pointer to a bitmap indicating whether a slot is occupied.
 * @param gpu_keys Device pointer to the input keys for this batch insertion.
 * @param gpu_values Device pointer to the input values for this batch insertion.
 * @param size Device pointer to an integer storing the current number of elements in the table.
 * @param HMAX Device pointer to an integer holding the maximum size of the hash table.
 *
 * @details Each thread in the grid processes one key-value pair. The kernel uses an open-addressing
 * strategy with linear probing to resolve hash collisions. Atomic operations (`atomicExch`, `atomicAdd`)
 * are used to ensure thread-safe updates to the shared hash table data structures.
 *
 * Algorithm: Linear Probing with Atomic Slot Acquisition.
 * Time Complexity: Average O(1), Worst Case O(N) per insertion, where N is HMAX.
 */
__global__ void insert(int *hash_keys, int *hash_values, int *empty_slots,
                        int *gpu_keys, int *gpu_values, int *size, int *HMAX)
{
	
	// Thread Indexing: Calculate a unique global thread ID for a 1D grid.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	
	// Data Loading: Each thread fetches its assigned key-value pair from global memory.
	int key = gpu_keys[i];
	int value = gpu_values[i];

	
	// Pre-condition: Ignore invalid key-value pairs (non-positive).
	if (key <= 0 || value <= 0) {
		return;
	}
						
	
	// Hashing: Compute the initial bucket index for the key.
	int hash_key = hash_func(key, HMAX);
	int index = hash_key;

	
	// Atomic Slot Acquisition: Attempt to claim the initial bucket.
	int elem = atomicExch(&empty_slots[index], 1);
	
	// Block Logic: If the slot was free (elem == 0), the thread claims it.
	if (elem == 0) {
		// Invariant: This thread now has exclusive write access to this slot.
		hash_keys[index] = key;
		hash_values[index] = value;
		// Atomically increment the total element count.
		atomicAdd(size, 1);
		return;
	}
	// Inline: If the slot was already taken, restore its state. This is a defensive
	// measure in a race-prone scenario, though `atomicCAS` would be a more typical pattern.
	atomicExch(&empty_slots[index], elem);
	
	// Pre-condition: Check if the key in the occupied slot matches the current key.
	if (hash_keys[index] == key) {
		// Invariant: Key already exists, so update its value atomically.
		atomicExch(&hash_values[index], value);
		return;
	}
	
	// Collision Resolution: Start linear probing from the next slot.
	index = (index + 1) % *HMAX;
	// Block Logic: Probe until an empty slot is found or we circle back to the start.
	while (index != hash_key) {
		// Atomic Slot Acquisition: Attempt to claim the probed bucket.
		elem = atomicExch(&empty_slots[index], 1);
		
		// Block Logic: If the slot was free (elem == 0), the thread claims it.
		if (elem == 0) {
			// Invariant: This thread now has exclusive write access to this slot.
			hash_keys[index] = key;
			hash_values[index] = value;
			// Atomically increment the total element count.
			atomicAdd(size, 1);
			return;
		}
		// Inline: If the slot was already taken, restore its state.
		atomicExch(&empty_slots[index], elem);
		
		// Pre-condition: Check if the key in the occupied slot matches the current key.
		if (hash_keys[index] == key) {
			// Invariant: Key already exists, so update its value atomically.
			atomicExch(&hash_values[index], value);
			return;
		}
		// Move to the next bucket in the probe sequence.
		index = (index + 1) % *HMAX;
	}
	
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys from the hash table.
 * @param hash_keys Device pointer to the array storing hash table keys.
 * @param hash_values Device pointer to the array storing hash table values.
 * @param empty_slots Device pointer to a bitmap indicating whether a slot is occupied.
 * @param keys Device pointer to the input keys for this batch lookup.
 * @param values Device pointer to the output array where found values will be stored.
 * @param HMAX Device pointer to an integer holding the maximum size of the hash table.
 *
 * @details Each thread in the grid processes one key lookup. It follows the same linear
 * probing path as the `insert` kernel to locate the key. If the key is found, the
 * corresponding value is written to the output array. No atomic operations are needed
 * for reading, as the hash table structure is considered stable during lookups (no deletions).
 *
 * Algorithm: Linear Probing Search.
 * Time Complexity: Average O(1), Worst Case O(N) per lookup, where N is HMAX.
 */
__global__ void get_val(int *hash_keys, int *hash_values, int *empty_slots,
                        int *keys, int* values, int *HMAX)
{
	
	// Thread Indexing: Calculate a unique global thread ID for a 1D grid.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int key = keys[i]; 
						
	
	// Hashing: Compute the initial bucket index for the key.
	int hash_key = hash_func(key, HMAX);
	int index = hash_key;

	
	// Block Logic: Probe sequentially as long as slots are marked as occupied.
	// Invariant: The key, if it exists, must be in this chain of occupied slots.
	while (empty_slots[index] == 1) {

		
		// Pre-condition: Check if the key at the current index matches the search key.
		if (hash_keys[index] == key) {
			// Invariant: Key found. Copy the value to the output array.
			values[i] = hash_values[index];
			return;
		}
		// Move to the next bucket in the probe sequence.
		index = (index + 1) % *HMAX;
		
		// Pre-condition: If we have probed the entire table and returned to the
		// starting hash_key, the key is not in the table.
		if (index == hash_key) {
			break;
		}
	}
}
