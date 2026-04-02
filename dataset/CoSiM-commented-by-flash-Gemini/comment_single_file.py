
import sys
import os

def generate_commented_code(file_path):
    try:
        with open(file_path, 'r') as f:
            original_code = f.readlines() # Read lines into a list
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

    modified_lines = []

    # Module-level comment (add unconditionally at the beginning)
    modified_lines.append("""/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA, designed for high-throughput key-value storage and retrieval.
 *
 * This file provides the core CUDA kernels and host-side functions for managing a hash table
 * on NVIDIA GPUs. It utilizes a linear probing strategy for collision resolution and
 * atomic operations to ensure correctness in parallel insertion scenarios.
 *
 * Domain: High-Performance Computing (HPC), GPU Programming, Data Structures, Concurrent Algorithms.
 */
""")

    line_idx = 0
    # Skip any initial blank lines before processing the actual code
    while line_idx < len(original_code) and len(original_code[line_idx].strip()) == 0:
        modified_lines.append(original_code[line_idx])
        line_idx += 1

    while line_idx < len(original_code):
        line = original_code[line_idx]

        # Handle global variable declarations and their comments
        if "int mapSize;" in line and "gpu_hashtable.hpp" in original_code[line_idx-1 if line_idx > 0 else 0]:
            modified_lines.append(line)
            modified_lines.append("// @brief Pointer to the number of elements currently stored in the hash map, managed on the device.
")
            modified_lines.append(original_code[line_idx+1]) # int *currentNo;
            modified_lines.append("// @brief Device-side pointer to a list of prime numbers used for hash function computations.
")
            modified_lines.append(original_code[line_idx+2]) # size_t *copyList = NULL;
            modified_lines.append("// @brief Device-side pointer to the actual hash map structure, storing key-value pairs.
")
            modified_lines.append(original_code[line_idx+3]) # struct realMap *hashMap = NULL;
            modified_lines.append("
") # Add a blank line for spacing
            line_idx += 4
        
        # kernel_get
        elif "__global__ void kernel_get" in line:
            modified_lines.append("""
/**
 * @brief CUDA kernel for retrieving values associated with given keys in parallel.
 *
 * This kernel iterates through a batch of keys, computes their hash indices, and
 * retrieves corresponding values from the GPU hash map. It handles collisions
 * using linear probing.
 *
 * @param keys Device-side array of keys to search for.
 * @param numKeys The number of keys to search.
 * @param myMap The GPU hash map structure.
 * @param values Device-side array to store the retrieved values.
 * @param num The total number of buckets in `myMap`.
 * @param list Device-side array of prime numbers for hashing.
 */
""")
            modified_lines.append(line)
            line_idx += 1
            while line_idx < len(original_code) and "}" not in original_code[line_idx]:
                current_line = original_code[line_idx]
                if "unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;" in current_line:
                    modified_lines.append("	// Functional Utility: Computes a unique global thread index.
")
                    modified_lines.append("	// Invariant: `i` correctly maps to a unique key in the `keys` array.
")
                    modified_lines.append(current_line)
                elif "if(i >= numKeys)" in current_line:
                    modified_lines.append("
	// Block Logic: Ensures that each thread processes a valid key within the bounds of the input array.
")
                    modified_lines.append("	// Pre-condition: `i` is the global thread index.
")
                    modified_lines.append("	// Invariant: Only threads with `i < numKeys` proceed to process a key.
")
                    modified_lines.append(current_line)
                elif "int index = hash1(keys[i], num, list);" in current_line:
                    modified_lines.append("
	// Functional Utility: Computes the initial hash index for the current key using `hash1`.
")
                    modified_lines.append("	// Invariant: `index` holds a valid initial hash table index.
")
                    modified_lines.append(current_line)
                elif "while(1)" in current_line:
                    modified_lines.append("
	/**
")
                    modified_lines.append("	 * Block Logic: Implements linear probing to find the key in the hash table.
")
                    modified_lines.append("	 * Pre-condition: `index` is the initial hash probe location.
")
                    modified_lines.append("	 * Invariant: The loop continues until the key is found or the probe wraps around.
")
                    modified_lines.append("	 */
")
                    modified_lines.append(current_line)
                elif "if(myMap[index].key == keys[i])" in current_line:
                    modified_lines.append("
		// Block Logic: Checks if the current hash map entry matches the target key.
")
                    modified_lines.append("		// Pre-condition: `myMap[index]` contains an entry to check.
")
                    modified_lines.append("		// Invariant: If a match is found, the value is retrieved, and the loop terminates.
")
                    modified_lines.append(current_line)
                elif "index ++;" in current_line:
                    modified_lines.append("
			// Block Logic: Moves to the next bucket in case of a collision or no match.
")
                    modified_lines.append("			// Invariant: `index` is incremented, and wraps around if it exceeds `num`.
")
                    modified_lines.append(current_line)
                else:
                    modified_lines.append(current_line)
                line_idx += 1
            modified_lines.append(original_code[line_idx]) # Append closing brace
            line_idx += 1

        # kernel_insert
        elif "__global__ void kernel_insert" in line:
            modified_lines.append("""
/**
 * @brief CUDA kernel for inserting key-value pairs into the hash table in parallel.
 *
 * This kernel processes a batch of key-value pairs, computes their hash indices,
 * and inserts them into the GPU hash table. It uses linear probing for collision
 * resolution and atomic operations to ensure thread-safe insertions.
 *
 * @param keys Device-side array of keys to insert.
 * @param numKeys The number of keys to insert.
 * @param values Device-side array of values corresponding to the keys.
 * @param myMap The GPU hash map structure.
 * @param num The total number of buckets in `myMap`.
 * @param list Device-side array of prime numbers for hashing.
 * @param current Device-side pointer to an integer tracking the current number of elements.
 */
""")
            modified_lines.append(line)
            line_idx += 1
            while line_idx < len(original_code) and "}" not in original_code[line_idx]:
                current_line = original_code[line_idx]
                if "unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;" in current_line:
                    modified_lines.append("	// Functional Utility: Computes a unique global thread index.
")
                    modified_lines.append("	// Invariant: `i` correctly maps to a unique key in the `keys` array.
")
                    modified_lines.append("	// Inline: threadIdx.x is the thread ID within the block, blockDim.x is the number of threads per block,
")
                    modified_lines.append("	// blockIdx.x is the block ID within the grid. This calculation yields a global thread index.
")
                    modified_lines.append(current_line)
                elif "if(i >= numKeys)" in current_line:
                    modified_lines.append("
	// Block Logic: Ensures that each thread processes a valid key within the bounds of the input array.
")
                    modified_lines.append("	// Pre-condition: `i` is the global thread index.
")
                    modified_lines.append("	// Invariant: Only threads with `i < numKeys` proceed to process a key.
")
                    modified_lines.append(current_line)
                elif "int index = hash1(keys[i], num, list);" in current_line:
                    modified_lines.append("
	// Functional Utility: Computes the initial hash index for the current key using `hash1`.
")
                    modified_lines.append("	// Invariant: `index` holds a valid initial hash table index.
")
                    modified_lines.append(current_line)
                elif "while(1)" in current_line:
                    modified_lines.append("
	/**
")
                    modified_lines.append("	 * Block Logic: Implements linear probing for insertion, handling potential collisions.
")
                    modified_lines.append("	 * Invariant: The loop continues until the key is inserted or an existing key's value is updated.
")
                    modified_lines.append("	 */
")
                    modified_lines.append(current_line)
                elif "if(myMap[index].key == keys[i])" in current_line:
                    modified_lines.append("
		// Block Logic: Checks if the current hash map entry already contains the key.
")
                    modified_lines.append("		// Pre-condition: `myMap[index]` contains an entry to check.
")
                    modified_lines.append("		// Invariant: If the key exists, its value is atomically updated, and the function returns.
")
                    modified_lines.append(current_line)
                    modified_lines.append("			// Inline: Atomically exchanges the value at `myMap[index].value` with `values[i]`.
")
                    modified_lines.append("			// This ensures thread-safe updates to existing entries in a parallel environment.
")
                elif "int result = atomicCAS(&myMap[index].key, 0, keys[i]);" in current_line:
                    modified_lines.append("		// Functional Utility: Attempts to atomically set the key in an empty bucket (key == 0).
")
                    modified_lines.append("		// Invariant: `result` will be 0 if the key was successfully set, otherwise it will be the original value of `myMap[index].key`.
")
                    modified_lines.append("		// This uses Compare-And-Swap to ensure only one thread claims an empty bucket.
")
                    modified_lines.append(current_line)
                elif "if(result == 0)" in current_line:
                    modified_lines.append("
		// Block Logic: If `atomicCAS` successfully set the key (bucket was empty).
")
                    modified_lines.append("		// Pre-condition: `result` is 0, indicating the key was inserted into an empty slot.
")
                    modified_lines.append("		// Invariant: The value is then atomically inserted, and `currentNo` is atomically incremented.
")
                    modified_lines.append(current_line)
                    modified_lines.append("			// Inline: Atomically exchanges the value at `myMap[index].value` with `values[i]`.
")
                    modified_lines.append(original_code[line_idx+1]) # atomicExch
                    modified_lines.append("			// Inline: Atomically increments `current` if it's not NULL, counting the number of elements.
")
                    modified_lines.append(original_code[line_idx+2]) # if(current != NULL)
                    line_idx += 2 # Skip atomicExch and if(current != NULL)
                elif "else {" in current_line:
                    modified_lines.append("
		else {
")
                    modified_lines.append("			// Block Logic: Continues linear probing if the current bucket is occupied by another key.
")
                    modified_lines.append("			// Invariant: `index` is incremented, and wraps around if it exceeds `num`.
")
                    modified_lines.append(original_code[line_idx+1]) # index ++;
                    modified_lines.append(original_code[line_idx+2]) # if(index >= num)
                    modified_lines.append(original_code[line_idx+3]) # index = 0;
                    modified_lines.append(original_code[line_idx+4]) # }
                    line_idx += 4 # Skip else block lines
                else:
                    modified_lines.append(current_line)
                line_idx += 1
            modified_lines.append(original_code[line_idx]) # Append closing brace
            line_idx += 1
        
        elif "GpuHashTable::GpuHashTable(int size)" in line:
            modified_lines.append("""
/**
 * @brief Constructor for the GpuHashTable class.
 *
 * Initializes the GPU hash table by allocating managed memory for the hash map
 * itself, a counter for current elements, and a list of primes for hashing.
 * It also initializes all hash map entries to a default empty state.
 *
 * @param size The initial desired number of buckets for the hash table.
 */
""")
            modified_lines.append(line)
            modified_lines.append(original_code[line_idx+1]) # {
            modified_lines.append("	// Functional Utility: Allocates unified memory for the hash map structure, accessible by both host and device.
")
            modified_lines.append(original_code[line_idx+2]) # cudaMallocManaged(&hashMap, size * sizeof(struct realMap));
            modified_lines.append("	// Functional Utility: Synchronizes the device, ensuring all previous CUDA calls complete before proceeding.
")
            modified_lines.append(original_code[line_idx+3]) # cudaDeviceSynchronize();
            modified_lines.append("
	// Functional Utility: Error handling macro to check for failed memory allocation.
")
            modified_lines.append(original_code[line_idx+4]) # DIE(hashMap == NULL, "allocation for hashMap failed");
            modified_lines.append("	// Functional Utility: Sets the initial size of the hash map.
")
            modified_lines.append(original_code[line_idx+5]) # mapSize = size;
            modified_lines.append("
	// Functional Utility: Allocates unified memory for the current element count.
")
            modified_lines.append(original_code[line_idx+6]) # cudaMallocManaged(&currentNo, sizeof(int));
            modified_lines.append("	// Functional Utility: Synchronizes the device.
")
            modified_lines.append(original_code[line_idx+7]) # cudaDeviceSynchronize();
            modified_lines.append("	// Functional Utility: Error handling macro for memory allocation.
")
            modified_lines.append(original_code[line_idx+8]) # DIE(currentNo == NULL, "allocation for currentNo failed");
            modified_lines.append("
	// Block Logic: Initializes all hash map entries with key and value set to 0, representing empty buckets.
")
            modified_lines.append("	// Pre-condition: `hashMap` is allocated and `mapSize` is set.
")
            modified_lines.append("	// Invariant: All buckets in `hashMap` are initialized to an empty state.
")
            modified_lines.append(original_code[line_idx+9]) # for(int i = 0; i < size; i ++) {
            modified_lines.append(original_code[line_idx+10]) # 	hashMap[i].key = 0;
            modified_lines.append(original_code[line_idx+11]) # 	hashMap[i].value = 0;
            modified_lines.append(original_code[line_idx+12]) # }
            modified_lines.append("
	// Functional Utility: Allocates unified memory for the prime number list used in hashing.
")
            modified_lines.append(original_code[line_idx+13]) # cudaMallocManaged(&copyList, count() * sizeof(size_t));
            modified_lines.append("	// Functional Utility: Synchronizes the device.
")
            modified_lines.append(original_code[line_idx+14]) # cudaDeviceSynchronize();
            modified_lines.append("
	// Functional Utility: Copies the host-side `primeList` to the device-side `copyList`.
")
            modified_lines.append(original_code[line_idx+15]) # cudaMemcpy(copyList, primeList, count() * sizeof(size_t), cudaMemcpyHostToDevice);
            modified_lines.append(original_code[line_idx+16]) # }
            line_idx += 17

        elif "GpuHashTable::~GpuHashTable()" in line:
            modified_lines.append("""
/**
 * @brief Destructor for the GpuHashTable class.
 *
 * Frees all CUDA managed memory allocated for the hash table components
 * to prevent memory leaks.
 */
""")
            modified_lines.append(line)
            modified_lines.append(original_code[line_idx+1]) # {
            modified_lines.append("	// Functional Utility: Frees the managed memory allocated for the hash map.
")
            modified_lines.append(original_code[line_idx+2]) # cudaFree(hashMap);
            modified_lines.append("	// Functional Utility: Frees the managed memory allocated for the current element count.
")
            modified_lines.append(original_code[line_idx+3]) # cudaFree(currentNo);
            modified_lines.append("	// Functional Utility: Frees the managed memory allocated for the prime number list.
")
            modified_lines.append(original_code[line_idx+4]) # cudaFree(copyList);
            modified_lines.append(original_code[line_idx+5]) # }
            line_idx += 6
        
        elif "void GpuHashTable::reshape(int numBucketsReshape)" in line:
            modified_lines.append("""
/**
 * @brief Reshapes (resizes) the GPU hash table to a new number of buckets.
 *
 * This function creates a new, larger hash table, rehashes all existing
 * elements from the old table into the new one, and then replaces the old table.
 *
 * @param numBucketsReshape The new desired number of buckets for the hash table.
 */
""")
            modified_lines.append(line)
            modified_lines.append(original_code[line_idx+1]) # {
            modified_lines.append("	// Functional Utility: Declares a temporary pointer for the new hash map.
")
            modified_lines.append(original_code[line_idx+2]) # struct realMap *hashMapCopy = 0;
            modified_lines.append("	// Functional Utility: Allocates unified memory for the new hash map with the resized number of buckets.
")
            modified_lines.append(original_code[line_idx+3]) # cudaMallocManaged(&hashMapCopy, numBucketsReshape * sizeof(struct realMap));
            modified_lines.append("	// Functional Utility: Synchronizes the device.
")
            modified_lines.append(original_code[line_idx+4]) # cudaDeviceSynchronize();
            modified_lines.append("
	// Functional Utility: Declares a temporary pointer for copying keys.
")
            modified_lines.append(original_code[line_idx+5]) # int *keyCopy;
            modified_lines.append("	// Functional Utility: Allocates unified memory for storing keys during the reshape process.
")
            modified_lines.append(original_code[line_idx+6]) # cudaMallocManaged(&keyCopy, *currentNo * sizeof(int));
            modified_lines.append("	// Functional Utility: Synchronizes the device.
")
            modified_lines.append(original_code[line_idx+7]) # cudaDeviceSynchronize();
            modified_lines.append("
	// Functional Utility: Declares a temporary pointer for copying values.
")
            modified_lines.append(original_code[line_idx+8]) # int *valuesCopy;
            modified_lines.append("	// Functional Utility: Allocates unified memory for storing values during the reshape process.
")
            modified_lines.append(original_code[line_idx+9]) # cudaMallocManaged(&valuesCopy, *currentNo * sizeof(int));
            modified_lines.append("	// Functional Utility: Synchronizes the device.
")
            modified_lines.append(original_code[line_idx+10]) # cudaDeviceSynchronize();
            modified_lines.append("
	// Block Logic: Initializes all entries in the new hash map to an empty state (key and value 0).
")
            modified_lines.append("	// Pre-condition: `hashMapCopy` is allocated.
")
            modified_lines.append("	// Invariant: All buckets in the new hash map are ready for new insertions.
")
            modified_lines.append(original_code[line_idx+11]) # for(int i = 0; i < numBucketsReshape; i ++) {
            modified_lines.append(original_code[line_idx+12]) # 	hashMapCopy[i].key = 0;
            modified_lines.append(original_code[line_idx+13]) # 	hashMapCopy[i].value = 0;
            modified_lines.append(original_code[line_idx+14]) # }
            modified_lines.append("
	// Functional Utility: Counter for valid keys found in the old hash map.
")
            modified_lines.append(original_code[line_idx+15]) # int x = 0;
            modified_lines.append(original_code[line_idx+16]) # 
            modified_lines.append(original_code[line_idx+17]) # 
            modified_lines.append("	// Block Logic: Iterates through the old hash map to extract existing key-value pairs.
")
            modified_lines.append("	// Pre-condition: `hashMap` contains existing data.
")
            modified_lines.append("	// Invariant: `keyCopy` and `valuesCopy` will contain all valid key-value pairs from `hashMap`.
")
            modified_lines.append(original_code[line_idx+18]) # for(int i = 0; i < mapSize; i ++) {
            modified_lines.append("		// Block Logic: Checks if a bucket in the old hash map contains a valid key (not 0).
")
            modified_lines.append("		// Invariant: Valid key-value pairs are copied to temporary arrays.
")
            modified_lines.append(original_code[line_idx+19]) # 	if(hashMap[i].key != 0) {
            modified_lines.append(original_code[line_idx+20]) # 		keyCopy[x] = hashMap[i].key;
            modified_lines.append(original_code[line_idx+21]) # 		valuesCopy[x] = hashMap[i].value;
            modified_lines.append(original_code[line_idx+22]) # 		x ++;
            modified_lines.append(original_code[line_idx+23]) # 	}
            modified_lines.append(original_code[line_idx+24]) # }
            modified_lines.append("
	// Functional Utility: Launches the `kernel_insert` to rehash all copied key-value pairs into the new hash map.
")
            modified_lines.append("	// This implicitly handles collision resolution and places elements in their new locations.
")
            modified_lines.append("	// Inline: The kernel is launched with a grid size calculated to ensure all 'x' elements are processed.
")
            modified_lines.append("	// Each block has 256 threads.
")
            modified_lines.append(original_code[line_idx+25]) # kernel_insert<<< (x + 255) / 256, 256 >>>(keyCopy, x, valuesCopy, hashMapCopy,
            modified_lines.append(original_code[line_idx+26]) # 	numBucketsReshape, copyList, NULL);
            modified_lines.append("	// Functional Utility: Synchronizes the device to ensure all insertions are complete.
")
            modified_lines.append(original_code[line_idx+27]) # cudaDeviceSynchronize();
            modified_lines.append("
	// Functional Utility: Frees the memory of the old hash map.
")
            modified_lines.append(original_code[line_idx+28]) # cudaFree(hashMap);
            modified_lines.append("
	// Functional Utility: Updates the `hashMap` pointer to point to the newly created and populated hash map.
")
            modified_lines.append(original_code[line_idx+29]) # hashMap = hashMapCopy;
            modified_lines.append("	// Functional Utility: Updates `mapSize` to reflect the new size of the hash table.
")
            modified_lines.append(original_code[line_idx+30]) # mapSize = numBucketsReshape;
            modified_lines.append(original_code[line_idx+31]) # }
            line_idx += 32

        elif "bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys)" in line:
            modified_lines.append("""
/**
 * @brief Inserts a batch of key-value pairs into the GPU hash table.
 *
 * This function manages memory allocation for the input data, performs
 * a reshape operation if the hash table is too full, and then launches
 * the `kernel_insert` to perform the parallel insertions.
 *
 * @param keys Host-side array of keys to insert.
 * @param values Host-side array of values to insert.
 * @param numKeys The number of key-value pairs to insert.
 * @return Always returns `false`. (Might indicate a placeholder or unused return value)
 */
""")
            modified_lines.append(line)
            modified_lines.append(original_code[line_idx+1]) # {
            modified_lines.append("
	// Block Logic: Checks if the current capacity is sufficient for the new batch of insertions.
")
            modified_lines.append("	// Pre-condition: `*currentNo` is the current count, `numKeys` is the batch size, `mapSize` is current capacity.
")
            modified_lines.append("	// Invariant: If capacity is insufficient, the hash table is reshaped to a larger size.
")
            modified_lines.append(original_code[line_idx+2]) # if(*currentNo + numKeys >= mapSize) {
            modified_lines.append("		// Functional Utility: Calculates a new map size with a 5% overhead.
")
            modified_lines.append(original_code[line_idx+3]) # 	reshape(*currentNo + numKeys + (*currentNo + numKeys) * 5 / 100);
            modified_lines.append("		// Functional Utility: Updates the `mapSize` global variable.
")
            modified_lines.append(original_code[line_idx+4]) # 	mapSize = *currentNo + numKeys + (*currentNo + numKeys) * 5 / 100;
            modified_lines.append(original_code[line_idx+5]) # }
            modified_lines.append("
	// Functional Utility: Allocates unified memory for a copy of the input keys.
")
            modified_lines.append(original_code[line_idx+6]) # int *keyCopy;
            modified_lines.append(original_code[line_idx+7]) # cudaMallocManaged(&keyCopy, numKeys * sizeof(int));
            modified_lines.append(original_code[line_idx+8]) # cudaDeviceSynchronize();
            modified_lines.append("
	// Functional Utility: Allocates unified memory for a copy of the input values.
")
            modified_lines.append(original_code[line_idx+9]) # int *valuesCopy;
            modified_lines.append(original_code[line_idx+10]) # cudaMallocManaged(&valuesCopy, numKeys * sizeof(int));
            modified_lines.append(original_code[line_idx+11]) # cudaDeviceSynchronize();
            modified_lines.append("
	// Functional Utility: Copies host-side `keys` data to device-side `keyCopy`.
")
            modified_lines.append(original_code[line_idx+12]) # memcpy(keyCopy, keys, numKeys * sizeof(int));
            modified_lines.append("	// Functional Utility: Copies host-side `values` data to device-side `valuesCopy`.
")
            modified_lines.append(original_code[line_idx+13]) # memcpy(valuesCopy, values, numKeys * sizeof(int));
            modified_lines.append("
	// Functional Utility: Launches the `kernel_insert` to perform parallel insertions using the copied data.
")
            modified_lines.append("	// It passes `currentNo` for atomic updates to the element count.
")
            modified_lines.append("	// Inline: The kernel is launched with a grid size calculated to ensure all `numKeys` elements are processed.
")
            modified_lines.append("	// Each block has 256 threads.
")
            modified_lines.append(original_code[line_idx+14]) # kernel_insert<<< (numKeys + 255) / 256, 256 >>>(keyCopy, numKeys, valuesCopy,
            modified_lines.append(original_code[line_idx+15]) # 	hashMap, mapSize, copyList, currentNo);
            modified_lines.append(original_code[line_idx+16]) # cudaDeviceSynchronize();
            modified_lines.append("
	// Functional Utility: Placeholder return value.
")
            modified_lines.append(original_code[line_idx+17]) # return false;
            modified_lines.append(original_code[line_idx+18]) # }
            line_idx += 19

        elif "int* GpuHashTable::getBatch(int* keys, int numKeys)" in line:
            modified_lines.append("""
/**
 * @brief Retrieves a batch of values for given keys from the GPU hash table.
 *
 * This function allocates device memory for the results and input key copies,
 * then launches the `kernel_get` to perform the parallel retrieval.
 *
 * @param keys Host-side array of keys to retrieve values for.
 * @param numKeys The number of keys to retrieve.
 * @return A device-side pointer to an array containing the retrieved values.
 */
""")
            modified_lines.append(line)
            modified_lines.append(original_code[line_idx+1]) # {
            modified_lines.append("
	// Functional Utility: Allocates unified memory for storing the results (retrieved values).
")
            modified_lines.append(original_code[line_idx+2]) # int* results = NULL;
            modified_lines.append(original_code[line_idx+3]) # cudaMallocManaged(&results, numKeys * sizeof(int));
            modified_lines.append(original_code[line_idx+4]) # cudaDeviceSynchronize();
            modified_lines.append("
	// Functional Utility: Allocates unified memory for a copy of the input keys.
")
            modified_lines.append(original_code[line_idx+5]) # int *keyCopy;
            modified_lines.append(original_code[line_idx+6]) # cudaMallocManaged(&keyCopy, numKeys * sizeof(int));
            modified_lines.append(original_code[line_idx+7]) # cudaDeviceSynchronize();
            modified_lines.append("
	// Functional Utility: Copies host-side `keys` data to device-side `keyCopy`.
")
            modified_lines.append(original_code[line_idx+8]) # memcpy(keyCopy, keys, numKeys * sizeof(int));
            modified_lines.append("
	// Functional Utility: Launches the `kernel_get` to perform parallel retrieval.
")
            modified_lines.append("	// Inline: The kernel is launched with a grid size calculated to ensure all `numKeys` elements are processed.
")
            modified_lines.append("	// Each block has 256 threads.
")
            modified_lines.append(original_code[line_idx+9]) # kernel_get<<< (numKeys + 255) / 256, 256 >>>(keyCopy, numKeys, hashMap,
            modified_lines.append(original_code[line_idx+10]) # 	results, mapSize, copyList);
            modified_lines.append(original_code[line_idx+11]) # cudaDeviceSynchronize();
            modified_lines.append("
	// Functional Utility: Returns the device-side pointer to the results.
")
            modified_lines.append(original_code[line_idx+12]) # return results;
            modified_lines.append(original_code[line_idx+13]) # }
            line_idx += 14

        elif "float GpuHashTable::loadFactor()" in line:
            modified_lines.append("""
/**
 * @brief Calculates the current load factor of the hash table.
 *
 * The load factor is the ratio of the number of elements in the hash table
 * to the total number of available buckets.
 *
 * @return The current load factor as a float.
 */
""")
            modified_lines.append(line)
            modified_lines.append(original_code[line_idx+1]) # {
            modified_lines.append("	
")
            modified_lines.append("	// Functional Utility: Computes the load factor, casting to float for accurate division.
")
            modified_lines.append(original_code[line_idx+2]) # return (float)(*currentNo) / (float)(mapSize);
            modified_lines.append(original_code[line_idx+3]) # }
            line_idx += 4

        elif "#define HASH_INIT" in line:
            modified_lines.append("
// @brief Macro to initialize the GpuHashTable with an initial size of 1.
")
            modified_lines.append(line)
            modified_lines.append("// @brief Macro to reshape the GpuHashTable to a specified size.
")
            modified_lines.append(original_code[line_idx+1]) # #define HASH_RESERVE(size)
            modified_lines.append("
// @brief Macro to insert a batch of key-value pairs into the GpuHashTable.
")
            modified_lines.append(original_code[line_idx+2]) # #define HASH_BATCH_INSERT(keys, values, numKeys)
            modified_lines.append("// @brief Macro to retrieve a batch of values for given keys from the GpuHashTable.
")
            modified_lines.append(original_code[line_idx+3]) # #define HASH_BATCH_GET(keys, numKeys)
            modified_lines.append("
// @brief Macro to get the current load factor of the GpuHashTable.
")
            modified_lines.append(original_code[line_idx+4]) # #define HASH_LOAD_FACTOR
            line_idx += 5
        
        elif '#include "test_map.cpp"' in line:
            modified_lines.append(line.strip() + " // Functional Utility: Includes a test map implementation.
")
            line_idx += 1
            while line_idx < len(original_code) and (original_code[line_idx].strip().startswith("#include") or len(original_code[line_idx].strip()) == 0):
                if "#include <unistd.h>" in original_code[line_idx]:
                    modified_lines.append(original_code[line_idx].strip() + " // Functional Utility: Standard POSIX header for system calls like `access`.
")
                elif "#include <sys/types.h>" in original_code[line_idx]:
                    modified_lines.append(original_code[line_idx].strip() + " // Functional Utility: Standard header for fundamental system data types.
")
                elif "#include <sys/stat.h>" in original_code[line_idx]:
                    modified_lines.append(original_code[line_idx].strip() + " // Functional Utility: Standard header for file status information.
")
                elif "#include <fcntl.h>" in original_code[line_idx]:
                    modified_lines.append(original_code[line_idx].strip() + " // Functional Utility: Standard header for file control options.
")
                else:
                    modified_lines.append(original_code[line_idx])
                line_idx += 1

        elif "#ifndef _HASHCPU_" in line:
            modified_lines.append(line)
            modified_lines.append(original_code[line_idx+1]) # #define _HASHCPU_
            modified_lines.append(original_code[line_idx+2]) # (blank line)
            modified_lines.append(original_code[line_idx+3]) # using namespace std;
            modified_lines.append(original_code[line_idx+4]) # (blank line)
            modified_lines.append("// @brief Defines an invalid key value, typically used to mark empty hash table slots.
")
            modified_lines.append(original_code[line_idx+5]) # #define	KEY_INVALID		0
            modified_lines.append("""
/**
 * @brief Error handling macro that prints an error message and exits if an assertion fails.
 *
 * This macro is used for critical error checking, often for memory allocations
 * or other operations where failure is unrecoverable.
 *
 * @param assertion A boolean expression that, if true, triggers the error handling.
 * @param call_description A string describing the failed call or operation.
 */
""")
            modified_lines.append(original_code[line_idx+6]) # #define DIE(assertion, call_description) 
            modified_lines.append(original_code[line_idx+7]) # 	do {	
            modified_lines.append(original_code[line_idx+8]) # 		if (assertion) {	
            modified_lines.append(original_code[line_idx+9]) # 		fprintf(stderr, "(%s, %d): ",	
            modified_lines.append(original_code[line_idx+10]) # 		__FILE__, __LINE__);	
            modified_lines.append(original_code[line_idx+11]) # 		perror(call_description);	
            modified_lines.append(original_code[line_idx+12]) # 		exit(errno);	
            modified_lines.append(original_code[line_idx+13]) # 	}	
            modified_lines.append(original_code[line_idx+14]) # } while (0)
            modified_lines.append("	
")
            modified_lines.append("""/**
 * @brief A constant array of large prime numbers.
 *
 * These prime numbers are utilized in the hash functions to minimize collisions
 * and distribute keys evenly across the hash table, thereby improving performance.
 * The use of `llu` (unsigned long long) ensures these large numbers are handled correctly.
 */
""")
            modified_lines.append(original_code[line_idx+15]) # const size_t primeList[] =
            
            line_idx += 16
            while line_idx < len(original_code) and "};" not in original_code[line_idx]:
                modified_lines.append(original_code[line_idx])
                line_idx += 1
            modified_lines.append(original_code[line_idx]) # };
            modified_lines.append("


") 
            line_idx += 1
            
            # struct realMap
            if line_idx < len(original_code) and "struct realMap {" in original_code[line_idx]:
                modified_lines.append("""/**
 * @brief Structure representing a single entry (bucket) in the hash table.
 *
 * Each entry stores an integer key and its corresponding integer value.
 * A key of 0 typically denotes an empty or invalid bucket.
 */
""")
                modified_lines.append(original_code[line_idx]) # struct realMap {
                modified_lines.append(original_code[line_idx+1]) # 	int key;
                modified_lines.append(original_code[line_idx+2]) # 	int value;
                modified_lines.append(original_code[line_idx+3]) # };
                modified_lines.append("


") 
                line_idx += 4
            
            # hash1
            if line_idx < len(original_code) and "__host__ __device__  int hash1(int data, int limit, size_t *primeList1)" in original_code[line_idx]:
                modified_lines.append("""
/**
 * @brief Primary hash function used to compute an initial index for a given key.
 *
 * This function takes an integer `data`, applies a multiplication with a large
 * prime from `primeList`, and then uses modulo operations to ensure the result
 * fits within the `limit` (hash table size). This hash function is designed
 * to work on both host and device (CPU and GPU).
 *
 * @param data The integer key to hash.
 * @param limit The upper bound for the hash index (typically the hash table size).
 * @param primeList1 Device-side array of prime numbers for hashing.
 * @return The computed hash index within the specified limit.
 */
""")
                modified_lines.append(original_code[line_idx]) # __host__ __device__  int hash1(...)
                modified_lines.append(original_code[line_idx+1]) # {
                modified_lines.append(original_code[line_idx+2].strip() + " // Inline: Applies multiplication with a large prime and two modulo operations to calculate the hash.
")
                modified_lines.append(original_code[line_idx+3]) # }
                line_idx += 4

            # hash2
            if line_idx < len(original_code) and "int hash2(int data, int limit)" in original_code[line_idx]:
                modified_lines.append("""
/**
 * @brief Secondary hash function (not explicitly used in kernels, but available).
 *
 * Provides an alternative hash computation. In a double hashing scheme, such
 * functions are used to determine the step size for probing.
 *
 * @param data The integer key to hash.
 * @param limit The upper bound for the hash index.
 * @return The computed hash index.
 */
""")
                modified_lines.append(original_code[line_idx]) # int hash2(...)
                modified_lines.append(original_code[line_idx+1]) # {
                modified_lines.append(original_code[line_idx+2].strip() + " // Inline: Uses different primes from `primeList` for hash calculation.
")
                modified_lines.append(original_code[line_idx+3]) # }
                line_idx += 4

            # hash3
            if line_idx < len(original_code) and "int hash3(int data, int limit)" in original_code[line_idx]:
                modified_lines.append("""
/**
 * @brief Tertiary hash function (not explicitly used in kernels, but available).
 *
 * Provides another alternative hash computation.
 *
 * @param data The integer key to hash.
 * @param limit The upper bound for the hash index.
 * @return The computed hash index.
 */
""")
                modified_lines.append(original_code[line_idx]) # int hash3(...)
                modified_lines.append(original_code[line_idx+1]) # {
                modified_lines.append(original_code[line_idx+2].strip() + " // Inline: Uses different primes from `primeList` for hash calculation.
")
                modified_lines.append(original_code[line_idx+3]) # }
                line_idx += 4

            # count()
            if line_idx < len(original_code) and "int count()" in original_code[line_idx]:
                modified_lines.append("""
/**
 * @brief Counts the number of elements in the global `primeList` array.
 *
 * This function iterates through `primeList` until a null terminator is found.
 * Given `primeList` contains `size_t` (unsigned long long) elements, finding a
 * null terminator (0) might be an unconventional way to determine its length
 * for an array of numbers. It assumes 0 is not a valid prime in the list.
 *
 * @return The number of elements in `primeList`.
 */
""")
                modified_lines.append(original_code[line_idx]) # int count()
                modified_lines.append(original_code[line_idx+1]) # {
                modified_lines.append("	// Block Logic: Iterates through the primeList array until a null (0) element is encountered.
")
                modified_lines.append("	// Invariant: `i` counts the number of valid (non-zero) primes.
")
                modified_lines.append(original_code[line_idx+2]) # 	int i = 0;
                modified_lines.append(original_code[line_idx+3]) # 
                modified_lines.append(original_code[line_idx+4].strip() + " // Inline: Checks for null terminator, likely indicating end of list.
")
                modified_lines.append(original_code[line_idx+5]) # 		i ++;
                modified_lines.append(original_code[line_idx+6]) # 	}
                modified_lines.append("	// Functional Utility: Returns the count of primes plus one (assuming `\0` is not counted as a prime and is an end marker).
")
                modified_lines.append(original_code[line_idx+7]) # 	return i + 1;
                modified_lines.append(original_code[line_idx+8]) # }
                line_idx += 9

            # GpuHashTable class declaration
            if line_idx < len(original_code) and "class GpuHashTable" in original_code[line_idx]:
                modified_lines.append("""

/**
 * @brief Declaration of the GpuHashTable class.
 *
 * This class provides a high-level interface for interacting with the
 * GPU-accelerated hash table, encapsulating the CUDA memory management
 * and kernel launch details.
 */
""")
                modified_lines.append(original_code[line_idx]) # class GpuHashTable
                modified_lines.append(original_code[line_idx+1]) # {
                modified_lines.append(original_code[line_idx+2]) # 	public:
                modified_lines.append("		// @brief Constructor: Initializes the hash table with a given size.
")
                modified_lines.append(original_code[line_idx+3]) # 		GpuHashTable(int size);
                modified_lines.append("		// @brief Reshapes the hash table to a new size.
")
                modified_lines.append(original_code[line_idx+4]) # 		void reshape(int sizeReshape);
                modified_lines.append("		
")
                modified_lines.append("		// @brief Inserts a batch of key-value pairs.
")
                modified_lines.append(original_code[line_idx+5]) # 		bool insertBatch(int *keys, int* values, int numKeys);
                modified_lines.append("		// @brief Retrieves a batch of values for given keys.
")
                modified_lines.append(original_code[line_idx+6]) # 		int* getBatch(int* key, int numItems);
                modified_lines.append("		
")
                modified_lines.append("		// @brief Calculates the current load factor.
")
                modified_lines.append(original_code[line_idx+7]) # 		float loadFactor();
                modified_lines.append("		// @brief Placeholder for occupancy calculation (function not defined in this file).
")
                modified_lines.append(original_code[line_idx+8]) # 		void occupancy();
                modified_lines.append("		// @brief Placeholder for printing hash table information (function not defined in this file).
")
                modified_lines.append(original_code[line_idx+9]) # 		void print(string info);
                modified_lines.append("	
")
                modified_lines.append("		// @brief Destructor: Frees all allocated GPU memory.
")
                modified_lines.append(original_code[line_idx+10]) # 		~GpuHashTable();
                modified_lines.append(original_code[line_idx+11]) # };
                modified_lines.append(original_code[line_idx+12]) # 
                modified_lines.append(original_code[line_idx+13]) # #endif
                line_idx += 14
        
        else:
            modified_lines.append(line)
            line_idx += 1

    return "".join(modified_lines)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python3 comment_single_file.py <file_path>
")
        sys.exit(1)
    
    file_to_comment = sys.argv[1]
    commented_content = generate_commented_code(file_to_comment)
    
    # Print the commented content to stdout
    sys.stdout.write(commented_content)

