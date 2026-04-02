
import os

def generate_commented_code_simplified(original_code):
    output_lines = []

    # Module-level comment
    output_lines.append("""/**
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

    lines = original_code.splitlines(keepends=True)
    
    i = 0
    while i < len(lines):
        line = lines[i]

        if "#include <iostream>" in line and i == 0: # First actual code line, after initial blank line
            output_lines.append(line)
            i += 1
            # Add general includes and global variables comments
            while i < len(lines) and (lines[i].strip().startswith("#include") or len(lines[i].strip()) == 0):
                output_lines.append(lines[i])
                i += 1
            
            # Global variables
            if "int mapSize;" in lines[i]:
                output_lines.append("// @brief The total number of buckets available in the hash map.
")
                output_lines.append(lines[i]) # int mapSize;
                output_lines.append("// @brief Pointer to the number of elements currently stored in the hash map, managed on the device.
")
                output_lines.append(lines[i+1]) # int *currentNo;
                output_lines.append("// @brief Device-side pointer to a list of prime numbers used for hash function computations.
")
                output_lines.append(lines[i+2]) # size_t *copyList = NULL;
                output_lines.append("// @brief Device-side pointer to the actual hash map structure, storing key-value pairs.
")
                output_lines.append(lines[i+3]) # struct realMap *hashMap = NULL;
                output_lines.append("
") # Add a blank line for spacing
                i += 4
            
        elif "__global__ void kernel_get" in line:
            output_lines.append("""
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
            output_lines.append(line)
            i += 1
            # Consume function body and add internal comments
            while i < len(lines) and "}" not in lines[i]:
                current_line = lines[i]
                if "unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;" in current_line:
                    output_lines.append("	// Functional Utility: Computes a unique global thread index.
")
                    output_lines.append("	// Invariant: `i` correctly maps to a unique key in the `keys` array.
")
                    output_lines.append(current_line)
                elif "if(i >= numKeys)" in current_line:
                    output_lines.append("
	// Block Logic: Ensures that each thread processes a valid key within the bounds of the input array.
")
                    output_lines.append("	// Pre-condition: `i` is the global thread index.
")
                    output_lines.append("	// Invariant: Only threads with `i < numKeys` proceed to process a key.
")
                    output_lines.append(current_line)
                elif "int index = hash1(keys[i], num, list);" in current_line:
                    output_lines.append("
	// Functional Utility: Computes the initial hash index for the current key using `hash1`.
")
                    output_lines.append("	// Invariant: `index` holds a valid initial hash table index.
")
                    output_lines.append(current_line)
                elif "while(1)" in current_line:
                    output_lines.append("
	/**
")
                    output_lines.append("	 * Block Logic: Implements linear probing to find the key in the hash table.
")
                    output_lines.append("	 * Pre-condition: `index` is the initial hash probe location.
")
                    output_lines.append("	 * Invariant: The loop continues until the key is found or the probe wraps around.
")
                    output_lines.append("	 */
")
                    output_lines.append(current_line)
                elif "if(myMap[index].key == keys[i])" in current_line:
                    output_lines.append("
		// Block Logic: Checks if the current hash map entry matches the target key.
")
                    output_lines.append("		// Pre-condition: `myMap[index]` contains an entry to check.
")
                    output_lines.append("		// Invariant: If a match is found, the value is retrieved, and the loop terminates.
")
                    output_lines.append(current_line)
                elif "index ++;" in current_line:
                    output_lines.append("
			// Block Logic: Moves to the next bucket in case of a collision or no match.
")
                    output_lines.append("			// Invariant: `index` is incremented, and wraps around if it exceeds `num`.
")
                    output_lines.append(current_line)
                else:
                    output_lines.append(current_line)
                i += 1
            output_lines.append(lines[i]) # Append closing brace
            i += 1

        elif "__global__ void kernel_insert" in line:
            output_lines.append("""
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
            output_lines.append(line)
            i += 1
            while i < len(lines) and "}" not in lines[i]:
                current_line = lines[i]
                if "unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;" in current_line:
                    output_lines.append("	// Functional Utility: Computes a unique global thread index.
")
                    output_lines.append("	// Invariant: `i` correctly maps to a unique key in the `keys` array.
")
                    output_lines.append("	// Inline: threadIdx.x is the thread ID within the block, blockDim.x is the number of threads per block,
")
                    output_lines.append("	// blockIdx.x is the block ID within the grid. This calculation yields a global thread index.
")
                    output_lines.append(current_line)
                elif "if(i >= numKeys)" in current_line:
                    output_lines.append("
	// Block Logic: Ensures that each thread processes a valid key within the bounds of the input array.
")
                    output_lines.append("	// Pre-condition: `i` is the global thread index.
")
                    output_lines.append("	// Invariant: Only threads with `i < numKeys` proceed to process a key.
")
                    output_lines.append(current_line)
                elif "int index = hash1(keys[i], num, list);" in current_line:
                    output_lines.append("
	// Functional Utility: Computes the initial hash index for the current key using `hash1`.
")
                    output_lines.append("	// Invariant: `index` holds a valid initial hash table index.
")
                    output_lines.append(current_line)
                elif "while(1)" in current_line:
                    output_lines.append("
	/**
")
                    output_lines.append("	 * Block Logic: Implements linear probing for insertion, handling potential collisions.
")
                    output_lines.append("	 * Invariant: The loop continues until the key is inserted or an existing key's value is updated.
")
                    output_lines.append("	 */
")
                    output_lines.append(current_line)
                elif "if(myMap[index].key == keys[i])" in current_line:
                    output_lines.append("
		// Block Logic: Checks if the current hash map entry already contains the key.
")
                    output_lines.append("		// Pre-condition: `myMap[index]` contains an entry to check.
")
                    output_lines.append("		// Invariant: If the key exists, its value is atomically updated, and the function returns.
")
                    output_lines.append(current_line)
                    output_lines.append("			// Inline: Atomically exchanges the value at `myMap[index].value` with `values[i]`.
")
                    output_lines.append("			// This ensures thread-safe updates to existing entries in a parallel environment.
")
                elif "int result = atomicCAS(&myMap[index].key, 0, keys[i]);" in current_line:
                    output_lines.append("		// Functional Utility: Attempts to atomically set the key in an empty bucket (key == 0).
")
                    output_lines.append("		// Invariant: `result` will be 0 if the key was successfully set, otherwise it will be the original value of `myMap[index].key`.
")
                    output_lines.append("		// This uses Compare-And-Swap to ensure only one thread claims an empty bucket.
")
                    output_lines.append(current_line)
                elif "if(result == 0)" in current_line:
                    output_lines.append("
		// Block Logic: If `atomicCAS` successfully set the key (bucket was empty).
")
                    output_lines.append("		// Pre-condition: `result` is 0, indicating the key was inserted into an empty slot.
")
                    output_lines.append("		// Invariant: The value is then atomically inserted, and `currentNo` is atomically incremented.
")
                    output_lines.append(current_line)
                    output_lines.append("			// Inline: Atomically exchanges the value at `myMap[index].value` with `values[i]`.
")
                    output_lines.append(lines[i+1]) # atomicExch
                    output_lines.append("			// Inline: Atomically increments `current` if it's not NULL, counting the number of elements.
")
                    output_lines.append(lines[i+2]) # if(current != NULL)
                    i += 2 # Skip atomicExch and if(current != NULL)
                elif "else {" in current_line:
                    output_lines.append("
		else {
")
                    output_lines.append("			// Block Logic: Continues linear probing if the current bucket is occupied by another key.
")
                    output_lines.append("			// Invariant: `index` is incremented, and wraps around if it exceeds `num`.
")
                    output_lines.append(lines[i+1]) # index ++;
                    output_lines.append(lines[i+2]) # if(index >= num)
                    output_lines.append(lines[i+3]) # index = 0;
                    output_lines.append(lines[i+4]) # }
                    i += 4 # Skip else block lines
                else:
                    output_lines.append(current_line)
                i += 1
            output_lines.append(lines[i]) # Append closing brace
            i += 1
        
        elif "GpuHashTable::GpuHashTable(int size)" in line:
            output_lines.append("""
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
            output_lines.append(line)
            output_lines.append(lines[i+1]) # {
            output_lines.append("	// Functional Utility: Allocates unified memory for the hash map structure, accessible by both host and device.
")
            output_lines.append(lines[i+2]) # cudaMallocManaged(&hashMap, size * sizeof(struct realMap));
            output_lines.append("	// Functional Utility: Synchronizes the device, ensuring all previous CUDA calls complete before proceeding.
")
            output_lines.append(lines[i+3]) # cudaDeviceSynchronize();
            output_lines.append("
	// Functional Utility: Error handling macro to check for failed memory allocation.
")
            output_lines.append(lines[i+4]) # DIE(hashMap == NULL, "allocation for hashMap failed");
            output_lines.append("	// Functional Utility: Sets the initial size of the hash map.
")
            output_lines.append(lines[i+5]) # mapSize = size;
            output_lines.append("
	// Functional Utility: Allocates unified memory for the current element count.
")
            output_lines.append(lines[i+6]) # cudaMallocManaged(&currentNo, sizeof(int));
            output_lines.append("	// Functional Utility: Synchronizes the device.
")
            output_lines.append(lines[i+7]) # cudaDeviceSynchronize();
            output_lines.append("	// Functional Utility: Error handling macro for memory allocation.
")
            output_lines.append(lines[i+8]) # DIE(currentNo == NULL, "allocation for currentNo failed");
            output_lines.append("
	// Block Logic: Initializes all hash map entries with key and value set to 0, representing empty buckets.
")
            output_lines.append("	// Pre-condition: `hashMap` is allocated and `mapSize` is set.
")
            output_lines.append("	// Invariant: All buckets in `hashMap` are initialized to an empty state.
")
            output_lines.append(lines[i+9]) # for(int i = 0; i < size; i ++) {
            output_lines.append(lines[i+10]) # 	hashMap[i].key = 0;
            output_lines.append(lines[i+11]) # 	hashMap[i].value = 0;
            output_lines.append(lines[i+12]) # }
            output_lines.append("
	// Functional Utility: Allocates unified memory for the prime number list used in hashing.
")
            output_lines.append(lines[i+13]) # cudaMallocManaged(&copyList, count() * sizeof(size_t));
            output_lines.append("	// Functional Utility: Synchronizes the device.
")
            output_lines.append(lines[i+14]) # cudaDeviceSynchronize();
            output_lines.append("
	// Functional Utility: Copies the host-side `primeList` to the device-side `copyList`.
")
            output_lines.append(lines[i+15]) # cudaMemcpy(copyList, primeList, count() * sizeof(size_t), cudaMemcpyHostToDevice);
            output_lines.append(lines[i+16]) # }
            i += 17

        elif "GpuHashTable::~GpuHashTable()" in line:
            output_lines.append("""
/**
 * @brief Destructor for the GpuHashTable class.
 *
 * Frees all CUDA managed memory allocated for the hash table components
 * to prevent memory leaks.
 */
""")
            output_lines.append(line)
            output_lines.append(lines[i+1]) # {
            output_lines.append("	// Functional Utility: Frees the managed memory allocated for the hash map.
")
            output_lines.append(lines[i+2]) # cudaFree(hashMap);
            output_lines.append("	// Functional Utility: Frees the managed memory allocated for the current element count.
")
            output_lines.append(lines[i+3]) # cudaFree(currentNo);
            output_lines.append("	// Functional Utility: Frees the managed memory allocated for the prime number list.
")
            output_lines.append(lines[i+4]) # cudaFree(copyList);
            output_lines.append(lines[i+5]) # }
            i += 6
        
        elif "void GpuHashTable::reshape(int numBucketsReshape)" in line:
            output_lines.append("""
/**
 * @brief Reshapes (resizes) the GPU hash table to a new number of buckets.
 *
 * This function creates a new, larger hash table, rehashes all existing
 * elements from the old table into the new one, and then replaces the old table.
 *
 * @param numBucketsReshape The new desired number of buckets for the hash table.
 */
""")
            output_lines.append(line)
            output_lines.append(lines[i+1]) # {
            output_lines.append("	// Functional Utility: Declares a temporary pointer for the new hash map.
")
            output_lines.append(lines[i+2]) # struct realMap *hashMapCopy = 0;
            output_lines.append("	// Functional Utility: Allocates unified memory for the new hash map with the resized number of buckets.
")
            output_lines.append(lines[i+3]) # cudaMallocManaged(&hashMapCopy, numBucketsReshape * sizeof(struct realMap));
            output_lines.append("	// Functional Utility: Synchronizes the device.
")
            output_lines.append(lines[i+4]) # cudaDeviceSynchronize();
            output_lines.append("
	// Functional Utility: Declares a temporary pointer for copying keys.
")
            output_lines.append(lines[i+5]) # int *keyCopy;
            output_lines.append("	// Functional Utility: Allocates unified memory for storing keys during the reshape process.
")
            output_lines.append(lines[i+6]) # cudaMallocManaged(&keyCopy, *currentNo * sizeof(int));
            output_lines.append("	// Functional Utility: Synchronizes the device.
")
            output_lines.append(lines[i+7]) # cudaDeviceSynchronize();
            output_lines.append("
	// Functional Utility: Declares a temporary pointer for copying values.
")
            output_lines.append(lines[i+8]) # int *valuesCopy;
            output_lines.append("	// Functional Utility: Allocates unified memory for storing values during the reshape process.
")
            output_lines.append(lines[i+9]) # cudaMallocManaged(&valuesCopy, *currentNo * sizeof(int));
            output_lines.append("	// Functional Utility: Synchronizes the device.
")
            output_lines.append(lines[i+10]) # cudaDeviceSynchronize();
            output_lines.append("
	// Block Logic: Initializes all entries in the new hash map to an empty state (key and value 0).
")
            output_lines.append("	// Pre-condition: `hashMapCopy` is allocated.
")
            output_lines.append("	// Invariant: All buckets in the new hash map are ready for new insertions.
")
            output_lines.append(lines[i+11]) # for(int i = 0; i < numBucketsReshape; i ++) {
            output_lines.append(lines[i+12]) # 	hashMapCopy[i].key = 0;
            output_lines.append(lines[i+13]) # 	hashMapCopy[i].value = 0;
            output_lines.append(lines[i+14]) # }
            output_lines.append("
	// Functional Utility: Counter for valid keys found in the old hash map.
")
            output_lines.append(lines[i+15]) # int x = 0;
            output_lines.append(lines[i+16]) # 
") 
            output_lines.append(lines[i+17]) # 
") 
            output_lines.append("	// Block Logic: Iterates through the old hash map to extract existing key-value pairs.
")
            output_lines.append("	// Pre-condition: `hashMap` contains existing data.
")
            output_lines.append("	// Invariant: `keyCopy` and `valuesCopy` will contain all valid key-value pairs from `hashMap`.
")
            output_lines.append(lines[i+18]) # for(int i = 0; i < mapSize; i ++) {
            output_lines.append("		// Block Logic: Checks if a bucket in the old hash map contains a valid key (not 0).
")
            output_lines.append("		// Invariant: Valid key-value pairs are copied to temporary arrays.
")
            output_lines.append(lines[i+19]) # 	if(hashMap[i].key != 0) {
            output_lines.append(lines[i+20]) # 		keyCopy[x] = hashMap[i].key;
            output_lines.append(lines[i+21]) # 		valuesCopy[x] = hashMap[i].value;
            output_lines.append(lines[i+22]) # 		x ++;
            output_lines.append(lines[i+23]) # 	}
            output_lines.append(lines[i+24]) # }
            output_lines.append("
	// Functional Utility: Launches the `kernel_insert` to rehash all copied key-value pairs into the new hash map.
")
            output_lines.append("	// This implicitly handles collision resolution and places elements in their new locations.
")
            output_lines.append("	// Inline: The kernel is launched with a grid size calculated to ensure all 'x' elements are processed.
")
            output_lines.append("	// Each block has 256 threads.
")
            output_lines.append(lines[i+25]) # kernel_insert<<< (x + 255) / 256, 256 >>>(keyCopy, x, valuesCopy, hashMapCopy,
            output_lines.append(lines[i+26]) # 	numBucketsReshape, copyList, NULL);
            output_lines.append("	// Functional Utility: Synchronizes the device to ensure all insertions are complete.
")
            output_lines.append(lines[i+27]) # cudaDeviceSynchronize();
            output_lines.append("
	// Functional Utility: Frees the memory of the old hash map.
")
            output_lines.append(lines[i+28]) # cudaFree(hashMap);
            output_lines.append("
	// Functional Utility: Updates the `hashMap` pointer to point to the newly created and populated hash map.
")
            output_lines.append(lines[i+29]) # hashMap = hashMapCopy;
            output_lines.append("	// Functional Utility: Updates `mapSize` to reflect the new size of the hash table.
")
            output_lines.append(lines[i+30]) # mapSize = numBucketsReshape;
            output_lines.append(lines[i+31]) # }
            i += 32

        elif "bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys)" in line:
            output_lines.append("""
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
            output_lines.append(line)
            output_lines.append(lines[i+1]) # {
            output_lines.append("
	// Block Logic: Checks if the current capacity is sufficient for the new batch of insertions.
")
            output_lines.append("	// Pre-condition: `*currentNo` is the current count, `numKeys` is the batch size, `mapSize` is current capacity.
")
            output_lines.append("	// Invariant: If capacity is insufficient, the hash table is reshaped to a larger size.
")
            output_lines.append(lines[i+2]) # if(*currentNo + numKeys >= mapSize) {
            output_lines.append("		// Functional Utility: Calculates a new map size with a 5% overhead.
")
            output_lines.append(lines[i+3]) # 	reshape(*currentNo + numKeys + (*currentNo + numKeys) * 5 / 100);
            output_lines.append("		// Functional Utility: Updates the `mapSize` global variable.
")
            output_lines.append(lines[i+4]) # 	mapSize = *currentNo + numKeys + (*currentNo + numKeys) * 5 / 100;
            output_lines.append(lines[i+5]) # }
            output_lines.append("
	// Functional Utility: Allocates unified memory for a copy of the input keys.
")
            output_lines.append(lines[i+6]) # int *keyCopy;
            output_lines.append(lines[i+7]) # cudaMallocManaged(&keyCopy, numKeys * sizeof(int));
            output_lines.append(lines[i+8]) # cudaDeviceSynchronize();
            output_lines.append("
	// Functional Utility: Allocates unified memory for a copy of the input values.
")
            output_lines.append(lines[i+9]) # int *valuesCopy;
            output_lines.append(lines[i+10]) # cudaMallocManaged(&valuesCopy, numKeys * sizeof(int));
            output_lines.append(lines[i+11]) # cudaDeviceSynchronize();
            output_lines.append("
	// Functional Utility: Copies host-side `keys` data to device-side `keyCopy`.
")
            output_lines.append(lines[i+12]) # memcpy(keyCopy, keys, numKeys * sizeof(int));
            output_lines.append("	// Functional Utility: Copies host-side `values` data to device-side `valuesCopy`.
")
            output_lines.append(lines[i+13]) # memcpy(valuesCopy, values, numKeys * sizeof(int));
            output_lines.append("
	// Functional Utility: Launches the `kernel_insert` to perform parallel insertions using the copied data.
")
            output_lines.append("	// It passes `currentNo` for atomic updates to the element count.
")
            output_lines.append("	// Inline: The kernel is launched with a grid size calculated to ensure all `numKeys` elements are processed.
")
            output_lines.append("	// Each block has 256 threads.
")
            output_lines.append(lines[i+14]) # kernel_insert<<< (numKeys + 255) / 256, 256 >>>(keyCopy, numKeys, valuesCopy,
            output_lines.append(lines[i+15]) # 	hashMap, mapSize, copyList, currentNo);
            output_lines.append(lines[i+16]) # cudaDeviceSynchronize();
            output_lines.append("
	// Functional Utility: Placeholder return value.
")
            output_lines.append(lines[i+17]) # return false;
            output_lines.append(lines[i+18]) # }
            i += 19

        elif "int* GpuHashTable::getBatch(int* keys, int numKeys)" in line:
            output_lines.append("""
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
            output_lines.append(line)
            output_lines.append(lines[i+1]) # {
            output_lines.append("
	// Functional Utility: Allocates unified memory for storing the results (retrieved values).
")
            output_lines.append(lines[i+2]) # int* results = NULL;
            output_lines.append(lines[i+3]) # cudaMallocManaged(&results, numKeys * sizeof(int));
            output_lines.append(lines[i+4]) # cudaDeviceSynchronize();
            output_lines.append("
	// Functional Utility: Allocates unified memory for a copy of the input keys.
")
            output_lines.append(lines[i+5]) # int *keyCopy;
            output_lines.append(lines[i+6]) # cudaMallocManaged(&keyCopy, numKeys * sizeof(int));
            output_lines.append(lines[i+7]) # cudaDeviceSynchronize();
            output_lines.append("
	// Functional Utility: Copies host-side `keys` data to device-side `keyCopy`.
")
            output_lines.append(lines[i+8]) # memcpy(keyCopy, keys, numKeys * sizeof(int));
            output_lines.append("
	// Functional Utility: Launches the `kernel_get` to perform parallel retrieval.
")
            output_lines.append("	// Inline: The kernel is launched with a grid size calculated to ensure all `numKeys` elements are processed.
")
            output_lines.append("	// Each block has 256 threads.
")
            output_lines.append(lines[i+9]) # kernel_get<<< (numKeys + 255) / 256, 256 >>>(keyCopy, numKeys, hashMap,
            output_lines.append(lines[i+10]) # 	results, mapSize, copyList);
            output_lines.append(lines[i+11]) # cudaDeviceSynchronize();
            output_lines.append("
	// Functional Utility: Returns the device-side pointer to the results.
")
            output_lines.append(lines[i+12]) # return results;
            output_lines.append(lines[i+13]) # }
            i += 14

        elif "float GpuHashTable::loadFactor()" in line:
            output_lines.append("""
/**
 * @brief Calculates the current load factor of the hash table.
 *
 * The load factor is the ratio of the number of elements in the hash table
 * to the total number of available buckets.
 *
 * @return The current load factor as a float.
 */
""")
            output_lines.append(line)
            output_lines.append(lines[i+1]) # {
            output_lines.append("	
")
            output_lines.append("	// Functional Utility: Computes the load factor, casting to float for accurate division.
")
            output_lines.append(lines[i+2]) # return (float)(*currentNo) / (float)(mapSize);
            output_lines.append(lines[i+3]) # }
            i += 4

        elif "#define HASH_INIT" in line:
            output_lines.append("
// @brief Macro to initialize the GpuHashTable with an initial size of 1.
")
            output_lines.append(line)
            output_lines.append("// @brief Macro to reshape the GpuHashTable to a specified size.
")
            output_lines.append(lines[i+1]) # #define HASH_RESERVE(size)
            output_lines.append("
// @brief Macro to insert a batch of key-value pairs into the GpuHashTable.
")
            output_lines.append(lines[i+2]) # #define HASH_BATCH_INSERT(keys, values, numKeys)
            output_lines.append("// @brief Macro to retrieve a batch of values for given keys from the GpuHashTable.
")
            output_lines.append(lines[i+3]) # #define HASH_BATCH_GET(keys, numKeys)
            output_lines.append("
// @brief Macro to get the current load factor of the GpuHashTable.
")
            output_lines.append(lines[i+4]) # #define HASH_LOAD_FACTOR
            i += 5
        
        elif '#include "test_map.cpp"' in line:
            output_lines.append(line.strip() + " // Functional Utility: Includes a test map implementation.
")
            i += 1
            while i < len(lines) and (lines[i].strip().startswith("#include") or len(lines[i].strip()) == 0):
                if "#include <unistd.h>" in lines[i]:
                    output_lines.append(lines[i].strip() + " // Functional Utility: Standard POSIX header for system calls like `access`.
")
                elif "#include <sys/types.h>" in lines[i]:
                    output_lines.append(lines[i].strip() + " // Functional Utility: Standard header for fundamental system data types.
")
                elif "#include <sys/stat.h>" in lines[i]:
                    output_lines.append(lines[i].strip() + " // Functional Utility: Standard header for file status information.
")
                elif "#include <fcntl.h>" in lines[i]:
                    output_lines.append(lines[i].strip() + " // Functional Utility: Standard header for file control options.
")
                else:
                    output_lines.append(lines[i])
                i += 1

        elif "#ifndef _HASHCPU_" in line:
            output_lines.append(line)
            output_lines.append(lines[i+1]) # #define _HASHCPU_
            output_lines.append(lines[i+2]) # (blank line)
            output_lines.append(lines[i+3]) # using namespace std;
            output_lines.append(lines[i+4]) # (blank line)
            output_lines.append("// @brief Defines an invalid key value, typically used to mark empty hash table slots.
")
            output_lines.append(lines[i+5]) # #define	KEY_INVALID		0
            output_lines.append("""
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
            output_lines.append(lines[i+6]) # #define DIE(assertion, call_description) 
            output_lines.append(lines[i+7]) # 	do {	
            output_lines.append(lines[i+8]) # 		if (assertion) {	
            output_lines.append(lines[i+9]) # 		fprintf(stderr, "(%s, %d): ",	
            output_lines.append(lines[i+10]) # 		__FILE__, __LINE__);	
            output_lines.append(lines[i+11]) # 		perror(call_description);	
            output_lines.append(lines[i+12]) # 		exit(errno);	
            output_lines.append(lines[i+13]) # 	}	
            output_lines.append(lines[i+14]) # } while (0)
            output_lines.append("	
")
            output_lines.append("""/**
 * @brief A constant array of large prime numbers.
 *
 * These prime numbers are utilized in the hash functions to minimize collisions
 * and distribute keys evenly across the hash table, thereby improving performance.
 * The use of `llu` (unsigned long long) ensures these large numbers are handled correctly.
 */
""")
            output_lines.append(lines[i+15]) # const size_t primeList[] =
            
            i += 16
            while i < len(lines) and "};" not in lines[i]:
                output_lines.append(lines[i])
                i += 1
            output_lines.append(lines[i]) # };
            output_lines.append("


") 
            i += 1
            
            # struct realMap
            if i < len(lines) and "struct realMap {" in lines[i]:
                output_lines.append("""/**
 * @brief Structure representing a single entry (bucket) in the hash table.
 *
 * Each entry stores an integer key and its corresponding integer value.
 * A key of 0 typically denotes an empty or invalid bucket.
 */
""")
                output_lines.append(lines[i]) # struct realMap {
                output_lines.append(lines[i+1]) # 	int key;
                output_lines.append(lines[i+2]) # 	int value;
                output_lines.append(lines[i+3]) # };
                output_lines.append("


") 
                i += 4
            
            # hash1
            if i < len(lines) and "__host__ __device__  int hash1(int data, int limit, size_t *primeList1)" in lines[i]:
                output_lines.append("""
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
                output_lines.append(lines[i]) # __host__ __device__  int hash1(...)
                output_lines.append(lines[i+1]) # {
                output_lines.append(lines[i+2].strip() + " // Inline: Applies multiplication with a large prime and two modulo operations to calculate the hash.
")
                output_lines.append(lines[i+3]) # }
                i += 4

            # hash2
            if i < len(lines) and "int hash2(int data, int limit)" in lines[i]:
                output_lines.append("""
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
                output_lines.append(lines[i]) # int hash2(...)
                output_lines.append(lines[i+1]) # {
                output_lines.append(lines[i+2].strip() + " // Inline: Uses different primes from `primeList` for hash calculation.
")
                output_lines.append(lines[i+3]) # }
                i += 4

            # hash3
            if i < len(lines) and "int hash3(int data, int limit)" in lines[i]:
                output_lines.append("""
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
                output_lines.append(lines[i]) # int hash3(...)
                output_lines.append(lines[i+1]) # {
                output_lines.append(lines[i+2].strip() + " // Inline: Uses different primes from `primeList` for hash calculation.
")
                output_lines.append(lines[i+3]) # }
                i += 4

            # count()
            if i < len(lines) and "int count()" in lines[i]:
                output_lines.append("""
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
                output_lines.append(lines[i]) # int count()
                output_lines.append(lines[i+1]) # {
                output_lines.append("	// Block Logic: Iterates through the primeList array until a null (0) element is encountered.
")
                output_lines.append("	// Invariant: `i` counts the number of valid (non-zero) primes.
")
                output_lines.append(lines[i+2]) # 	int i = 0;
                output_lines.append(lines[i+3]) # 
                output_lines.append(lines[i+4].strip() + " // Inline: Checks for null terminator, likely indicating end of list.
")
                output_lines.append(lines[i+5]) # 		i ++;
                output_lines.append(lines[i+6]) # 	}
                output_lines.append("	// Functional Utility: Returns the count of primes plus one (assuming `\0` is not counted as a prime and is an end marker).
")
                output_lines.append(lines[i+7]) # 	return i + 1;
                output_lines.append(lines[i+8]) # }
                i += 9

            # GpuHashTable class declaration
            if i < len(lines) and "class GpuHashTable" in lines[i]:
                output_lines.append("""

/**
 * @brief Declaration of the GpuHashTable class.
 *
 * This class provides a high-level interface for interacting with the
 * GPU-accelerated hash table, encapsulating the CUDA memory management
 * and kernel launch details.
 */
""")
                output_lines.append(lines[i]) # class GpuHashTable
                output_lines.append(lines[i+1]) # {
                output_lines.append(lines[i+2]) # 	public:
                output_lines.append("		// @brief Constructor: Initializes the hash table with a given size.
")
                output_lines.append(lines[i+3]) # 		GpuHashTable(int size);
                output_lines.append("		// @brief Reshapes the hash table to a new size.
")
                output_lines.append(lines[i+4]) # 		void reshape(int sizeReshape);
                output_lines.append("		
")
                output_lines.append("		// @brief Inserts a batch of key-value pairs.
")
                output_lines.append(lines[i+5]) # 		bool insertBatch(int *keys, int* values, int numKeys);
                output_lines.append("		// @brief Retrieves a batch of values for given keys.
")
                output_lines.append(lines[i+6]) # 		int* getBatch(int* key, int numItems);
                output_lines.append("		
")
                output_lines.append("		// @brief Calculates the current load factor.
")
                output_lines.append(lines[i+7]) # 		float loadFactor();
                output_lines.append("		// @brief Placeholder for occupancy calculation (function not defined in this file).
")
                output_lines.append(lines[i+8]) # 		void occupancy();
                output_lines.append("		// @brief Placeholder for printing hash table information (function not defined in this file).
")
                output_lines.append(lines[i+9]) # 		void print(string info);
                output_lines.append("	
")
                output_lines.append("		// @brief Destructor: Frees all allocated GPU memory.
")
                output_lines.append(lines[i+10]) # 		~GpuHashTable();
                output_lines.append(lines[i+11]) # };
                output_lines.append(lines[i+12]) # 
                output_lines.append(lines[i+13]) # #endif
                i += 14
        else:
            output_lines.append(line)
            i += 1

    return "".join(output_lines)

# Read the original content from the file
file_path = './ae14c66f-836c-4b35-a52d-049aece8b11c/gpu_hashtable.cu'
with open(file_path, 'r') as f:
    original_code = f.read()

# Generate the commented code
commented_code_output = generate_commented_code_simplified(original_code)

# Write the commented code to a temporary file
output_file_path = "./commented_gpu_hashtable.cu"
with open(output_file_path, "w") as f:
    f.write(commented_code_output)

