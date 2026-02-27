
/**
 * @file gpu_hashtable.cu
 * @brief This file implements a GPU-accelerated hash table using CUDA.
 *
 * @details The hash table is designed for high-throughput key-value storage and retrieval
 *          on NVIDIA GPUs. It utilizes linear probing for collision resolution and
 *          CUDA's atomic operations to ensure thread safety during concurrent insertions.
 *          The implementation includes device kernels for insertion and retrieval,
 *          as well as host-side methods for managing the hash table's lifecycle,
 *          including dynamic resizing (reshaping) to maintain performance.
 *          Domain: HPC & Parallelism (CUDA).
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"
#define LOAD_FACTOR 0.8f




/**
 * @brief CUDA kernel for inserting key-value pairs into the hash table.
 *
 * @details This kernel is responsible for concurrently inserting multiple key-value pairs
 *          into the hash table using a linear probing collision resolution strategy.
 *          Each thread processes a single key-value pair.
 *          It utilizes atomic operations to ensure thread-safe insertions and
 *          handles potential race conditions during updates or initial insertions.
 *          Memory hierarchy usage: Accesses global memory for the hashmap, keys, values,
 *          and num_inserted counter.
 *
 * @param hashmap Pointer to the global memory hash table.
 * @param cap Capacity of the hash table.
 * @param keys Pointer to the array of keys to insert (global memory).
 * @param values Pointer to the array of values to insert (global memory).
 * @param numKeys Total number of key-value pairs to insert.
 * @param num_inserted Pointer to a global counter for successfully inserted unique elements.
 */
__global__ void insertHash(unsigned long long *hashmap, int cap, int *keys, int *values, int numKeys, unsigned int *num_inserted) {
	// Thread ID calculation: Each thread processes a unique key-value pair.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	// Precondition: Ensure the thread ID is within the bounds of the number of keys to process.
	if(i < numKeys)
	{
		unsigned long long key_value_pair; // Stores the combined key and value.
		
		// Packing key and value into a single unsigned long long for efficient storage.
		// Key occupies the upper 32 bits, value occupies the lower 32 bits.
		key_value_pair = (unsigned long long)keys[i] << 32;
		key_value_pair += values[i];
		// Initial hash calculation for the key to determine the starting index.
		int index = hash1(keys[i], cap);
		
		
		// Block Logic: Linear probing loop for collision resolution.
		// Invariant: Continues until an empty slot is found or the key is updated.
		while(true)
		{
			// Inline: Atomically attempts to insert the key-value pair if the slot is empty (0).
			// This prevents race conditions where multiple threads might try to write to the same empty slot.
			atomicCAS(


				(unsigned long long *) &hashmap[index], 
				(unsigned long long) 0, 
				key_value_pair);

			// Check if the insertion was successful or if a collision occurred.
			if(hashmap[index] != key_value_pair)
			{
				// Collision detected or existing entry found.
				// Extract the key from the existing entry for comparison.
				unsigned long long existing_key = hashmap[index] >> 32;
				// If the existing key matches the key being inserted, it's an update operation.
				if(existing_key == key_value_pair >> 32)
				{
					// Inline: Atomically updates the value for an existing key.
					// This ensures the update is thread-safe and only happens if the key still matches.
					atomicCAS(&hashmap[index], hashmap[index], key_value_pair);
					return; // Key updated, exit the kernel for this thread.
				}	
				// If keys don't match, move to the next slot (linear probing).
				index = (index + 1) % cap;
			} else 
			{
				// Successful initial insertion into an empty slot.
				// Inline: Atomically increments the counter for total inserted elements.
				// This accounts for unique successful insertions.
				atomicInc(num_inserted, 4294967295); // 4294967295 is (2^32 - 1), representing a max value for the atomicInc wrap-around.
				return; // Insertion complete, exit the kernel for this thread.
			}			

		}
	}	
}


/**
 * @brief CUDA kernel for retrieving values associated with keys from the hash table.
 *
 * @details This kernel allows for concurrent retrieval of values for multiple keys
 *          stored in the hash table. It employs a linear probing strategy,
 *          mirroring the insertion logic to locate the correct key-value pair.
 *          Each thread processes a single key, searching for its corresponding value.
 *          If a key is not found, its corresponding value in the output array is set to 0.
 *          Memory hierarchy usage: Accesses global memory for the hashmap, keys, and values.
 *
 * @param hash Pointer to the global memory hash table.
 * @param cap Capacity of the hash table.
 * @param keys Pointer to the array of keys to search for (global memory).
 * @param values Pointer to the output array where retrieved values will be stored (global memory).
 * @param numKeys Total number of keys to retrieve.
 */
__global__ void getHash(unsigned long long *hash, int cap, int *keys, int *values, int numKeys) {
	
	// Thread ID calculation: Each thread processes a unique key to retrieve.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	// Precondition: Ensure the thread ID is within the bounds of the number of keys to process.
	if(i < numKeys)
	{
		// Initial hash calculation for the key to determine the starting index.
		int index = hash1(keys[i], cap);
		int start = index; // Store the starting index to detect a full scan without finding the key.
		
		
		// Block Logic: Linear probing loop to search for the key.
		// Invariant: Continues until the key is found, an empty slot is encountered, or a full scan is completed.
		while(true)
		{
			// If the key at the current index matches the target key, extract and return the value.
			if(hash[index] >> 32 == keys[i])
			{
				unsigned long long temp = hash[index];
				// Inline: Extracts the value (lower 32 bits) from the combined key-value pair.
				temp = temp << 32;
				temp = temp >> 32;
				values[i] =(int) temp; // Assign the retrieved value to the output array.
				return;
			} 
			// If an empty slot is encountered, the key is not in the hash table.
			else if( hash[index] >> 32 == 0)
			{
				values[i] = 0; // Indicate that the key was not found.
				return;
			} 
			// If a collision (different key) is encountered, move to the next slot (linear probing).
			else 
			{
				index = (index + 1) % cap;
				// If the search has wrapped around and returned to the starting point, the key is not found.
				if(start == index)
				{
					values[i] = 0; // Indicate that the key was not found after a full scan.
					return;
				}	
			}
		}
	}
	
}
 

/**
 * @brief Constructs a new GpuHashTable object.
 *
 * @details Initializes a new hash table on the GPU with a specified capacity.
 *          It allocates global device memory for the hash map and initializes
 *          all entries to zero, signifying empty slots.
 *          Memory hierarchy usage: Allocates and initializes global device memory.
 *
 * @param size The initial capacity (number of buckets) for the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	// Allocate device memory for the hashmap.
	cudaMalloc((void**) &hashmap, size * sizeof(unsigned long long));
	if(!hashmap)
	{
		perror("hashmap alloc"); // Error handling for memory allocation failure.
		return;
	}

	// Initialize all allocated memory to 0, marking all hash table slots as empty.
	cudaMemset(hashmap, 0, size * sizeof(unsigned long long));
	tot_size = size;        // Store the total capacity of the hash table.
	num_elements = 0;       // Initialize the count of elements to zero.
}


/**
 * @brief Destroys the GpuHashTable object.
 *
 * @details Frees the device memory allocated for the hash table.
 *          Resets the hashmap pointer, total size, and element count to their initial states.
 *          Ensures proper memory management and avoids memory leaks on the device.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashmap);      // Free the allocated device memory.
	hashmap = NULL;         // Reset the pointer to avoid dangling pointers.
	tot_size = 0;           // Reset the total size.
	num_elements = 0;       // Reset the element count.
}


/**
 * @brief Reshapes the hash table to a new capacity.
 *
 * @details This method handles the dynamic resizing of the hash table. It allocates
 *          a new hash table with the specified new capacity, re-inserts all existing
 *          key-value pairs from the old hash table into the new one, and then
 *          replaces the old hash table with the new one. This process is crucial
 *          for maintaining hash table performance (low collision rate) as the
 *          number of elements grows.
 *          Memory hierarchy usage: Involves host-to-device and device-to-host memory transfers,
 *          as well as new device memory allocation and deallocation.
 *
 * @param numBucketsReshape The new desired capacity (number of buckets) for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	
	int threads_num_block = 1024, num_blocks = tot_size / threads_num_block + 1;
	int *k_h = 0, *v_h = 0, *k_d = 0, *v_d = 0;
	unsigned long long *temp = 0;

	// Block Logic: Allocate new device memory for the reshaped hash table.
	if(cudaMalloc((void**) &temp, numBucketsReshape * sizeof(unsigned long long)) != cudaSuccess)
	{
		perror("temp alloc"); // Error handling for memory allocation failure.
		return;
	}

	// Initialize the new hash table memory to 0 (empty slots).
	cudaMemset(temp, 0, numBucketsReshape * sizeof(unsigned long long));

	// Precondition: If there are no elements, simply replace the old hashmap with the new one.
	if(num_elements == 0)
	{
		cudaFree(hashmap);      // Free the old (empty) hashmap.
		hashmap = 0;            // Reset pointer.
		hashmap = temp;         // Point to the new hashmap.
		tot_size = numBucketsReshape; // Update total size.
		return;
	}

	int num_el = 0; // Counter for elements currently in the hash table.

	// Allocate host memory to temporarily store existing keys and values.
	k_h = (int *) malloc (num_elements * sizeof(int));
	v_h = (int *) malloc (num_elements * sizeof(int));
	if( !k_h || !v_h)
	{
		perror("k_h v_h"); // Error handling for host memory allocation failure.
		return;
	}

	unsigned long long *hash_host = 0; // Host-side buffer for reading the current hashmap.
	hash_host = (unsigned long long *) malloc ( tot_size * sizeof(unsigned long long));
	// Transfer the current hash table from device to host memory.
	cudaMemcpy(hash_host, hashmap, tot_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	if(!hash_host)
	{
		perror("hash_host alloc"); // Error handling for host memory allocation failure.
		return;
	}

	unsigned long long el;
	// Block Logic: Iterate through the old hash table to extract existing key-value pairs.
	// Invariant: Only valid (non-zero) key-value pairs are extracted for re-insertion.
	for(int i = 0; i < tot_size; i++)
	{
		el = hash_host[i];
		// If the entry contains a valid key (not 0), extract key and value.
		if(el >> 32 != 0)
		{
			k_h[num_el] = el >> 32;       // Extract key (upper 32 bits).
			el = el << 32;                // Shift to isolate value.
			el = el >> 32;                // Shift back to get value.
			v_h[num_el] = el;             // Store value (lower 32 bits).
			num_el++;                     // Increment count of valid elements.
		}
	}
	
	// Allocate device memory for the temporary key and value arrays (for re-insertion).
	if(cudaMalloc((void **) &k_d, num_el * sizeof(int)) != cudaSuccess)
		perror("cudaMalloc"); // Error handling for device memory allocation.

	if(cudaMalloc((void **) &v_d, num_el * sizeof(int)) != cudaSuccess)
		perror("cudaMalloc"); // Error handling for device memory allocation.
	
	// Transfer existing keys and values from host to device memory.
	cudaMemcpy(k_d, k_h, num_el * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(v_d, v_h, num_el * sizeof(int), cudaMemcpyHostToDevice);

	unsigned int *num_inserted = 0; // Device pointer for tracking inserted elements during reshape.

	// Allocate and initialize device memory for the insertion counter.
	cudaMalloc((void **) &num_inserted, sizeof(unsigned int));
	cudaMemset(num_inserted, 0, sizeof(unsigned int));

	// Launch the insertHash kernel to re-insert all elements into the new hash table.
	insertHash>>(temp, numBucketsReshape, k_d, v_d, num_el, num_inserted);

	cudaDeviceSynchronize(); // Wait for all GPU operations to complete.
		
	cudaFree(hashmap); // Free the old hash map from device memory.
	hashmap = temp;    // Update the hashmap pointer to the new reshaped hash table.
	tot_size = numBucketsReshape; // Update the total capacity.

	// Block Logic: Clean up all temporary host and device memory.
	cudaFree(k_d);
	k_d = 0;
	cudaFree(v_d);	
	v_d = 0;
	cudaFree(num_inserted);
	num_inserted = 0;
	free(k_h);
	k_h = 0;
	free(v_h);
	v_h = 0;
	free(hash_host); // Free the host buffer used for the old hashmap.
	hash_host = 0;
	temp = 0;
}


/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 *
 * @details This host-side method orchestrates the insertion of multiple key-value pairs.
 *          It first checks if the current load factor necessitates a reshape operation
 *          to maintain performance. It then transfers the key-value pairs from host
 *          to device memory and launches the `insertHash` CUDA kernel to perform
 *          the parallel insertion on the GPU. After the kernel completes, it updates
 *          the host-side count of elements and cleans up temporary device memory.
 *          Memory hierarchy usage: Involves host-to-device memory transfers for keys and values,
 *          and device-to-host transfer for the count of inserted elements.
 *
 * @param keys Pointer to an array of keys on the host to be inserted.
 * @param values Pointer to an array of values on the host to be inserted.
 * @param numKeys The number of key-value pairs in the batch.
 * @return True if the batch insertion was successful, false otherwise.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *k_d = 0, *v_d = 0; // Device pointers for keys and values.
	unsigned int *num_inserted_d = 0, *num_inserted_h = 0; // Device and host pointers for insertion count.
	int acc = 0, threads_num_block = 1024;	
	int num_blocks = numKeys / threads_num_block + 1;

	// Allocate device memory for the keys to be inserted.
	cudaMalloc((void**) &k_d, numKeys * sizeof(int));
	if(!k_d)
	{
		perror("key alloc"); // Error handling for device memory allocation.
		return false;
	}

	// Allocate device memory for the values to be inserted.
	cudaMalloc((void**) &v_d, numKeys * sizeof(int));
	if(!v_d)
	{
		printf("value alloc"); // Error handling for device memory allocation.
		return false;
	}	

	// Transfer keys and values from host to device memory.
	cudaMemcpy(k_d, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(v_d, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);	

	// Block Logic: Calculate the number of valid (non-zero) key-value pairs.
	// Invariant: Only positive keys and values are considered valid for insertion.
	for(int i = 0; i < numKeys; i++)
		if(keys[i] > 0 && values[i] > 0)
		acc++;		

	int new_cap;
	// Block Logic: Check if resizing of the hash table is required based on the load factor.
	// Precondition: Reshape is triggered if the load factor exceeds LOAD_FACTOR (0.8).
	if( (acc + num_elements) / (float)tot_size > LOAD_FACTOR) {
		new_cap = (acc + num_elements) / LOAD_FACTOR; // Calculate new capacity based on desired load factor.
		reshape((int) (1.5 * new_cap)); // Reshape with a buffer for future growth.
	}


	num_inserted_h = (unsigned int *) malloc (sizeof(unsigned int));
	memset(num_inserted_h, 0, sizeof(unsigned int)); // Initialize host counter to zero.

	// Allocate device memory for the insertion counter and transfer initial zero value.
	cudaMalloc((void **) &num_inserted_d, sizeof(unsigned int));
	cudaMemcpy(num_inserted_d, num_inserted_h, sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	// Launch the insertHash CUDA kernel to perform parallel insertions.
	insertHash>>(hashmap, tot_size, k_d, v_d, numKeys, num_inserted_d);
	cudaDeviceSynchronize(); // Wait for the kernel to complete.

	// Check for any CUDA errors during kernel execution.
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
		perror(cudaGetErrorString(error));

	// Transfer the updated number of inserted elements from device to host.
	cudaMemcpy(num_inserted_h, num_inserted_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);	
	num_elements += (*num_inserted_h); // Update the total number of elements in the hash table.

	// Block Logic: Clean up all temporary device and host memory.
	cudaFree(k_d);
	k_d = 0;
	cudaFree(v_d);
	v_d = 0;
	cudaFree(num_inserted_d);
	num_inserted_d = 0;
	free(num_inserted_h);
	num_inserted_h = 0;

	return true; // Indicate successful batch insertion.
}


/**
 * @brief Retrieves a batch of values for given keys from the hash table.
 *
 * @details This host-side method facilitates the parallel retrieval of values
 *          associated with a batch of keys. It transfers the keys from host
 *          to device memory, launches the `getHash` CUDA kernel to perform
 *          the parallel lookup on the GPU, and then transfers the results
 *          (values) back to host memory.
 *          Memory hierarchy usage: Involves host-to-device memory transfers for keys
 *          and device-to-host memory transfers for the retrieved values.
 *
 * @param keys Pointer to an array of keys on the host to search for.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to an array of integers (on the host) containing the
 *         retrieved values. If a key is not found, its corresponding value is 0.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	// Device pointers for input keys and output values.
	int *k_d, *v_d, *values; 
	int threads_num_block = 1024;
	int num_blocks = numKeys / threads_num_block + 1;

	// Allocate device memory for input keys.
	if(cudaMalloc((void **) &k_d, numKeys * sizeof(int)) != cudaSuccess)
		perror("cudaMalloc"); // Error handling for device memory allocation.
	
	// Allocate device memory for output values.
	if(cudaMalloc((void **) &v_d, numKeys * sizeof(int)) != cudaSuccess)
		perror("cudaMalloc"); // Error handling for device memory allocation.

	// Transfer input keys from host to device memory.
	cudaMemcpy(k_d, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Launch the getHash CUDA kernel to perform parallel lookups.
	getHash>>(hashmap,tot_size,k_d, v_d, numKeys);

	// Allocate host memory for the retrieved values.
	values = (int *) malloc (numKeys * sizeof(int));
	// Transfer retrieved values from device to host memory.
	cudaMemcpy(values, v_d, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	
	// Block Logic: Clean up temporary device memory.
	cudaFree(k_d);
	cudaFree(v_d);
	k_d = 0; // Reset pointers.
	v_d = 0; // Reset pointers.

	return values; // Return the array of retrieved values on the host.
}


/**
 * @brief Calculates the current load factor of the hash table.
 *
 * @details The load factor is a measure of how full the hash table is,
 *          defined as the ratio of the number of elements to the total capacity.
 *          This metric is used to determine when the hash table should be reshaped
 *          to maintain efficient performance and a low collision rate.
 *
 * @return The current load factor as a floating-point number.
 */
float GpuHashTable::loadFactor() {
	return num_elements / (tot_size*1.0f); // Calculate load factor, ensuring floating-point division.
}



#define LOAD_FACTOR 0.8f

/**
 * @brief Initializes the GpuHashTable object.
 * @details This macro is a convenient way to create and initialize a GpuHashTable instance
 *          with a default capacity of 1. It simplifies the setup process for the hash table.
 */
#define HASH_INIT GpuHashTable GpuHashTable(1);

/**
 * @brief Reserves a specified capacity for the hash table.
 * @details This macro calls the `reshape` method to adjust the hash table's capacity.
 *          It is used to pre-allocate space or expand the hash table to accommodate
 *          a larger number of elements efficiently.
 * @param size The desired new capacity for the hash table.
 */
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @details This macro provides a simplified interface for calling the `insertBatch` method
 *          of the GpuHashTable, allowing for bulk insertion of elements.
 * @param keys An array of keys to insert.
 * @param values An array of values to insert.
 * @param numKeys The number of key-value pairs in the batch.
 */
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)

/**
 * @brief Retrieves a batch of values for given keys from the hash table.
 * @details This macro simplifies the call to the `getBatch` method of the GpuHashTable,
 *          enabling retrieval of multiple values based on their corresponding keys.
 * @param keys An array of keys to retrieve values for.
 * @param numKeys The number of keys in the batch.
 * @return An array of retrieved values.
 */
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

/**
 * @brief Gets the current load factor of the hash table.
 * @details This macro provides easy access to the `loadFactor` method of the GpuHashTable,
 *          which indicates how full the hash table is.
 * @return The current load factor.
 */
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
