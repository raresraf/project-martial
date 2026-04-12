
/**
 * @file gpu_hashtable.cu
 * @brief A mostly correct GPU-accelerated hash table using open addressing and
 *        linear probing with Unified Memory.
 * @details This file implements a hash table on the GPU using a correct linear
 * probing algorithm for collision resolution and a proper parallel re-hashing
 * strategy. It correctly uses atomic operations for claiming slots and for
 * counting the number of new insertions in parallel.
 *
 * @warning RACE CONDITION: The `insert_kernel` contains a race condition on the
 * non-atomic value update for existing keys, which can lead to data loss.
 * @warning LOGIC BUG: The `get_kernel` incorrectly terminates its search if it
 * encounters an empty slot, which will fail to find keys that are located after
 * a deleted element in a probe chain.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief Host-side constructor for the GpuHashTable.
 * @details Allocates Unified Memory for the hash table, making the `h.p` pointer
 * accessible from both the host (CPU) and the device (GPU).
 */
GpuHashTable::GpuHashTable(int size) {
	h.p = 0;
	// Allocate Unified Memory.
	cudaMallocManaged(&h.p, size * sizeof(Node));
	if (h.p == 0) {
		printf("Eroare alocare h.p init
");
		exit(-1);
	}
	// This host-side loop is valid because the memory is unified.
	for (int i = 0; i < size; i++) {
		h.p[i].key = 0;
		h.p[i].value = 0;
	}
	h.maxLength = size;
	h.size = 0;
}

/**
 * @brief Host-side destructor. Frees GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(h.p);
}

/**
 * @brief CUDA kernel to resize the hash table by re-hashing all elements.
 * @details Each thread is responsible for one element from the old table, which
 * it re-hashes and inserts into the new, larger table using linear probing.
 */
__global__ void reshape_kernel(HashTable ht1, HashTable ht2) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (i < ht1.maxLength) {
		// Skip empty slots in the old table.
		if (ht1.p[i].key != 0) {
			unsigned int key = ht1.p[i].key;
			unsigned int value = ht1.p[i].value;
			int l2 = ht2.maxLength;
			unsigned int rez;
			int j;
			
			// Re-hash the key for the new table size.
			int ind = (key * 52679969llu) % 71267046102139967llu % l2;

			// --- Re-insertion with correct Linear Probing ---
			// Probes from the initial index to the end of the new table.
			for (j = ind; j < l2; j++) {
				// Atomically claim an empty slot. This is safe as each thread migrates a unique key.
				rez = atomicCAS(&(ht2.p[j].key), 0, key);
				if (rez == 0) {
					ht2.p[j].value = value;
					break;
				}
			}

			// If no slot was found, wrap around and probe from the beginning.
			if (j == l2) {
				for (j = 0; j < ind; j++) {
					rez = atomicCAS(&(ht2.p[j].key), 0, key);
					if (rez == 0) {
						ht2.p[j].value = value;
						break;
					}
				}
			}
		}
	}
}

/**
 * @brief Resizes the hash table to a new, larger capacity.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	HashTable ht;
	
	// 1. Allocate a new, larger table using Unified Memory.
	ht.p = 0;
	cudaMallocManaged(&ht.p, numBucketsReshape * sizeof(Node*));
	if (ht.p == 0) {
		printf("Eroare alocare ht.p reshape
");
		exit(-1);
	}
	ht.maxLength = numBucketsReshape;
	ht.size = h.size;
	
	if (h.size != 0) {
		// 2. Calculate launch parameters and launch the kernel to perform parallel re-hashing.
		const size_t block_size = 1024;
	    size_t blocks_no = (h.maxLength + block_size - 1) / block_size;
	    reshape_kernel>>(h, ht);
	    cudaDeviceSynchronize();
	}

	// 3. Free the old table memory and update the pointer.
	cudaFree(h.p);
	h = ht;
}


/**
 * @brief CUDA kernel for inserting a batch of key-value pairs.
 * @warning RACE CONDITION: The value update (`ht.p[j].value = values[i]`) is not
 * atomic. This can lead to data loss if multiple threads update the same key.
 */
__global__ void insert_kernel(int *keys, int *values, int numKeys, HashTable ht, int* nr) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < numKeys) {
		unsigned int rez;
		int l = ht.maxLength;
		int ind = (keys[i] * 52679969llu) % 71267046102139967llu % l;
		
		// --- Linear Probing with Full Wrap-Around ---
		int j;
		// Probe from hash index to the end.
		for (j = ind; j < l; j++) {
			// Atomically check for an empty slot (0) or an existing key.
			rez = atomicCAS(&(ht.p[j].key), 0, keys[i]);
			if (rez == keys[i] || rez == 0) {
				// BUG: Non-atomic write creates a race condition on update.
				ht.p[j].value = values[i];
				break;
			}
		}

		// If not found, wrap around and probe from the beginning.
		if (j == l) {
			for (j = 0; j < ind; j++) {
				rez = atomicCAS(&(ht.p[j].key), 0, keys[i]);
				if (rez == keys[i] || rez == 0) {
					ht.p[j].value = values[i];
					break;
				}
			}
		}
		
		// If a new element was inserted (slot was empty), atomically increment counter.
		if (rez == 0) {
			atomicAdd(&nr[0], 1);
		}
	}
}

/**
 * @brief Inserts a batch of key-value pairs from the host.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	// Resize if the load factor will exceed the threshold.
	int total = h.size + numKeys;
    if (total >= 0.8 * h.maxLength) {
    	int size_reshape = total / 0.85; // Corrected calculation
		reshape(size_reshape);
    }
    
    // 1. Allocate temporary device memory for the input batch.
    const size_t block_size = 1024;
    size_t blocks_no = (numKeys + block_size - 1) / block_size;
    int *device_keys = 0;
    int *device_values = 0;
    cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
    cudaMalloc((void **) &device_values, numKeys * sizeof(int));
    if (device_keys == 0 || device_values == 0) {
    	printf("Eroare alocare device_keys/device_values insert
");
		exit(-1);
    }
    cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    // 2. Allocate a managed memory counter to track new insertions.
    int *nr = 0;
    cudaMallocManaged(&nr, sizeof(int));
    if (nr == 0) {
    	printf("Eroare alocare nr insert");
    	exit(-1);
    }
    nr[0] = 0;

    // 3. Launch the insertion kernel.
    insert_kernel>>(device_keys, device_values, numKeys, h, nr);
    cudaDeviceSynchronize();

    // 4. Update host-side count with the atomically calculated number of new elements.
    h.size += nr[0];

    // 5. Free temporary buffers.
    cudaFree(nr);
    cudaFree(device_keys);
    cudaFree(device_values);

	return true;
}


/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @warning LOGIC BUG: The search incorrectly stops if it finds an empty slot (`key == 0`),
 * which will fail to find keys that are located after a deleted element in a probe chain.
 */
__global__ void get_kernel(int *keys, int *values, int numKeys, HashTable ht) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (i < numKeys) {
		int l = ht.maxLength;
		int j;
		int ind = (keys[i] * 52679969llu) % 71267046102139967llu % l;

		// --- Search with Flawed Linear Probing ---
		for (j = ind; j < l; j++) {
			// BUG: This check is incorrect. If an element was deleted from this slot,
			// the search should continue. It should only stop if the keys match.
			if (ht.p[j].key == keys[i] || ht.p[j].key == 0) {
				values[i] = ht.p[j].value;
				break;
			}
		}

		if (j == l) {
			for (j = 0; j < ind; j++) {
				if (ht.p[j].key == keys[i] || ht.p[j].key == 0) {
					values[i] = ht.p[j].value;
					break;
				}
			}
		}
	}
}

/**
 * @brief Retrieves values for a batch of keys from the host.
 */
int* GpuHashTable::getBatch(int *keys, int numKeys) {
    const size_t block_size = 1024;
    size_t blocks_no = (numKeys + block_size - 1) / block_size;
    
    // Allocate Unified Memory, which simplifies data handling.
    int *device_keys = 0;
    int *device_values = 0;
    cudaMallocManaged(&device_keys, numKeys * sizeof(int));
    cudaMallocManaged(&device_values, numKeys * sizeof(int));

    if (device_keys == 0 || device_values == 0) {
    	printf("Eroare alocare device_keys/device_values get
");
		exit(-1);
    }

    // Inefficient Copy: This host-side loop is less efficient than a single cudaMemcpy.
    for (int i = 0; i < numKeys; i++) {
    	device_keys[i] = keys[i];
    }

    // Launch the get kernel.
    get_kernel>>(device_keys, device_values, numKeys, h);
    cudaDeviceSynchronize();

    cudaFree(device_keys);

	// Return the Unified Memory pointer. The host can access it directly.
	return device_values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
	return (float)(h.size) / h.maxLength;
}

// --- Test harness and boilerplate ---
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

// ... (rest of boilerplate)
#endif
