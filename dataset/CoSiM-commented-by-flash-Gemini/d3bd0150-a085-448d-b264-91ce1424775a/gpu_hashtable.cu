/**
 * @file gpu_hashtable.cu
 * @brief CUDA Hash Table with fallback linear search and device-to-device migration.
 * 
 * This implementation uses a single-hash approach with a fallback exhaustive 
 * linear search to resolve collisions. It demonstrates a naive resizing strategy 
 * utilizing direct device memory copying.
 * 
 * Algorithm: Single-hash with fallback linear scan.
 * Memory Model: Global memory (cudaMalloc) for device-side storage.
 * Domain: HPC, Parallel Data Structures.
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


/**
 * @brief Constructor: Allocates primary device memory for the hash table.
 * 
 * @param size Initial capacity of the table.
 */
GpuHashTable::GpuHashTable(int size) {
	this->capacity = size;
	this->size = 0;

	// Memory Hierarchy: Global memory allocation for key-value pairs.
	cudaMalloc(&this->hashTableDevice, this->capacity * sizeof(Pair));
	cudaMemset(this->hashTableDevice, 0, this->capacity * sizeof(Pair));
}


/**
 * @brief Destructor: Releases device-side global memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(this->hashTableDevice);
}


/**
 * @brief Resizes the hash table using direct device-to-device memory copy.
 * 
 * Functional Utility: Increases capacity by allocating a new buffer. 
 * Note: Performs a raw memory copy which assumes consistent index mapping.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	Pair *newHashTableDevice = 0;
	cudaMalloc(&newHashTableDevice, numBucketsReshape * sizeof(Pair));
	cudaMemset(newHashTableDevice, 0, numBucketsReshape * sizeof(Pair));

	// Optimization: Direct memory transfer to minimize host-device synchronization.
	cudaMemcpy(newHashTableDevice,
		   this->hashTableDevice,
		   this->capacity * sizeof(Pair),
		   cudaMemcpyDeviceToDevice);

	this->capacity = numBucketsReshape;
	cudaFree(this->hashTableDevice);
	this->hashTableDevice = newHashTableDevice;
}


/**
 * @brief CUDA kernel for parallel entry insertion with fallback linear scan.
 * 
 * Functional Utility: Attempts to insert into the primary hashed bucket. If occupied 
 * by a different key, it performs an exhaustive linear scan of the entire table.
 */
__global__ void insertBatchDevice(Pair *hashTableDevice, int capacity, int *keys, int *values, int numKeys) {
	
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	
	if (i < numKeys) {
		int key = keys[i];
		int value = values[i];
		int bucket = hash2(key, capacity);
		// Logic: Case 1 - Direct hit or update in primary bucket.
		if (hashTableDevice[bucket].key == 0 || hashTableDevice[bucket].key == key) {
			hashTableDevice[bucket].key = key;
			hashTableDevice[bucket].value = value;
		} else {
			/**
			 * Block Logic: Fallback linear search.
			 * Logic: Exhaustively scans the memory buffer to find an empty or matching slot.
			 */
			int found = 0;
			for (int j = 0; j < capacity && found == 0; ++j) {
				if (hashTableDevice[j].key == 0 || hashTableDevice[j].key == key) {
					hashTableDevice[j].key = key;
					hashTableDevice[j].value = value;
					found = 1;
				}
			}

			// Logic: Wrap-around check for secondary scan segment.
			for (int j = 0; j < bucket && found == 0; ++j) {
				if (hashTableDevice[j].key == 0 || hashTableDevice[j].key == key) {
					hashTableDevice[j].key = key;
					hashTableDevice[j].value = value;
					found = 1;
				}
			}
		}
	}
}


/**
 * @brief Host-side orchestrator for batch parallel insertions.
 * 
 * Logic: Ensures sufficient capacity via proactive reshaping before launching 
 * the insertion kernel. Manages host-to-device data transfers.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	
	while (numKeys > this->capacity - this->size) {
		this->reshape(2 * this->capacity);
	}

	int *keys_device = 0;
	int *values_device = 0;

	cudaMalloc(&keys_device, numKeys * sizeof(int));


	cudaMalloc(&values_device, numKeys * sizeof(int));

	cudaMemcpy(keys_device,
		   keys,
		   numKeys * sizeof(int),
		   cudaMemcpyHostToDevice);
	cudaMemcpy(values_device,
		   values,
		   numKeys * sizeof(int),
		   cudaMemcpyHostToDevice);

	const size_t block_size = 1024;
	size_t blocks_no = numKeys / block_size;

	if (numKeys % block_size) {
		++blocks_no;
	}

	insertBatchDevice>>(this->hashTableDevice, this->capacity, keys_device, values_device, numKeys);
	cudaDeviceSynchronize();

	this->size += numKeys;

	cudaFree(keys_device);
	cudaFree(values_device);

	return true;
}


/**
 * @brief CUDA kernel for parallel value retrieval with linear scan fallback.
 * 
 * Functional Utility: Searches for keys using a primary bucket hash followed 
 * by a circular linear search if the primary bucket misses.
 */
__global__ void getBatchDevice(Pair *hashTableDevice, int capacity, int *keys, int *values, int numKeys) {
  
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

  
	if (i < numKeys) {
		int key = keys[i];
		int bucket = hash2(key, capacity);

		// Logic: Fast path search in primary bucket.
		if (hashTableDevice[bucket].key == key) {
			values[i] = hashTableDevice[bucket].value;
		} else {
			/**
			 * Block Logic: Exhaustive search traversal.
			 * Invariant: Probes the entire buffer linearly to resolve potential 
			 * displacement caused by collisions.
			 */
			int found = 0;
			for (int j = bucket + 1; j < capacity && found == 0; ++j) {
				if (hashTableDevice[j].key == key) {
					found = 1;
					values[i] = hashTableDevice[j].value;
				}
			}

			for (int j = 0; j < bucket && found == 0; ++j) {
				if (hashTableDevice[j].key == key) {
					found = 1;
					values[i] = hashTableDevice[j].value;
				}
			}
		}
	}
}


/**
 * @brief Performs batch retrieval of values for a set of keys.
 * 
 * Logic: Synchronizes host data with the GPU and executes the parallel search kernel.
 */
int* GpuHashTable::getBatch(int *keys, int numKeys) {
	int *values = (int *)malloc(numKeys * sizeof(int));

	int *keys_device = 0;
	int *values_device = 0;

	cudaMalloc(&keys_device, numKeys * sizeof(int));


	cudaMalloc(&values_device, numKeys * sizeof(int));

	cudaMemcpy(keys_device,
		   keys,
		   numKeys * sizeof(int),
		   cudaMemcpyHostToDevice);

	const size_t block_size = 1024;
	size_t blocks_no = numKeys / block_size;

	if (numKeys % block_size) {
		++blocks_no;
	}

	getBatchDevice>>(this->hashTableDevice, this->capacity, keys_device, values_device, numKeys);
	cudaDeviceSynchronize();

	cudaMemcpy(values,
		   values_device,
		   numKeys * sizeof(int),
		   cudaMemcpyDeviceToHost);

	return values;
}


/**
 * @brief Returns the current occupancy density.
 */
float GpuHashTable::loadFactor() {
	return (float)this->size / (float) this->capacity;
}


// ... primeList documentation omitted ...

/**
 * @brief Hashing algorithms based on modular arithmetic with prime constants.
 */
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
__host__ __device__ int hash2(int data, int limit) {
	return ((long)abs(data) * 20906033llu) % 5351951779llu % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}


/**
 * @struct Pair
 * @brief Device-side storage unit for a key-value mapping.
 */
typedef struct Pair {
	int key;
	int value;
} Pair;

/**
 * @class GpuHashTable
 * @brief Controller for managing a persistent hash table in device memory.
 */
class GpuHashTable
{
	public:
		Pair *hashTableDevice;  // Device memory buffer.
		int size;               // Occupancy count.
		int capacity;           // Buffer capacity.

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

