/**
 * @file gpu_hashtable.cu
 * @brief CUDA Hash Table with update reconciliation and adaptive scaling.
 * 
 * This module implements a parallel hash table designed for high-concurrency 
 * workloads. It features an open-addressing scheme with circular linear probing 
 * and includes a mechanism to reconcile key updates vs. new insertions via 
 * device-side atomic counters, ensuring accurate load factor monitoring and 
 * proactive resizing.
 * 
 * Algorithm: Open addressing with two-pass circular linear probing.
 * Memory Model: Global memory (cudaMalloc) for pair storage and update tracking.
 * Domain: HPC, Parallel Data Structures.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief Device-side hash calculation using prime modular constants.
 */
__device__ unsigned int hash_func(int key, int size) {

	return (((long)key * PRIME_A) % PRIME_B % size);
}

/**
 * @brief CUDA kernel for parallel entry insertion.
 * 
 * Functional Utility: Uses atomicCAS to claim slots or update existing keys. 
 * Increments updates_count if an existing key is overridden.
 * 
 * @param updates_count Pointer to a device counter for tracking redundant insertions.
 */
__global__ void insert_pair(int *keys, int *values, int numKeys, struct my_pair *hashtable,
				int size, int* updates_count) {

	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= numKeys)
		return;
	if (keys[index] <= 0 || values[index] <= 0) {
		
		return;
	}

	int old_key;
	int key = keys[index];
	int value = values[index];
	unsigned int pos = hash_func(keys[index], size);
	unsigned int i = 0;

	/**
	 * Block Logic: Circular probe insertion loop.
	 * Invariant: Traverses the table until the key is successfully placed or updated.
	 */
	while(true) {

		
		
		pos = (pos + i) % size;

		
		
		old_key = atomicCAS(&hashtable[pos].key, 0, key);

		if (old_key == key) {
			
			// Logic: Key matches; perform thread-safe update and track redundant op.
			hashtable[pos].value = value;
			
			atomicAdd(updates_count, 1);
			return;
		}
		else if (old_key == 0) {
			
			
			// Logic: Successfully claimed a new slot.
			hashtable[pos].value = value;
			return;
		}
		else {
			
			// Logic: Collision occurred; increment probe offset.
			i++;
		}
	}
}



/**
 * @brief CUDA kernel for re-hashing data during capacity migration.
 */
__global__ void copy_values(struct my_pair *old_pairs, struct my_pair *new_pairs, 
															int size, int new_size) {
	
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= size)
		return;

	// Logic: Skip empty source slots to optimize re-hashing throughput.
	if (old_pairs[index].key == 0)
		return;

	unsigned int pos = hash_func(old_pairs[index].key, new_size);
	unsigned int i = 0;

	int old_key = old_pairs[index].key;
	int old_value = old_pairs[index].value;
	int value;

	while(true) {

		pos = (pos + i) % new_size;

		
		
		// Logic: Atomic reservation in the new memory space.
		value = atomicCAS(&new_pairs[pos].key, 0, old_key);
		if (value == 0) {
			new_pairs[pos].value = old_value;
			return;
		}
		i++;
	}

}

/**
 * @brief CUDA kernel for parallel entry retrieval.
 */
__global__ void get_pair(int *keys, int numKeys, struct my_pair *hashtable, int size, 


						int *values) {

	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= numKeys)
		return;

	unsigned int pos = hash_func(keys[index], size);
	unsigned int i = 0;
	int key;

	/**
	 * Block Logic: Linear probe search.
	 */
	while(true) {

		pos = (pos + i) % size;
		key = hashtable[pos].key;
		
		
		if (key == keys[index]) {
			// Logic: Match found; extract associated value.
			values[index] = hashtable[pos].value;
			return;
		}
		i++;
	}
}



/**
 * @brief Constructor: Initializes table storage and atomic counters.
 */
GpuHashTable::GpuHashTable(int size) {


	cudaError_t err;
	
	// Memory Hierarchy: Global memory allocation for the pair array.
	err = cudaMalloc((void **)&my_hashtable, size * sizeof(struct my_pair));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s!\n", cudaGetErrorString(err));
		return;
	}

	// Memory Hierarchy: Global memory allocation for the update tracker.
	err = cudaMalloc((void **)&d_num_updates, sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s!\n", cudaGetErrorString(err));
		return;
	}

	

	err = cudaMemset(my_hashtable, 0, size * sizeof(struct my_pair));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s!\n", cudaGetErrorString(err));
		return;
	}

	err = cudaMemset(d_num_updates, 0, sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s!\n", cudaGetErrorString(err));
		return;
	}

	

	host_size = size;
	host_current_size = 0;

}


/**
 * @brief Destructor: Frees all device-side allocations.
 */
GpuHashTable::~GpuHashTable() {

	cudaFree(my_hashtable);
	cudaFree(d_num_updates);
}


/**
 * @brief Resizes the hash table using parallel re-insertion.
 */
void GpuHashTable::reshape(int numBucketsReshape) {

	cudaError_t err;
	struct my_pair *new_hashtable;

	err = cudaMalloc(&new_hashtable, numBucketsReshape * sizeof(struct my_pair));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s ( reshape)!\n", cudaGetErrorString(err));
		return;
	}
	err = cudaMemset(new_hashtable, 0, numBucketsReshape * sizeof(struct my_pair));
	if (err != cudaSuccess) {
		fprintf (stderr, "[ERROR] %s ( reshape)!\n", cudaGetErrorString(err));
		return;
	}

	
	const size_t block_size = BLOCK_SIZE;
	size_t block_num = host_size / block_size;

	if (host_size % block_size != 0)
		block_num++;

	// Synchronization: Ensures re-hashing completes before old memory is released.
	copy_values>>(my_hashtable, new_hashtable, 
				host_size, numBucketsReshape);

	
	cudaDeviceSynchronize();

	
	
	cudaFree(my_hashtable);

	my_hashtable = new_hashtable;
	host_size = numBucketsReshape;

}


/**
 * @brief Performs host-initiated batch parallel insertion.
 * 
 * Logic: Monitors occupancy via update tracking and triggers expansion 
 * if density exceeds 85% or if the batch would overflow.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {

	cudaError_t err;	
	int new_size;
	int *keys_d;
	int *values_d;

	

	err = cudaMalloc((void **)&keys_d, numKeys * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s ( in insertBatch)!\n", cudaGetErrorString(err));
		return false;
	}
	err = cudaMalloc((void **)&values_d, numKeys * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s ( in insertBatch)!\n", cudaGetErrorString(err));
		return false;	
	}

	

	cudaMemcpy(keys_d, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(values_d, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	
	
	

	int prev_size = host_size;
	new_size = getNewSize(numKeys);

	// Adaptive Scaling: Proactively expands table to maintain O(1) targets.
	if (new_size != -1)
		reshape(new_size);

	
	const size_t block_size = BLOCK_SIZE;
	size_t block_num = numKeys / block_size;

	if (numKeys % block_size != 0)
		block_num++;


	insert_pair>>(keys_d, values_d, numKeys, my_hashtable, host_size, d_num_updates);
	
	cudaDeviceSynchronize();

	

	int *host_updates = (int*) malloc(sizeof(int));
	cudaMemcpy(host_updates, d_num_updates, sizeof(int), cudaMemcpyDeviceToHost);

	// Logic: Reconciles actual insertion count by subtracting overridden keys.
	host_current_size += numKeys - (*host_updates);
	

	if (prev_size != host_size) {

		float current_lf = loadFactor();
		if (current_lf < MIN_LF - ROUND_ERROR) {
			
			
			int reshape_size = host_current_size / MIN_LF;
			reshape(reshape_size);
		}
	}

	
	
	if (host_updates != 0) {
		err = cudaMemset(d_num_updates, 0, sizeof(int));
		if (err != cudaSuccess) {
			fprintf(stderr, "[ERROR] %s!\n", cudaGetErrorString(err));
			return false;
		}
	}

	
	cudaFree(keys_d);
	cudaFree(values_d);
	free(host_updates);

	return true;
}


/**
 * @brief Performs host-initiated batch parallel retrieval.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	
	int *keys_d; 
	int *values_d;
	int *host_values;

	cudaError_t err;	

	
	

	err = cudaMalloc(&values_d, numKeys * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s ( in 'getBatch')!\n", cudaGetErrorString(err));
		return NULL;
	}

	err = cudaMalloc(&keys_d, numKeys * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s ( in 'getBatch')!\n", cudaGetErrorString(err));
		return NULL;
	}

	

	cudaMemcpy(keys_d, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	const size_t block_size = BLOCK_SIZE;
	size_t block_num = numKeys / block_size;

	if (numKeys % block_size != 0)
		block_num++;

	
	get_pair>>(keys_d, numKeys, my_hashtable, host_size, values_d);

	cudaDeviceSynchronize();

	

	host_values = (int *) malloc(numKeys * sizeof(int));
	if (host_values == NULL) {
		fprintf(stderr, "[ERROR] MALLOC !\n");
		return NULL;
	}

	err = cudaMemcpy(host_values, values_d, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s (memcpy for 'values')!\n", cudaGetErrorString(err));
		return NULL;
	}
	cudaFree(keys_d);
	cudaFree(values_d);

	return host_values;

}


/**
 * @brief Returns the current occupancy density.
 */
float GpuHashTable::loadFactor() {
	if (host_size == 0)
		return -1;
	return ((float)host_current_size/host_size);
}


/**
 * @brief Calculates the new target size based on incoming batch pressure.
 */
int GpuHashTable::getNewSize(int numKeys) {

	int new_size = -1;
	int current_size = host_current_size;
	int size = host_size;
	float current_lf = (float)(current_size + numKeys) / size;

	if (current_size + numKeys > size || current_lf >= MIN_LF + MAX_RANGE)
		new_size = (current_size + numKeys) / MIN_LF;

	
	return new_size;

}


// ... hash functions and primeList documentation ...

/**
 * @struct my_pair
 * @brief Representation of a key-value mapping on the GPU.
 */
struct my_pair {

	int key;
	int value;
};




/**
 * @class GpuHashTable
 * @brief Host controller for Managing the GPU hash mapping lifecycle.
 */
class GpuHashTable
{
	struct my_pair *my_hashtable;   // Pointer to device entry array.
	int *d_num_updates;            // Device counter for reconcile updates.
	int host_size;                 // Current capacity.
	int host_current_size;         // Actual occupancy.

	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		
		float loadFactor();
		int getNewSize(int numKeys);
		void printInfo(const char*);
		void occupancy();
		void print(string info);
	
		~GpuHashTable();
};

#endif

