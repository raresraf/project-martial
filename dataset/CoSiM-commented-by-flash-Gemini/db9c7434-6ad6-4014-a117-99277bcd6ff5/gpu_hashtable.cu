/**
 * @file gpu_hashtable.cu
 * @brief CUDA Hash Table with macro-based hashing and circular linear probing.
 * 
 * This module implements a parallel hash table utilizing macro-defined hashing 
 * and circular linear probing for collision resolution. It includes mechanisms 
 * for tracking updates during insertion to accurately maintain the load factor.
 * 
 * Algorithm: Open addressing with circular linear probing and update tracking.
 * Memory Model: Global memory (cudaMalloc) for pair storage.
 * Domain: HPC, Parallel Data Structures.
 */

#include 
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define HASH_COMPUTE(k, a, b, l) ((((long)k * a) % b) % l)
#define HASH_A 13169977llu
#define HASH_B 5351951779llu
#define MAX_LOAD_FACTOR 0.9f
#define MIN_LOAD_FACTOR 0.8f




/**
 * @brief CUDA kernel for parallel data migration.
 * 
 * Functional Utility: Re-hashes existing entries into a new, larger table buffer.
 */
__global__ void reshape_table(struct Pair *old_table, struct Pair *new_table, int old_size, int new_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int old_key, new_key;
	int hash, i, count;

	if (idx > old_size || old_table[idx].key == KEY_INVALID) {
		return;
	}

	new_key = old_table[idx].key;
	hash = HASH_COMPUTE(new_key, HASH_A, HASH_B, new_size);
	/**
	 * Block Logic: Linear probe search for re-insertion.
	 * Invariant: Probes until an empty slot is atomically claimed in the new table.
	 */
	for (i = hash, count = 0; count < new_size; i = (i + 1) % new_size, count++) {
		old_key = atomicCAS(&(new_table[i].key), KEY_INVALID, new_key);
		if (old_key == KEY_INVALID) {
			new_table[i].value = old_table[idx].value;
			return;
		}
	}
}


/**
 * @brief CUDA kernel for parallel batch insertion.
 * 
 * Functional Utility: Uses atomicCAS to claim slots or update existing keys. 
 * Increments an update counter if an existing key is modified.
 */
__global__ void insert_table(struct Pair *table, int *keys, int *values, int *num_updates, int num_pairs, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j, count;
	int old_key;
	int hash;

	if (idx >= num_pairs)
		return;

	hash = HASH_COMPUTE(keys[idx], HASH_A, HASH_B, size);

	// Block Logic: Multi-pass insertion sequence.
	for (i = 0; i < num_pairs; i++) {
		/**
		 * Block Logic: Circular linear probe sequence.
		 * Logic: Probes the entire table buffer to find a valid slot or matching key.
		 */
		for (j = hash, count = 0; count < size; j = (j + 1) % size, count++) {
			old_key = atomicCAS(&(table[j].key), KEY_INVALID, keys[idx]);
			if (old_key == KEY_INVALID || old_key == keys[idx]) {
				table[j].value = values[idx];
				
				if (old_key != KEY_INVALID) {
					
					// Logic: Tracks key overrides for occupancy reconciliation.
					atomicAdd(&(num_updates[0]), 1);
				}
				return;
			}
		}
	}
}



/**
 * @brief CUDA kernel for parallel batch retrieval.
 */
__global__ void get_table(struct Pair *table, int *keys, int *values, int num_pairs, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j, count;
	int hash;

	if (idx >= num_pairs)
		return;

	hash = HASH_COMPUTE(keys[idx], HASH_A, HASH_B, size);

	// Block Logic: Search probe sequence.
	for (i = 0; i < num_pairs; i++) {
		for (j = hash, count = 0; count < size; j = (j + 1) % size, count++) {
			if (table[j].key == keys[idx]) {
				values[idx] = table[j].value;
				return;
			}
		}
	}
}



/**
 * @brief Constructor: Initializes the hash table buffer in global device memory.
 */
GpuHashTable::GpuHashTable(int size) {
	hash_table = NULL;
	table_pairs = 0;
	table_size = size;
	cudaError_t rc;

	
	rc = cudaMalloc((void **) &hash_table, size * sizeof(struct Pair));
	if (rc != cudaSuccess) {
		cout << "Mem alloc error" << endl;
		return;
	}
	rc = cudaMemset(hash_table, 0, size * sizeof(struct Pair));
	if (rc != cudaSuccess) {
		cout << "Mem set error" << endl;
		return;
	}
}


/**
 * @brief Destructor: Frees the device-resident hash table.
 */
GpuHashTable::~GpuHashTable() {
	
	cudaFree(hash_table);
	hash_table = NULL;
}


/**
 * @brief Rebuilds the hash table with a new capacity.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t rc;
	int num_blocks;

	if (hash_table == NULL) {
		cout << "Table not created" << endl;
		return;
	}

	struct Pair *new_hash_table = NULL;
	int new_size = numBucketsReshape;
	
	rc = cudaMalloc((void **) &new_hash_table, numBucketsReshape * sizeof(struct Pair));
	if (rc != cudaSuccess) {
		cout << "Reshape alloc error" << endl;
		return;
	}

	rc = cudaMemset(new_hash_table, 0, numBucketsReshape * sizeof(struct Pair));
	if (rc != cudaSuccess) {
		cout << "Reshape set error" << endl;
		return;
	}
	
	num_blocks = table_size/THREADS_PER_BLOCK;
	if (table_size % THREADS_PER_BLOCK != 0)


		num_blocks++;
	reshape_table>>(hash_table, new_hash_table, table_size, new_size);
	cudaDeviceSynchronize();

	
	cudaFree(hash_table);
	hash_table = new_hash_table;
	table_size = new_size;
}


/**
 * @brief Performs batch parallel insertion with automated capacity management.
 * 
 * Logic: Triggers expansion if load factor exceeds 90%, and reconciles the 
 * actual number of new entries by subtracting updates.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *devKeys, *devValues = 0;
	int num_blocks = 0;
	int *num_updates = 0;
	int *host_updates = 0;

	if (hash_table == NULL) {
		cout << "Table not created" << endl;
		return false;
	}

	host_updates = (int *) malloc(sizeof(int));

	
	cudaMalloc((void **) &devKeys, numKeys * sizeof(int));
    cudaMalloc((void **) &devValues, numKeys * sizeof(int));
    cudaMalloc((void **) &num_updates, sizeof(int));
    if (devKeys == 0 || devValues == 0 || num_updates == 0) {
    	cout << "Insert alloc error" << endl;
    	return false;
    }

    cudaMemset(num_updates, 0, sizeof(int));

    cudaMemcpy(devKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);


    num_blocks = numKeys/THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK != 0)
		num_blocks++;

	// Adaptive Scaling: Resizes to maintain O(1) performance targets.
	if ((float)(numKeys + table_pairs)/table_size >= MAX_LOAD_FACTOR) {
		
		int new_size = (int)((numKeys + table_pairs)/MIN_LOAD_FACTOR);
		reshape(new_size);
	}

	
	num_blocks = numKeys/THREADS_PER_BLOCK;


	if (numKeys % THREADS_PER_BLOCK != 0)
		num_blocks++;
	insert_table>>(hash_table, devKeys, devValues, num_updates, numKeys, table_size);
	cudaDeviceSynchronize();

	cudaMemcpy(host_updates, num_updates, sizeof(int), cudaMemcpyDeviceToHost);
	
    // Logic: Corrects the pair count by excluding existing keys that were updated.
    table_pairs = table_pairs + numKeys - host_updates[0];
    

    if ((float)table_pairs/table_size < MIN_LOAD_FACTOR) {
    	
    	int new_size = (int)(table_pairs/MIN_LOAD_FACTOR);
		reshape(new_size);
    }
	
	


	cudaFree(devKeys);
	cudaFree(devValues);
	cudaFree(num_updates);
	free(host_updates);
	return true;
}


/**
 * @brief Performs batch parallel retrieval.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *hostValues = 0;
	int *devValues = 0;
	int *devKeys = 0;
	int num_blocks;

	if (hash_table == NULL) {
		cout << "Table not created" << endl;
		return NULL;
	}
	
	hostValues = (int *) malloc(numKeys * sizeof(int));
	cudaMalloc((void **) &devKeys, numKeys * sizeof(int));
    cudaMalloc((void **) &devValues, numKeys * sizeof(int));
	if (hostValues == 0 || devValues == 0 || devKeys == 0) {
    	cout << "Get alloc error" << endl;
    	return NULL;
    }

    cudaMemcpy(devKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    num_blocks = numKeys/THREADS_PER_BLOCK;
    if (numKeys % THREADS_PER_BLOCK != 0)
    	num_blocks++;
    get_table>>(hash_table, devKeys, devValues, numKeys, table_size);
    cudaDeviceSynchronize();

    cudaMemcpy(hostValues, devValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

    
    cudaFree(devKeys);
    cudaFree(devValues);
	return hostValues;
}


/**
 * @brief Current saturation ratio.
 */
float GpuHashTable::loadFactor() {
	if (hash_table == NULL) {
		cout << "Table not created" << endl;
		return 0.f;
	}
	if (table_size == 0) {
		cout << "Table is empty" << endl;
		return 0.f;
	}
	if (table_pairs > table_size) {
		cout << "Error: nr of pairs exceed size" << endl;
		return 0.f;
	}
	float load_factor = (float) table_pairs/table_size;
	return load_factor; 
}


// ... primeList documentation ...

/**
 * @brief Modular hash functions.
 */
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

/**
 * @struct Pair
 * @brief Representation of a key-value mapping on the GPU.
 */
struct Pair {
	int key;
	int value;
};




/**
 * @class GpuHashTable
 * @brief Host controller for Managing the GPU hash mapping lifecycle.
 */
class GpuHashTable
{
	struct Pair *hash_table;  // Device pointer to the hash structure.
	long table_size;          // Total capacity.
	long table_pairs;         // Actual count of unique entries.

	public:
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

