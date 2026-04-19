/**
 * @file gpu_hashtable.cu
 * @brief Unified Memory-based GPU Hash Table with Linear Probing.
 * 
 * This implementation utilizes CUDA's Managed Memory (Unified Memory) to simplify 
 * host-device data transitions. It employs a single-tier hash table with linear 
 * probing for collision resolution and hardware-accelerated atomic operations 
 * for thread-safe concurrent access.
 * 
 * Algorithm: Single-hash open addressing with linear probing.
 * Memory Model: Unified Memory (cudaMallocManaged) for global table storage.
 * Domain: HPC, GPGPU Data Structures.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief CUDA kernel for parallel entry insertion.
 * 
 * Functional Utility: Uses atomicCAS to claim empty slots (key=0) or update existing 
 * keys. Tracks successful insertions via a global counter for load factor management.
 * 
 * @param keys Source keys from host.
 * @param values Source values from host.
 * @param table Pointer to the GPU-resident hash table.
 * @param numKeys Batch size.
 * @param dimensiune Current table capacity.
 * @param nr_inserate Global counter for successful new insertions.
 */
__global__ void insert_kernel(int *keys, int *values, struct element *table, int numKeys, int dimensiune, unsigned int *nr_inserate) {
	
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x; 
	if (i >= numKeys) {
		return;
	}
	
	unsigned int place = hash3(keys[i], dimensiune);

	/**
	 * Block Logic: Linear probing sequence.
	 * Invariant: Probes the table until an empty slot is claimed or the key is matched.
	 */
	while(1) {
		
		// Logic: Attempt to claim an empty slot (key=0) atomically.
		unsigned int ret = atomicCAS(&(table[place].key), 0, (unsigned int) keys[i]);
		if (ret == 0) {
			table[place].value = values[i];
			// Inline: Increment global counter for occupancy tracking.
			atomicAdd(&nr_inserate[0], 1);
			return;
		}
		else { 
			// Logic: Key already exists; perform update.
			if (ret == keys[i]) {
				table[place].value = values[i];
				return;
			}
		}
		// Logic: Circular linear probing step.
		place++;
		if (place >= dimensiune) { 
			place = 0;
		}
	}
}


/**
 * @brief CUDA kernel for parallel value retrieval.
 * 
 * Functional Utility: Implements the search phase of the linear probing algorithm 
 * to find key-value mappings in a concurrent environment.
 */
__global__ void  get_kernel(int *keys, int *values, struct element *table , int numKeys, int dimensiune) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x; 
	if (i >= numKeys) {
		return;
	}
	unsigned int place = hash3(keys[i], dimensiune);

	/**
	 * Block Logic: Search probe sequence.
	 * Invariant: Continues probing until the target key is located.
	 */
	while(1) {
		
		if (table[place].key == keys[i]) {
			values[i] = table[place].value;
			return;
		}
		place++;
		if (place >= dimensiune) {
			place = 0;
		}
	}
}




/**
 * @brief CUDA kernel for parallel data migration during table expansion.
 * 
 * Functional Utility: Efficiently re-hashes existing elements from the old table 
 * into a new, larger table buffer.
 */
__global__ void reshape_kernel(struct element *table_old, struct element *table_new, int dimensiune_old, int dimensiune_new) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= dimensiune_old) {
		return;
	}
	// Logic: Skip empty slots to optimize migration throughput.
	if (table_old[i].key == 0) {
		return;
	}
	unsigned int place = hash3(table_old[i].key, dimensiune_new);

	/**
	 * Block Logic: Re-insertion sequence in the new memory space.
	 */
	while(1) {
		
		unsigned int ret = atomicCAS(&(table_new[place].key), 0, (unsigned int) table_old[i].key);
		if (ret == 0) {
			table_new[place].value = table_old[i].value;
			return;
		}

		place++;
		if (place >= dimensiune_new) {
			place = 0;
		}
	}	
}


/**
 * @brief Constructor: Initializes the hash table using Unified Memory.
 * 
 * Functional Utility: Allocates memory that is automatically managed and accessible 
 * by both CPU and GPU, simplifying the data model.
 * 
 * @param size Initial capacity of the table.
 */
GpuHashTable::GpuHashTable(int size) {

	cudaMallocManaged(&table, size*sizeof(struct element));
	cudaMemset(table, 0, size*sizeof(struct element));

	used_size = 0;
	table_size = size;

}


/**
 * @brief Destructor: Reclaims Unified Memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(table);
}


/**
 * @brief Triggers a global table expansion.
 * 
 * Functional Utility: Allocates a new managed buffer and migrates data via 
 * the reshape_kernel.
 * 
 * @param numBucketsReshape New target capacity.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	
	const size_t block_size = 1024;
	size_t blocks_no = table_size / block_size;
 	if (table_size % block_size) {
		++blocks_no;
	}
	struct element *table_aux;
	cudaMallocManaged(&table_aux, numBucketsReshape*sizeof(struct element));
	cudaMemset(table_aux, 0, numBucketsReshape*sizeof(struct element));

	reshape_kernel>>(table, table_aux, table_size, numBucketsReshape);
	cudaDeviceSynchronize();

	table_size = numBucketsReshape;
	cudaFree(table);
	table = table_aux;

}



/**
 * @brief Batch insertion interface from host memory.
 * 
 * Functional Utility: Handles host-to-device data transfer and occupancy-based 
 * proactive resizing before launching the insertion kernel.
 * 
 * @param keys Host pointer to keys.
 * @param values Host pointer to values.
 * @param numKeys Size of the batch.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	const size_t block_size = 1024;
	size_t blocks_no = numKeys / block_size;
 	if (numKeys % block_size) {
		++blocks_no;
	}
	int *cheie;
	int *valori;
	cudaMalloc((void **) &cheie, sizeof(int) * numKeys);
	cudaMalloc((void **) &valori, sizeof(int) * numKeys);
	cudaMemcpy(cheie, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(valori, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Adaptive Scaling: Resizes when the load factor exceeds 95% to prevent performance degradation.
	if (((float)(used_size + numKeys)) / ((float)(table_size)) >= 0.95f) {
		reshape((int)((used_size + numKeys) / 0.85f));
	}

	unsigned int* nr_inserate = NULL;
	cudaMallocManaged(&nr_inserate, sizeof(unsigned int));
	cudaMemset(nr_inserate, 0, sizeof(unsigned int));

	insert_kernel>>(cheie, valori, table , numKeys, table_size, nr_inserate);
	cudaDeviceSynchronize();
	// Logic: Updates host-side tracking of table occupancy.
	used_size = used_size + *nr_inserate;

	cudaFree(nr_inserate);
	cudaFree(cheie);
	cudaFree(valori);
	return false;
}


/**
 * @brief Batch retrieval interface.
 * 
 * Functional Utility: Retrieves values for a given set of keys in a single 
 * parallel operation.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	const size_t block_size = 1024;
	size_t blocks_no = numKeys / block_size;
	if (numKeys % block_size) {
		++blocks_no;
	}
	int *values = (int*) malloc(numKeys*sizeof(int));
	int *cheie;
	int *valori;
	cudaMalloc((void **) &cheie, sizeof(int) * numKeys);
	cudaMalloc((void **) &valori, sizeof(int) * numKeys);
	cudaMemcpy(cheie, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	get_kernel>>(cheie, valori, table , numKeys, table_size);
	cudaDeviceSynchronize();	
	
	cudaMemcpy(values, valori, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(valori);
	cudaFree(cheie);
	return values;
}


/**
 * @brief Returns the current occupancy ratio of the table.
 */
float GpuHashTable::loadFactor() {
	return used_size /((float)table_size) ; 
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0

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
 * @struct element
 * @brief Representation of a key-value pair in GPU memory.
 */
struct element {
	unsigned int key;
	unsigned int value;
};

// ... primeList documentation omitted for brevity but preserved in file ...

/**
 * @brief Hashing algorithms based on large prime modular arithmetic.
 * 
 * functional Utility: Maps input keys to table indices with high entropy 
 * to minimize clustering.
 */
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
__device__ int hash3(unsigned int data, unsigned int limit) {
	return ((long) data * 135931102921llu) % 2227095190691797llu % limit;
}


/**
 * @class GpuHashTable
 * @brief Host-side handle for managing the GPU-resident hash table.
 */
class GpuHashTable
{
	private: 		
		struct element *table;  // Pointer to Unified Memory table.
		int used_size;          // Count of occupied slots.
		int table_size;         // Total capacity.

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

