/**
 * @file gpu_hashtable.cu
 * @brief Batch-optimized GPU Hash Table with adaptive buffer management.
 * 
 * This module implements a high-performance hash table on the GPU, optimized for 
 * batch operations. it utilizes internal temporary buffers to manage host-device 
 * data flow and features a robust circular linear probing algorithm. The system 
 * proactively monitors occupancy and automatically scales capacity to maintain 
 * O(1) lookup performance.
 * 
 * Algorithm: Open addressing with circular linear probing and batch-based resizing.
 * Memory Model: Global memory (cudaMalloc) for primary storage and staging buffers.
 * Domain: HPC, Parallel Data Structures.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

int nb_threads = 64;

/**
 * @brief CUDA kernel for parallel entry insertion.
 * 
 * functional Utility: Performs atomic reservations using CAS. Tracks key updates 
 * vs new insertions via a shared device counter.
 * 
 * @param new_inserted Pointer to device counter for tracking redundant ops.
 */
__global__ void insert_element(int *keys, int *values, int n, struct Nod *hashmap, int size_hash, int *new_inserted)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx >= n)
        return;
    // Logic: Initial hash calculation using prime modular scheme.
    int index = ((long)abs(keys[idx]) * 13169977) % 5351951779 % size_hash;
    unsigned int val = values[idx];
    unsigned int key = keys[idx];
    int old;

    /**
     * Block Logic: Circular probe sequence.
     * Invariant: Probes until an available slot is found or the key is matched.
     */
    while(1)
    {
        if(hashmap[index].key == key)
        {
            // Logic: Case 1 - Key already exists (Update).
            hashmap[index].value = val;
            atomicAdd(new_inserted, 1);
            break;
        }
        // Logic: Case 2 - Attempt to claim empty slot (Insert).
        old = atomicCAS(&hashmap[index].key, 0, key);
        if (old == 0)
        {
            hashmap[index].value = val;
            break;
        }
        // Logic: Case 3 - Collision; wrap around if necessary.
        index++;
        if (index == size_hash)
            index = 0;
    }  
}

/**
 * @brief CUDA kernel for parallel entry retrieval.
 */
__global__ void get_elements(int *keys, int *values, int n, struct Nod *hashmap, int size_hash)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    int index = ((long)abs(keys[idx]) * 13169977) % 5351951779 % size_hash;
    
    // Block Logic: Search probe traversal.
    while(1)
    {
        if(hashmap[index].key == keys[idx])
        {
            values[idx] = hashmap[index].value;
            break;
        }
        index++;
        if (index == size_hash)
            index = 0;
    }
}

/**
 * @brief CUDA kernel for data migration during table expansion.
 */
__global__ void reshapeHashmap(struct Nod *oldHashmap, struct Nod *newHashmap, int old_n, int new_n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= old_n || oldHashmap[idx].key == 0)
        return;
    int index = ((long)abs((int)oldHashmap[idx].key) * 13169977) % 5351951779 % new_n;
    int old;
    
    // Logic: Re-hashes existing elements into the larger destination buffer.
    while(1)
    {
        old = atomicCAS(&newHashmap[index].key, 0, oldHashmap[idx].key);
        if (old == 0)
        {
            newHashmap[index].value = oldHashmap[idx].value;
            break;
        }
        index++;
        if (index == new_n)
            index = 0;
    }
}


/**
 * @brief Constructor: Initializes table and pre-allocates batch staging buffers.
 */
GpuHashTable::GpuHashTable(int size2) {
    size = size2;
    batch_size = 10; 
    used_size = 0;
    // Memory Hierarchy: Global memory allocation for entries and staging vectors.
    cudaMalloc((void **) &hashmap, size * sizeof(struct Nod));
    cudaMalloc((void **) &new_inserted, 1 * sizeof(int));
    cudaMemset(hashmap, 0, size * sizeof(struct Nod));
    cudaMalloc((void **) &keys_to_insert, batch_size * sizeof(int));
    cudaMalloc((void **) &values_to_insert, batch_size * sizeof(int));
    if(hashmap == 0 || keys_to_insert == 0 || values_to_insert == 0 || new_inserted == 0)
    {
        printf("problema la alocare\n");
        
    }
}


/**
 * @brief Destructor: Releases all allocated device and staging memory.
 */
GpuHashTable::~GpuHashTable() {
    cudaFree(hashmap);
    cudaFree(keys_to_insert);
    cudaFree(values_to_insert);
    cudaFree(new_inserted);
}


/**
 * @brief Resizes the hash table using re-allocation and re-hashing.
 * 
 * Logic: Scales capacity by ~17% beyond the requested increase to maintain sparse 
 * distribution and low collision rates.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
    numBucketsReshape = ((used_size + numBucketsReshape) * 1.178);
    struct Nod *new_hashmap;
    cudaMalloc((void **) &new_hashmap, numBucketsReshape * sizeof(struct Nod));
    if (new_hashmap == 0)
    {
        printf("eroare la alocare in reshape\n");
        return;
    }
    cudaMemset(new_hashmap, 0, numBucketsReshape * sizeof(struct Nod));
    int nrblocks = size / nb_threads;
    if(size % nb_threads != 0)


        nrblocks++;
    
    // Synchronization: Ensures all mappings are migrated before cleaning up old memory.
    reshapeHashmap>> (hashmap, new_hashmap, size, numBucketsReshape);
    cudaDeviceSynchronize();
    cudaFree(hashmap);
    hashmap = new_hashmap;
    size = numBucketsReshape; 
}


/**
 * @brief Performs host-initiated batch parallel insertion.
 * 
 * Logic: Dynamically expands staging buffers and monitors occupancy to 
 * trigger automatic expansion. Reconciles unique element count via updates counter.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {

    // Adaptive Scaling: Ensures performance by maintaining density targets.
    if (used_size + numKeys > size)
        reshape(numKeys);
    
    // Optimization: Reuse or resize internal staging buffers to minimize re-allocation overhead.
    if (batch_size < numKeys)
    {
        cudaFree(keys_to_insert);
        cudaFree(values_to_insert);
        batch_size = numKeys;
        cudaMalloc((void **) &keys_to_insert, batch_size * sizeof(int));
        cudaMalloc((void **) &values_to_insert, batch_size * sizeof(int));
    }
    int nr_noi = 0;
    cudaMemset(new_inserted, 0, sizeof(int));
    if(hashmap == 0 || keys_to_insert == 0 || values_to_insert == 0)


        return false;
    cudaMemcpy(keys_to_insert, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(values_to_insert, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    int nrblocks = numKeys / nb_threads;


    if(numKeys % nb_threads != 0)
        nrblocks++;
    insert_element>> (keys_to_insert, values_to_insert, numKeys, hashmap, size, new_inserted);
    cudaDeviceSynchronize();
    
    // Logic: Reconciliation of unique insertion count (Batch - Updates).
    cudaMemcpy(&nr_noi, new_inserted, sizeof(int), cudaMemcpyDeviceToHost);
    used_size += (numKeys - nr_noi);
    if (loadFactor() < 0.8)
        reshape(ceil(used_size / 0.85));
	return true;
}


/**
 * @brief Performs host-initiated batch parallel retrieval.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
    
    if (batch_size < numKeys)
    {
        cudaFree(keys_to_insert);
        cudaFree(values_to_insert);
        batch_size = numKeys;
        cudaMalloc((void **) &keys_to_insert, batch_size * sizeof(int));
        cudaMalloc((void **) &values_to_insert, batch_size * sizeof(int));
    }
 
    if(hashmap == 0 || keys_to_insert == 0 || values_to_insert == 0)
        return NULL;
    cudaMemcpy(keys_to_insert, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    int nrblocks = numKeys / nb_threads;
    if(numKeys % nb_threads != 0)
        nrblocks++;
    get_elements>>(keys_to_insert, values_to_insert, numKeys, hashmap, size);
    cudaDeviceSynchronize();
    int *result = (int *)malloc(numKeys * sizeof(int));
    cudaMemcpy(result, values_to_insert, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
    
	return result;
}


/**
 * @brief Returns the current occupancy ratio.
 */
float GpuHashTable::loadFactor() {
	return (float) used_size / size;
    
}


// ... hash functions and primeList ...

/**
 * @struct Nod
 * @brief Atomic storage unit for a key-value mapping on the GPU.
 */
struct Nod
{
    uint32_t key, value;
};

/**
 * @class GpuHashTable
 * @brief Host-side manager for GPU-resident hash storage and staging.
 */
class GpuHashTable
{
    struct Nod *hashmap;                // Pointer to device entry buffer.
    uint32_t size, used_size;           // Capacity and occupancy metrics.
    uint32_t batch_size;                // Current size of staging buffers.
    int *keys_to_insert, *values_to_insert; // Staging vectors.
    int *new_inserted;                  // Device-side atomic reconciliation counter.
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

