/**
 * @file gpu_hashtable.cu
 * @brief Vector-based GPU Hash Table with atomic linear probing.
 * 
 * This implementation manages separate device-side vectors for keys and values to 
 * optimize memory access patterns during parallel operations. It uses atomic 
 * Compare-And-Swap (CAS) for collision resolution via linear probing.
 * 
 * Algorithm: Open addressing with linear probing.
 * Memory Model: Separate global memory vectors for keys and values.
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
 * @brief Constructor: Initializes dual-vector storage on the GPU.
 */
GpuHashTable::GpuHashTable(int size) {
	
	totalSize = size;
	currentSize = 0;
	// Memory Hierarchy: Separate global memory allocations for keys and values.
	cudaMalloc(&keysVector, size * sizeof(int));
	cudaMalloc(&valuesVector, size * sizeof(int));
	cudaMemset(keysVector, KEY_INVALID, size * sizeof(int));
	cudaMemset(valuesVector, KEY_INVALID, size * sizeof(int));	
}


/**
 * @brief Destructor: Releases device vector memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(keysVector);
	cudaFree(valuesVector);
}



/**
 * @brief CUDA kernel for parallel data migration.
 * 
 * Functional Utility: Transfers and re-hashes existing key-value pairs into 
 * a larger memory space.
 */
__global__ void reshape_func(int *oldKeys, int *oldValues, int oldSize, int *newKeys, int* newValues, int newSize)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < oldSize)
    {
    	
    	if(oldKeys[i] == KEY_INVALID)
    		return;
    	int index = hash_function(oldKeys[i], newSize);
    	int result = atomicCAS(&newKeys[index], KEY_INVALID, oldKeys[i]);
    	
    	/**
    	 * Block Logic: Linear probe resolution for re-hashed keys.
    	 */
    	while(result != 0)
    	{
    		index = (index + 1) % newSize;
    		result = atomicCAS(&newKeys[index], KEY_INVALID, oldKeys[i]);
    	}
    	
    	newValues[index] = oldValues[i];
    }
}

/**
 * @brief Dynamically expands table capacity.
 * 
 * Logic: Allocates new vectors, clears them, and invokes reshape_func to 
 * migrate the current data set.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	
	int *oldKeys = keysVector;
	int *oldValues = valuesVector;
	int oldSize = totalSize;
	cudaMalloc(&keysVector, numBucketsReshape * sizeof(int));
	cudaMalloc(&valuesVector, numBucketsReshape * sizeof(int));
	cudaMemset(keysVector, KEY_INVALID, numBucketsReshape * sizeof(int));
	cudaMemset(valuesVector, KEY_INVALID, numBucketsReshape * sizeof(int));
	totalSize = numBucketsReshape;
	const size_t block_size = 8;
    size_t blocks_no = oldSize / block_size;
    if (oldSize % block_size)
    	blocks_no++;
    
    reshape_func>>(oldKeys, oldValues, oldSize, keysVector, valuesVector, totalSize);
    cudaDeviceSynchronize();
	cudaFree(oldKeys);
	cudaFree(oldValues);
}



/**
 * @brief CUDA kernel for massively parallel entry insertion.
 * 
 * Functional Utility: Performs atomic reservations in the key vector and 
 * updates corresponding indices in the value vector.
 */
__global__ void insert_func(int *vramKeys, int *vramValues, int numKeys, int *keysVector, int *valuesVector, int totalSize)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < numKeys)
	{
		int index = hash_function(vramKeys[i], totalSize);
		int result = atomicCAS(&keysVector[index], KEY_INVALID, vramKeys[i]);
		
		/**
		 * Block Logic: Atomic probe sequence.
		 * Logic: Iteratively probes for an available slot or matches an existing key 
		 * for concurrent updates.
		 */
		while(result != 0)
		{
			// Optimization: Double-check key consistency after CAS failure.
			keysVector[index] = result;
			
			if(keysVector[index] == vramKeys[i])
			{
				valuesVector[index] = vramValues[i];
				return;
			}
			index = (index + 1) % totalSize;
			result = atomicCAS(&keysVector[index], KEY_INVALID, vramKeys[i]);
		}
		valuesVector[index] = vramValues[i];
	}
}

/**
 * @brief Performs host-initiated batch parallel insertion.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	currentSize += numKeys;
	
	// Adaptive Scaling: Resizes if the load exceeds buffer capacity.
	if(currentSize >= totalSize)
		reshape(currentSize * 1.1 + 1);
	int *vramKeys = 0;
	int *vramValues = 0;
	cudaMalloc((void **) &vramKeys, numKeys * sizeof(int));


	cudaMalloc((void **) &vramValues, numKeys * sizeof(int));
	cudaMemcpy(vramKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(vramValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	const size_t block_size = 8;
	size_t blocks_no = numKeys / block_size;
	if (numKeys % block_size) 
  		blocks_no++;
  	
	insert_func>>(vramKeys, vramValues, numKeys, keysVector, valuesVector, totalSize);
	cudaDeviceSynchronize();
	return true;
}



/**
 * @brief CUDA kernel for parallel key search.
 */
__global__ void search_func(int *vramKeys, int *resultValues, int numKeys, int *keysVector, int *valuesVector, int totalSize)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < numKeys)
	{
		int index = hash_function(vramKeys[i], totalSize);
		
		/**
		 * Block Logic: Linear probing search pass.
		 */
		while(keysVector[index] != vramKeys[i])
		{
			index = (index + 1) % totalSize;
		}
		resultValues[i] = valuesVector[index];
	}
}

/**
 * @brief Performs host-initiated batch retrieval.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *vramKeys = 0;
	int *resultValues = 0;
	cudaMalloc((void **) &vramKeys, numKeys * sizeof(int));


	cudaMalloc((void **) &resultValues, numKeys * sizeof(int));
	cudaMemcpy(vramKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	const size_t block_size = 8;
	size_t blocks_no = numKeys / block_size;
	if (numKeys % block_size) 
  		blocks_no++;
	search_func>>(vramKeys, resultValues, numKeys, keysVector, valuesVector, totalSize);
	cudaDeviceSynchronize();
	int *cpu_result = (int *) malloc(numKeys * sizeof(int));
	cudaMemcpy(cpu_result, resultValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	return cpu_result;
}


/**
 * @brief Current saturation ratio.
 */
float GpuHashTable::loadFactor() {
	return ((float) currentSize) / totalSize;
}


// ... primeList documentation omitted ...

/**
 * @brief Hashing algorithms based on modular arithmetic.
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
 * @brief Primary hash index calculation for device operations.
 */
__host__ __device__ int hash_function(int data, int limit)
{
	return ((long)abs(data) * 17399181177241llu) % 132745199llu % limit;
}




/**
 * @class GpuHashTable
 * @brief Controller for managing dual-vector device-side storage.
 */
class GpuHashTable
{
	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		
		float loadFactor();
		void occupancy();
		void print(string info);
	
		~GpuHashTable();

		
		int totalSize;      // Maximum entries allowed in current allocation.
		int currentSize;    // High-water mark of entries inserted.
		int *keysVector;    // Device pointer to key array.
		int *valuesVector;  // Device pointer to value array.
};

#endif

