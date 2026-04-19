
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

__device__ int kernelHashFunction(int data, int limit) {
	return ((long)abs(data) * 653267llu) % 56564691976601587llu % limit;
}

__global__ void kernelInitHashTable(DeviceHashTable *hashTable, GpuEntry *entries) {
	hashTable->entries = entries;
	hashTable->elements = 0;
}



__global__ void kernelInsertHashTable(DeviceHashTable *hashTable, int *keys, int *values, int numKey) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int hashID;
	int old;

	if (idx >= numKey) {
		return;
	}

	hashID = kernelHashFunction(keys[idx], hashTable->size);
	cudaDeviceSynchronize();

	
	while (hashTable->entries[hashID].key != 0 && hashTable->entries[hashID].key != keys[idx]) {
		hashID = (hashID + 1) % hashTable->size;
	}

	
	if (hashTable->entries[hashID].key == keys[idx]) {
		atomicExch(&hashTable->entries[hashID].value, values[idx]);
	} else {
		old = atomicCAS(&hashTable->entries[hashID].key, 0, keys[idx]);
		
		while (old != 0) {
			hashID = (hashID + 1) % hashTable->size;
			old = atomicCAS(&hashTable->entries[hashID].key, 0, keys[idx]);
		}
		
		atomicExch(&hashTable->entries[hashID].value, values[idx]);
		atomicAdd(&hashTable->elements, 1);
	}
}

__global__ void kernelCopyHashTable(DeviceHashTable *dstHashTable, DeviceHashTable *srcHashTable) {
	dstHashTable->entries = srcHashTable->entries;
	dstHashTable->size = srcHashTable->size;
	dstHashTable->elements = srcHashTable->elements;
}

__global__ void kernelResizeHashTable(DeviceHashTable *newHashTable, DeviceHashTable *oldHashTable) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int hashID;
	int old;

	if (idx >= oldHashTable->size) {
		return;
	}

	int key = oldHashTable->entries[idx].key;
	int value = oldHashTable->entries[idx].value;
	

	if (key == 0) {
		return;
	}

	hashID = kernelHashFunction(key, newHashTable->size);
	cudaDeviceSynchronize();
	
	while (newHashTable->entries[hashID].key != 0 && newHashTable->entries[hashID].key != key) {
		hashID = (hashID + 1) % newHashTable->size;
	}

	
	if (newHashTable->entries[hashID].key == key) {
		atomicExch(&newHashTable->entries[hashID].value, value);
	} else {
		old = atomicCAS(&newHashTable->entries[hashID].key, 0, key);
		
		while (old != 0) {
			hashID = (hashID + 1) % newHashTable->size;
			old = atomicCAS(&newHashTable->entries[hashID].key, 0, key);
		}
		
		atomicExch(&newHashTable->entries[hashID].value, value);
		atomicAdd(&newHashTable->elements, 1);
	}
}



__global__ void kernelGetHashTable(DeviceHashTable *hashTable, int *keys, int *values, int numKeys) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int hashID;
	int initialPos;

	if (idx >= numKeys) {
		return;
	}
	hashID = kernelHashFunction(keys[idx], hashTable->size);
	
	if (hashTable->entries[hashID].key == keys[idx]) {
		values[idx] = hashTable->entries[hashID].value;
	} else {
		initialPos = hashID;
		hashID = (hashID + 1) % hashTable->size;
		
		while (hashTable->entries[hashID].key != keys[idx] && initialPos != hashID) {
			hashID = (hashID + 1) % hashTable->size;
		}
		
		if (initialPos == hashID) {
			values[idx] = 0;
		} else {
			
			values[idx] = hashTable->entries[hashID].value;
		}
	}
	
}

__global__ void kernelGetEntries(GpuEntry *entries, DeviceHashTable *hashTable) {
	
	
	entries = hashTable->entries;
}


/**
 * @brief Constructor: Initializes table and calculates optimal launch configurations.
 */
GpuHashTable::GpuHashTable(int size) {
	initHashTable(size);
}

/**
 * @brief Internal helper to allocate and initialize device memory for the hash table.
 */
void GpuHashTable::initHashTable(int size) {
	GpuEntry *entries;


	// Memory Hierarchy: Global memory allocation for entries.
	cudaMalloc((void **)&entries, size * sizeof(GpuEntry));
	cudaMemset(entries, 0, size * sizeof(GpuEntry));
	// Memory Hierarchy: Global memory allocation for the metadata structure.
	cudaMalloc((void **)&this->hashTable, sizeof(DeviceHashTable));
	cudaMemcpy(&this->hashTable->size, &size, sizeof(int), cudaMemcpyHostToDevice);
	kernelInitHashTable>>(this->hashTable, entries);
	cudaDeviceSynchronize();
}


GpuHashTable::~GpuHashTable() {
	GpuEntry *entries = NULL;
	kernelGetEntries>>(entries, this->hashTable);
	cudaFree(entries);
	cudaFree(this->hashTable);
}


/**
 * @brief Resizes the hash table using parallel re-hashing.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int hostSize, hostElems;
	DeviceHashTable *oldHashTable;
	GpuEntry *oldEntries = NULL;

	// Synchronization: Transfers current metadata to host for launch calculation.
	cudaMemcpy(&hostSize, &this->hashTable->size, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&hostElems, &this->hashTable->elements, sizeof(int), cudaMemcpyDeviceToHost);
	if (numBucketsReshape < hostElems) {
		return;
	}

	cudaMalloc((void **)&oldHashTable, sizeof(DeviceHashTable));
	if (oldHashTable == NULL) {
		return;
	}
	kernelCopyHashTable>>(oldHashTable, this->hashTable);

	this->hashTable = NULL;
	initHashTable(numBucketsReshape);

	const size_t block_size = hostSize / 4 + 4 < 1024 ? hostSize / 4 + 4 : 1024;
  	size_t blocks_no = hostSize / block_size;
	if (hostSize % block_size) 
		++blocks_no;

	kernelResizeHashTable>>(this->hashTable, oldHashTable);
	cudaDeviceSynchronize();

	kernelGetEntries>>(oldEntries, oldHashTable);
	cudaFree(oldEntries);
	cudaFree(oldHashTable);
}


/**
 * @brief Performs batch parallel insertion.
 * 
 * Logic: Monitors load factor and triggers expansion if density > 95%.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *deviceKeys;
	int *deviceValues;
	int hostSize;
	int hostElems;

	cudaMemcpy(&hostSize, &this->hashTable->size, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&hostElems, &this->hashTable->elements, sizeof(int), cudaMemcpyDeviceToHost);
	// Adaptive Scaling: Ensures performance by maintaining density targets.
	if ((hostElems + numKeys) * 1.0f / hostSize > 0.95f) {

		DeviceHashTable *oldHashTable;
		GpuEntry *oldEntries = NULL;
		const size_t block_size = hostSize / 4 + 4 < 1024 ? hostSize / 4 + 4 : 1024;
	  	size_t blocks_no = hostSize / block_size;
		if (hostSize % block_size) 
			++blocks_no;
		cudaMalloc((void **)&oldHashTable, sizeof(DeviceHashTable));
		if (oldHashTable == NULL) {
			return false;
		}
		kernelCopyHashTable>>(oldHashTable, this->hashTable);
		this->hashTable = NULL;
		initHashTable((hostElems + numKeys) * 1.2f);

		kernelResizeHashTable>>(this->hashTable, oldHashTable);
		cudaDeviceSynchronize();
		kernelGetEntries>>(oldEntries, oldHashTable);
		cudaFree(oldEntries);
		cudaFree(oldHashTable);
	}

	cudaMalloc((void **)&deviceKeys, numKeys * sizeof(int));
	cudaMalloc((void **)&deviceValues, numKeys * sizeof(int));
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	const size_t block_size = numKeys / 4 + 4 < 1024 ? numKeys / 4 + 4 : 1024;
  	size_t blocks_no = numKeys / block_size;

	if (numKeys % block_size) 
		++blocks_no;

	kernelInsertHashTable>>(this->hashTable, deviceKeys, deviceValues, numKeys);
	cudaDeviceSynchronize();
	cudaFree(deviceValues);
	cudaFree(deviceKeys);

	return true;
}


/**
 * @brief Performs batch parallel retrieval of values.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *values;
	int *deviceValues;
	int *deviceKeys;

	cudaMalloc((void **)&deviceKeys, numKeys * sizeof(int));
	cudaMalloc((void **)&deviceValues, numKeys * sizeof(int));
	if (deviceValues == NULL || deviceKeys == NULL) {
		return NULL;
	}

	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	const size_t block_size = 1024;
	size_t blocks_no = numKeys / block_size;
	if (numKeys % block_size)
		++blocks_no;

	kernelGetHashTable>>(this->hashTable, deviceKeys, deviceValues, numKeys);
	cudaDeviceSynchronize();
	values = (int *)malloc(numKeys * sizeof(int));
	cudaMemcpy(values, deviceValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	return values;
}


/**
 * @brief Returns the current occupancy ratio.
 */
float GpuHashTable::loadFactor() {
	int hostNumElements, hostSize;
	cudaMemcpy(&hostNumElements, &this->hashTable->elements, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&hostSize, &this->hashTable->size, sizeof(int), cudaMemcpyDeviceToHost);
	return hostNumElements * 1.0f / hostSize; 
}


// ... primeList documentation ...

/**
 * @brief Modular hash functions for device indexing.
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
 * @struct GpuEntry
 * @brief Atomic storage unit for a key-value mapping on the GPU.
 */
struct GpuEntry
{
	uint32_t key;
	uint32_t value;
};

/**
 * @struct DeviceHashTable
 * @brief Internal metadata and entry pointers for the device-resident hash map.
 */
struct DeviceHashTable
{
	GpuEntry *entries;  // Pointer to device entry array.
	uint32_t elements;  // Active element count.
	uint32_t size;      // Capacity of the entry buffer.
};




/**
 * @class GpuHashTable
 * @brief Host controller for Managing the GPU hash mapping lifecycle.
 */
class GpuHashTable
{
	private:
	DeviceHashTable *hashTable; // Persistent device state.
	public:
		GpuHashTable(int size);
		void initHashTable(int size);

		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		
		float loadFactor();
		void occupancy();
		void print(string info);
	
		~GpuHashTable();
};

#endif

