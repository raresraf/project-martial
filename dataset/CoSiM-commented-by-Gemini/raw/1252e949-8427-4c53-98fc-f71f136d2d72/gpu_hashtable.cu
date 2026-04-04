/**
 * @file gpu_hashtable.cu
 * @brief A self-contained, robust implementation of a GPU-accelerated hash table.
 * @details This file provides a complete hash table using CUDA. It features an
 * open-addressing, linear-probing collision strategy with a clean and correct
 * `atomicCAS`-based insertion kernel. The implementation includes an efficient,
 * fully device-side reshape kernel (`kernel_copy`) and a dynamic resizing
 * policy based on load factor thresholds.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gpu_hashtable.hpp"

#define PRIME_A	823117
#define PRIME_B 3452434812973

/**
 * @brief Computes a hash value for a given key on the device.
 * @param Key The integer key to hash.
 * @param Size The capacity of the hash table for the modulo operation.
 * @return An integer hash value within the range [0, Size-1].
 */
__device__ int ComputeHash(int Key, int Size)
{
	return ((long long)abs(Key) * PRIME_A) % PRIME_B % Size;
}

/**
 * @brief CUDA kernel to insert or update a batch of key-value pairs.
 * @param keys Device pointer to the input keys.
 * @param values Device pointer to the input values.
 * @param NumElements The number of pairs in the batch.
 * @param Size The capacity of the hash table.
 * @param Container Device pointer to the hash table's data array.
 * @details Each thread handles one key-value pair. It uses linear probing and a
 * race-free `atomicCAS` loop to either claim an empty slot for a new key or
 * update the value of an existing key.
 */
__global__ void kernel_insert(int *keys, int *values, int NumElements, int Size, Pair *Container)
{
	int key, idx, hash, ret;

	idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= NumElements)
		return;

	key = keys[idx];
	hash = ComputeHash(key, Size);

	// Block Logic: Infinite loop for linear probing, broken on successful insertion/update.
	while (true) {
		// Atomic Operation: Attempt to claim an empty slot or find a matching key.
		ret = atomicCAS(&Container[hash].key, KEY_INVALID, key);
		
		// Invariant: If the original value was empty (KEY_INVALID) or already our key,
		// this thread has secured the slot for writing.
		if (ret == KEY_INVALID || ret == key) {
			Container[hash].value = values[idx];
			return;
		}
		
		// Collision: The slot was occupied by another key; probe the next slot.
		hash = (hash + 1) % Size;
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param keys Device pointer to the keys to look up.
 * @param values Device pointer to the output array for found values.
 * @param NumElements The number of keys in the batch.
 * @param Size The capacity of the hash table.
 * @param Container Device pointer to the hash table's data.
 */
__global__ void kernel_get(int *keys, int *values, int NumElements, int Size, Pair *Container)
{
	int idx, key, hash, ret;

	idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= NumElements)
		return;

	key = keys[idx];
	hash = ComputeHash(key, Size);

	// Block Logic: Linear probe until the key is found or an empty slot is encountered
	// (which would be implicit in a well-formed table, though this loop is infinite).
	while (true) {
		// Pre-condition: Check if the key at the current slot matches.
		if (Container[hash].key == key) {
			values[idx] = Container[hash].value;
			return;
		}

		// Probe the next slot.
		hash = (hash + 1) % Size;
	}
}

/**
 * @brief CUDA kernel to rehash all elements from an old table to a new, resized table.
 * @param OldContainer Device pointer to the old hash table data.
 * @param OldSize The capacity of the old table.
 * @param NewContainer Device pointer to the new (empty) hash table data.
 * @param NewSize The capacity of the new table.
 */
__global__ void kernel_copy(Pair *OldContainer, int OldSize, Pair *NewContainer, int NewSize)
{
	int idx, key, hash, ret;

	idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= OldSize)
		return;

	key = OldContainer[idx].key;
	// Pre-condition: Only migrate valid entries.
	if (key == KEY_INVALID)
		return;
		
	hash = ComputeHash(key, NewSize);

	// Block Logic: Linear probing to find an empty slot in the new table.
	while(true) {
		// Atomically claim a slot in the new table.
		ret = atomicCAS(&NewContainer[hash].key, KEY_INVALID, key); 
		if (ret == KEY_INVALID) {
			NewContainer[hash].value = OldContainer[idx].value;
			return;
		}
		// Probe the next slot.
		hash = (hash + 1) % NewSize;
	}
}

/**
 * @brief Constructs a GpuHashTable object.
 */
GpuHashTable::GpuHashTable(int size) {
	cudaError_t error;
	NumElements = 0;
	Size = size;

	// Memory Hierarchy: Allocate the main table storage on the device.
	error = cudaMalloc((void**)&Container, Size * sizeof(struct Pair));
	if (error != cudaSuccess) {
		printf("%s
",  cudaGetErrorString(error));
		return;
	}
	cudaMemset(Container, 0, Size * sizeof(Pair));
}

/**
 * @brief Destroys the GpuHashTable object.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(Container);
}

/**
 * @brief Resizes the hash table and rehashes all existing elements.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	Pair *NewContainer;
	int ret, NewSize = numBucketsReshape;
	unsigned int NumBlocks;

	ret = cudaMalloc(&NewContainer, NewSize * sizeof(Pair));
	if (NewContainer == NULL)
		return;

	cudaMemset(NewContainer, 0, NewSize * sizeof(Pair));
	// Kernel Launch Configuration: Calculate grid size for the copy kernel.
	NumBlocks = (Size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	// Launch the fully device-side copy/rehash kernel.
	kernel_copy<<<NumBlocks, THREADS_PER_BLOCK>>>(Container, Size, NewContainer, NewSize);
	cudaDeviceSynchronize();

	// Free the old table and swap to the new one.
	cudaFree(Container);
	Size = NewSize;
	Container = NewContainer;
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *DeviceKeys, *DeviceValues;
	unsigned int NumBlocks;
	size_t size = numKeys * sizeof(int);

	cudaMalloc(&DeviceKeys, size);
	cudaMalloc(&DeviceValues, size);

	if (DeviceKeys == NULL || DeviceValues == NULL)
		return false;

	// Pre-condition: Check if the load factor will exceed the maximum threshold.
	if ((float)(NumElements + numKeys) / Size >= MAX_LOAD_FACTOR) {
		// Resize the table to bring the load factor back down towards the minimum threshold.
		reshape((int)((NumElements + numKeys) / MIN_LOAD_FACTOR) + 1);
	}

	// Data Transfer: Copy input batch from host to device.
	cudaMemcpy(DeviceKeys, keys, size, cudaMemcpyHostToDevice);
	cudaMemcpy(DeviceValues, values, size, cudaMemcpyHostToDevice);
	
	NumBlocks = (numKeys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	// Launch the insertion kernel.
	kernel_insert<<<NumBlocks, THREADS_PER_BLOCK>>>(DeviceKeys, DeviceValues, numKeys, Size, Container);
	cudaDeviceSynchronize();

	NumElements += numKeys;

	cudaFree(DeviceKeys);
	cudaFree(DeviceValues);

	return true;
}

/**
 * @brief Retrieves values for a batch of keys.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *DeviceValues, *DeviceKeys, *Values;
	unsigned int NumBlocks;
	size_t KeysSize = numKeys * sizeof(int);

	cudaMalloc(&DeviceValues, KeysSize);
	if (DeviceValues == NULL) return NULL;
	cudaMalloc(&DeviceKeys, KeysSize);
	if (DeviceKeys == NULL) return NULL;
	Values = (int *)malloc(KeysSize);
	if (Values == NULL) return NULL;

	cudaMemcpy(DeviceKeys, keys, KeysSize, cudaMemcpyHostToDevice);
	NumBlocks = (numKeys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	// Launch the retrieval kernel.
	kernel_get<<<NumBlocks, THREADS_PER_BLOCK>>>(DeviceKeys, DeviceValues, numKeys, Size, Container);
	cudaDeviceSynchronize();

	// Data Transfer: Copy results from device back to host.
	cudaMemcpy(Values, DeviceValues, KeysSize, cudaMemcpyDeviceToHost);

	cudaFree(DeviceValues);
	cudaFree(DeviceKeys);

	// The caller is responsible for freeing the returned host memory.
	return Values;
}

/**
 * @brief Calculates the current load factor.
 */
float GpuHashTable::loadFactor() {
	return (Size == 0) ? 0 : ((float)NumElements / Size);
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

#define	KEY_INVALID			0
#define THREADS_PER_BLOCK	1024
#define MIN_LOAD_FACTOR		0.80
#define MAX_LOAD_FACTOR		0.97

#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)
	
const size_t primeList[] =
{
	2llu, 3llu, 5llu, 7llu, 11llu, 13llu, 17llu, 23llu, 29llu, 37llu, 47llu,
	59llu, 73llu, 97llu, 127llu, 151llu, 197llu, 251llu, 313llu, 397llu,
	499llu, 631llu, 797llu, 1009llu, 1259llu, 1597llu, 2011llu, 2539llu,
	3203llu, 4027llu, 5087llu, 6421llu, 8089llu, 10193llu, 12853llu, 16193llu,
	20399llu, 25717llu, 32401llu, 40823llu, 51437llu, 64811llu, 81649llu,
	102877llu, 129607llu, 163307llu, 205759llu, 259229llu, 326617llu,
	411527llu, 518509llu, 653267llu, 823117llu, 1037059llu, 1306601llu,
	1646237llu, 2074129llu, 2613229llu, 3292489llu, 4148279llu, 5226491llu,
	6584983llu, 8296553llu, 10453007llu, 13169977llu, 16593127llu, 20906033llu,
	26339969llu, 33186281llu, 41812097llu, 52679969llu, 66372617llu,
	83624237llu, 105359939llu, 132745199llu, 167248483llu, 210719881llu,
	265490441llu, 334496971llu, 421439783llu, 530980861llu, 668993977llu,
	842879579llu, 1061961721llu, 1337987929llu, 1685759167llu, 2123923447llu,
	2675975881llu, 3371518343llu, 4247846927llu, 5351951779llu, 6743036717llu,
	8495693897llu, 10703903591llu, 13486073473llu, 16991387857llu,
	21407807219llu, 26972146961llu, 33982775741llu, 42815614441llu,
	53944293929llu, 67965551447llu, 85631228929llu, 107888587883llu,
	135931102921llu, 171262457903llu, 215777175787llu, 271862205833llu,
	342524915839llu, 431554351609llu, 543724411781llu, 685049831731llu,
	863108703229llu, 1087448823553llu, 1370099663459llu, 1726217406467llu,
	2174897647073llu, 2740199326961llu, 3452434812973llu, 4349795294267llu,
	5480398654009llu, 6904869625999llu, 8699590588571llu, 10960797308051llu,
	13809739252051llu, 17399181177241llu, 21921594616111llu, 27619478504183llu,
	34798362354533llu, 43843189232363llu, 55238957008387llu, 69596724709081llu,
	87686378464759llu, 110477914016779llu, 139193449418173llu,
	175372756929481llu, 220955828033581llu, 278386898836457llu,
	350745513859007llu, 441911656067171llu, 556773797672909llu,
	701491027718027llu, 883823312134381llu, 1113547595345903llu,
	1402982055436147llu, 1767646624268779llu, 2227095190691797llu,
	2805964110872297llu, 3535293248537579llu, 4454190381383713llu,
	5611928221744609llu, 7070586497075177llu, 8908380762767489llu,
	11223856443489329llu, 14141172994150357llu, 17816761525534927llu,
	22447712886978529llu, 28282345988300791llu, 35633523051069991llu,
	44895425773957261llu, 56564691976601587llu, 71267046102139967llu,
	89790851547914507llu, 113129383953203213llu, 142534092204280003llu,
	179581703095829107llu, 226258767906406483llu, 285068184408560057llu,
	359163406191658253llu, 452517535812813007llu, 570136368817120201llu,
	718326812383316683llu, 905035071625626043llu, 1140272737634240411llu,
	1436653624766633509llu, 1810070143251252131llu, 2280545475268481167llu,
	2873307249533267101llu, 3620140286502504283llu, 4561090950536962147llu,
	5746614499066534157llu, 7240280573005008577llu, 9122181901073924329llu,
	11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu
};




int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

struct Pair {
	int key, value;
};




class GpuHashTable
{
	Pair *Container;
	int Size;
	int NumElements;

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
