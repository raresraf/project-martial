
/**
 * @file gpu_hashtable.cu
 * @brief Dynamic GPU Hash Table with linear probing and load-balanced resizing.
 * 
 * Algorithm: Open addressing with linear probing and multiplicative hashing.
 * Memory Model: Global memory allocation for the node array.
 * Synchronization: Uses atomicCAS for race-free key insertion and thread-safe updates.
 * Domain: HPC, Parallel Data Structures.
 */

#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#include "gpu_hashtable.hpp"


/**
 * @brief Multiplicative hash function for device-side index generation.
 * 
 * Time Complexity: O(1)
 */
__device__ int gpu_hash(int data, int maxSize) {
	return ((long) abs(data) * 823117) % 42815614441 % maxSize;
}


/**
 * @brief Constructor: Initializes the GPU hash table with a fixed capacity.
 */
GpuHashTable::GpuHashTable(int size) {

	// Memory Hierarchy: Global memory allocation for the hash bucket pool.
	cudaMalloc(&table, sizeof(Node) * size);
	if (table == NULL)
		return;
	
	// Pre-condition: Table must be zeroed to mark all slots as KEY_INVALID.
	cudaMemset(table, 0, size * sizeof(Node));

	maxSize = size;
	currentSize = 0;
}


/**
 * @brief Destructor: Releases device resources.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(table);
}


/**
 * @brief CUDA kernel for migrating entries during table expansion.
 * 
 * Functional Utility: Re-hashes existing valid entries into a larger buffer, 
 * resolving collisions via linear probing in the new address space.
 */
__global__ void gpu_hashtable_rehashing(Node *old_table, Node *new_table, int old_size, int new_size) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Pre-condition: Thread index must be within the old table bounds.
	if (idx >= old_size)
		return;

	// Optimization: Skip empty slots during migration.
	if (old_table[idx].key == KEY_INVALID)
		return;

	int old_key = old_table[idx].key;
	int value = old_table[idx].value;

	int position = gpu_hash(old_key, new_size);
	int res, step = 0;

	/**
	 * Block Logic: Insertion into the new table.
	 * Invariant: Moves through the new table until an empty slot is claimed via atomicCAS.
	 */
	while (step < new_size) {
		
		res = atomicCAS(&new_table[position].key, 0, old_key);

		if (res == 0 || res == old_key) {
			new_table[position].value = value;
			return;
		}

		// Linear probe wrap-around.
		++position;
		position %= new_size;

		++step;
	}
}


/**
 * @brief Rebuilds the hash table with a new capacity.
 * 
 * Logic: Synchronizes the device after migration to ensure data integrity 
 * before freeing the old buffer.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	Node *new_table;
	int numBlocks;
	
	cudaMalloc(&new_table, sizeof(Node) * numBucketsReshape);
	if (new_table == NULL)
		return;

	cudaMemset(new_table, 0, numBucketsReshape * sizeof(Node));

	// Block Logic: occupancy-optimized kernel configuration.
	numBlocks = (maxSize % 1024 != 0) ? (maxSize / 1024 + 1) : (maxSize / 1024);
	gpu_hashtable_rehashing<<<numBlocks, 1024>>>(table, new_table, maxSize, numBucketsReshape);

	cudaDeviceSynchronize();

	cudaFree(table);

	maxSize = numBucketsReshape;
	table = new_table;
}


/**
 * @brief CUDA kernel for parallel entry insertion.
 * 
 * Logic: Uses atomic compare-and-swap to claim slots, handling both new keys 
 * and updates to existing keys.
 */
__global__ void gpu_hashtable_insert(Node* table, int maxSize, int *keys, int *values, int numKeys) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numKeys)
		return;

	int key = keys[idx];
	int value = values[idx];

	int position = gpu_hash(key, maxSize);
	int res, step = 0;

	/**
	 * Block Logic: Linear probing search and claim.
	 * Invariant: Guaranteed to terminate if maxSize items are inspected.
	 */
	while (step < maxSize) {

		res = atomicCAS(&table[position].key, 0, key);

		if (res == 0 || res == key) {
			table[position].value = value;
			return;
		}

		++position;
		position %= maxSize;
		
		++step;
	}
}


/**
 * @brief Orchestrates batch insertion and handles adaptive resizing.
 * 
 * Optimization: Resizes when occupancy exceeds 60% to maintain O(1) probe performance.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys;
	int *device_values;
	int numBlocks;

	if (float(numKeys + currentSize) / maxSize >= 0.6) {
		reshape(int((currentSize + numKeys) / 0.8));
	}

	cudaMalloc(&device_keys, sizeof(int) * numKeys);
	cudaMalloc(&device_values, sizeof(int) * numKeys);
	if (device_values == NULL || device_keys == NULL)
		return false;

	cudaMemcpy(device_keys, keys, sizeof(int) * numKeys, cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, sizeof(int) * numKeys, cudaMemcpyHostToDevice);
	
	numBlocks = (numKeys % 1024 != 0) ? (numKeys / 1024 + 1) : (numKeys / 1024);
	gpu_hashtable_insert<<<numBlocks, 1024>>>(table, maxSize, device_keys, device_values, numKeys);

	cudaDeviceSynchronize();

	currentSize += numKeys;

	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}


/**
 * @brief CUDA kernel for parallel value lookup.
 */
__global__ void gpu_hashtable_get(Node* table, int maxSize, int *keys, int *values, int numKeys) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numKeys)
		return;

	int key = keys[idx];
	int position = gpu_hash(key, maxSize);
	int step = 0;

	/**
	 * Block Logic: Linear search traversal.
	 * Invariant: Returns VALUE_INVALID if the entire table is searched without a match.
	 */
	while (step < maxSize) {

		if (key == table[position].key) {
			values[idx] = table[position].value;
			return;
		}

		++position;
		position %= maxSize;

		++step;
	}

	values[idx] = VALUE_INVALID;
}


/**
 * @brief Batch retrieval interface using Managed Memory for host access.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_keys;
	int *values;
	int numBlocks;

	cudaMalloc(&device_keys, sizeof(int) * numKeys);
	// Memory Hierarchy: Managed Memory (Unified) simplifies result retrieval to host.
	cudaMallocManaged(&values, sizeof(int) * numKeys);
	if (values == NULL || device_keys == NULL)
		return NULL;

	cudaMemcpy(device_keys, keys, sizeof(int) * numKeys, cudaMemcpyHostToDevice);

	numBlocks = (numKeys % 1024 != 0) ? (numKeys / 1024 + 1) : (numKeys / 1024);
	gpu_hashtable_get<<<numBlocks, 1024>>>(table, maxSize, device_keys, values, numKeys);

	cudaDeviceSynchronize();

	cudaFree(device_keys);

	return values;
}


/**
 * @brief Returns current occupancy density.
 */
float GpuHashTable::loadFactor() {
	return (maxSize == 0) ? 0 : ((float)currentSize / maxSize);
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
#define	VALUE_INVALID	0

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
 * @brief Metadata for prime-based hashing distribution.
 */
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


/**
 * @brief Simple hash variations for dispersion tests.
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
 * @struct Node
 * @brief Representation of a key-value pair for GPU storage.
 */
typedef struct _Node {
	int key;
	int value;
} Node;


/**
 * @class GpuHashTable
 * @brief Controller for managing the life-cycle of a GPU-resident hash table.
 */
class GpuHashTable
{
	public:
		int maxSize;
		int currentSize;
		Node *table;

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
