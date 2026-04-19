/**
 * @cafb0f03-7101-46ad-9f97-174d4bcf855a/gpu_hashtable.cu
 * @brief CUDA-accelerated Hash Table implementation using dual-array storage and interleaved linear probing.
 * Domain: HPC Parallel Data Structures, GPU Systems Optimization.
 * Architecture: Employs two parallel bucket arrays (nodes1, nodes2) to reduce collision clustering and improve memory throughput.
 * Hashing Strategy: Uses a multiplicative congruential hash function (hash1) with prime multipliers for uniform bit distribution.
 * Synchronization: Implements a lock-free update model using atomic Compare-And-Swap (atomicCAS) for thread-safe slot reservation across both memory banks.
 * Execution Model: Orchestrates parallel batch processing and dynamic capacity management with Host-side monitoring of load factors.
 */

#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <stdio.h>
#include <string>

#include "gpu_hashtable.hpp"

#define BLOCKSIZE 256


/**
 * @brief GpuHashTable Constructor.
 * Strategy: Allocates global memory for two separate bucket arrays on the device and initializes state metadata.
 */
GpuHashTable::GpuHashTable(int size) {
	cudaMalloc((void**)&my_ht.nodes1, size * sizeof(Node));
	cudaMalloc((void**)&my_ht.nodes2, size * sizeof(Node));
	my_ht.items = 0;
    // Initialization: Zeroes out both table regions.
	cudaMemset(my_ht.nodes1, 0, size * sizeof(Node));
	cudaMemset(my_ht.nodes2, 0, size * sizeof(Node));
	my_ht.size = size;
}


/**
 * @brief Cleanup of device resources.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(my_ht.nodes1);
	cudaFree(my_ht.nodes2);
	my_ht.items = 0;
	my_ht.size = 0;
}


/**
 * @brief CUDA Kernel for rehashing entries into a new table during expansion.
 * @param input Source bucket array.
 * @param size Capacity of the source table.
 * @param output Destination bucket array (larger).
 * Logic: Parallel migration of valid entries using a circular linear probing search in the new table.
 */
__global__ void kernel_reshape_HashTableElems(Node *input, int size, Node *output)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= size)
		return;

	int rKey = input[i].key;
	int rValue = input[i].value;
    
    // Condition: Only migrate entries with non-zero keys and values.
	if (rKey <= 0 || rValue <= 0)
		return;
	int where = hash1(rKey, size);

	// Block Logic: Linear probing search for an empty slot in the destination table.
	while (1) {
        // Synchronization: atomicCAS ensures thread-safe reservation of empty (0) slots.
		if(atomicCAS(&(output[where].key), 0, rKey) == 0)
			break;
		where = (where + 1) % size;
	}
    // Invariant: Once slot is acquired, transfer the associated value.
	output[where].value = rValue;
}


/**
 * @brief Resizes the hash table and rehashes entries across both internal banks.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	Node *result1, *result2;	
	cudaMalloc((void**)&result1, numBucketsReshape * sizeof(Node));
	if (!result1)
		return;

	cudaMalloc((void**)&result2, numBucketsReshape * sizeof(Node));
	if (!result2)
		return;

	unsigned int block_no = my_ht.size / BLOCKSIZE;
	if (my_ht.size % BLOCKSIZE)
		block_no++;
	my_ht.size = numBucketsReshape;
	
	
    // Execution: Preserves the original broken '>>' kernel launch syntax as required by the Zero Mutation policy.
	kernel_resize_HashTableElems>>(
		my_ht.nodes1, my_ht.size, result1);
	cudaDeviceSynchronize();
	
	kernel_resize_HashTableElems>>(
		my_ht.nodes2, my_ht.size, result2);
	cudaDeviceSynchronize();
	
	cudaFree(my_ht.nodes1);
	cudaFree(my_ht.nodes2);
	my_ht.nodes1 = result1;
	my_ht.nodes2 = result2;
}


/**
 * @brief CUDA Kernel for parallel batch insertion into a dual-array table.
 * Strategy: Sequentially attempts to reserve slots in nodes1, then nodes2, before resorting to linear probing.
 * Optimization: Reduces bank contention by interleaving access across two memory regions.
 */
__global__ void kernel_insert_HastTableElems(HashTable h, int *keys, int *values, int numKeys)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= numKeys)
		return;
	if(keys[i] <= 0 || values[i] <= 0)
		return;
	int where = hash1(keys[i], h.size);
	int key = keys[i]; 

	// Block Logic: Multi-bank insertion loop.
	while (1){
		// Synchronization Phase 1: Try Bank 1 (nodes1).
		if (atomicCAS(&(h.nodes1[where].key), 0, key) == 0){
			h.nodes1[where].value =  values[i];
			return;
		}
		if (atomicCAS(&(h.nodes1[where].key), key, key) == key){
			h.nodes1[where].value =  values[i];
			return;
		}

		// Synchronization Phase 2: Try Bank 2 (nodes2).
		if (atomicCAS(&(h.nodes2[where].key), 0, key) == 0){


			h.nodes2[where].value =  values[i];
			return;
		}
		if (atomicCAS(&(h.nodes2[where].key), key, key) == key){
			h.nodes2[where].value =  values[i];
			return;
		}	
		// Logic: Step to next bucket index.
		where = (where + 1) % (h.size);
	}
}


/**
 * @brief Batch insertion interface from Host.
 * Optimization: Performs dynamic load factor monitoring and triggers expansion if density > 90%.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *gpuKeys, *gpuValues;
	cudaMalloc((void**)&gpuKeys, numKeys * sizeof(int));
	cudaMalloc((void**)&gpuValues, numKeys * sizeof(int));
	cudaMemcpy(gpuKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	my_ht.items += numKeys;
	// Decision Logic: Proactive capacity scaling to maintain O(1) performance invariants.
	if ( ((float)((numKeys + my_ht.items) / my_ht.size)) >= 0.9f)
		reshape((int)(my_ht.size / 0.8f));
	
	int block_no = numKeys / BLOCKSIZE;
	if (numKeys % BLOCKSIZE)
		block_no++;
	
	kernel_insert_HastTableElems>>(
		my_ht, gpuKeys, gpuValues, numKeys);
	cudaDeviceSynchronize();
	
	cudaFree(gpuKeys);
	cudaFree(gpuValues);
	return false;
}


/**
 * @brief CUDA Kernel for parallel batch retrieval.
 * Logic: Parallel search sequence mirroring the dual-bank insertion strategy.
 */
__global__ void kernel_get_HashTableElems(HashTable h, int *keys, int *values, int numKeys)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= numKeys)
		return;

	int where = hash1(keys[i], h.size);
	int key = keys[i];
	if (key == 0)
		return;
	
    // Block Logic: Multi-bank search loop.
	while (1) {


		if (atomicCAS(&(h.nodes1[where].key), key, key) == key){
			values[i] = h.nodes1[where].value;
			break;
		}


		if (atomicCAS(&(h.nodes2[where].key), key, key) == key){
			values[i] = h.nodes2[where].value;
			break;
		}
		where = (where + 1) % (h.size);	
	}
}


/**
 * @brief Batch retrieval interface from Host.
 * @return Pointer to host-allocated results array.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *result;
	int *gpuKeys, *gpuValues;
	result = (int*)calloc(numKeys, sizeof(int));
	if (!result)
		return NULL;
	cudaMalloc((void**)&gpuValues, numKeys * sizeof(int));
	if (!gpuValues)
		return NULL;
	cudaMalloc((void**)&gpuKeys, numKeys * sizeof(int));
	if (!gpuKeys)
		return NULL;
	cudaMemcpy(gpuKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int block_no = numKeys / BLOCKSIZE;
	if (numKeys % BLOCKSIZE)
		block_no++;
	


	kernel_get_HashTableElems>>(
		my_ht, gpuKeys, gpuValues, numKeys);
	cudaDeviceSynchronize();

    // Memory Flow: Synchronous retrieval of query results to Host.
	cudaMemcpy(result, gpuValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(gpuKeys);
	cudaFree(gpuValues);

	return result;
}


/**
 * @brief Returns the current occupancy ratio.
 */
float GpuHashTable::loadFactor() {
		return 1.0f * (float) my_ht.items / my_ht.size; 
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




__device__ int hash1(int data, int limit) {
	return ((long)abs(data) * 3llu) % 9122181901073924329llu % limit;
}
__device__ int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
__device__ int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

/**
 * @brief Represents a single key-value entry on the GPU.
 */
typedef struct {
	int key;
	int value;
} Node;

/**
 * @brief Internal metadata and memory layout for the dual-bank GPU hash table.
 */
typedef struct {
	Node *nodes1;
	Node *nodes2;
	int size;
	int items;
} HashTable;




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

		HashTable my_ht;
		~GpuHashTable();
};

#endif
