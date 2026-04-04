/**
 * @file gpu_hashtable.cu
 * @brief A self-contained implementation of a GPU-accelerated hash table.
 * @details This file provides a complete hash table using CUDA. It uses an
 * open-addressing, linear-probing collision strategy. A unique feature of this
 * implementation is its "Structure of Arrays" (SoA) memory layout, where keys
 * and values are stored in separate, parallel arrays rather than an array of
 * key-value structs.
 *
 * @warning The core `put` kernel contains a significant race condition in its
 * `atomicCAS` logic, which may lead to incorrect behavior under contention. The
 * `getKeys` kernel is also inefficient as it may not terminate early when an

 * empty slot is found.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_hashtable.hpp"


/**
 * @brief CUDA kernel to insert or update a batch of key-value pairs.
 * @param ht_keys Device pointer to the hash table's key array.
 * @param ht_values Device pointer to the hash table's value array.
 * @param capacity The capacity of the hash table.
 * @param keys Device pointer to the input keys for the batch.
 * @param values Device pointer to the input values for the batch.
 * @param n The number of pairs in the batch.
 * @warning This kernel has a severe race condition. It writes the value in a
 * subsequent loop iteration after the `atomicCAS`, based on a stale index.
 * This can lead to data corruption or incorrect value assignments.
 */
__global__ void put(int *ht_keys, int *ht_values, int capacity, int *keys, int *values, int n){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Bug: This should be idx < n. As written, it may lead to out-of-bounds access.
        if (idx < n) {
                if (keys[idx] <= 0 || values[idx] <= 0){
                        return;
                }
                int idx_in_hashtable = ((long)abs(keys[idx]) * 6421llu) % 110477914016779llu % capacity;
                int old = -1;
                int last_idx_in_hashtable;
                // Block Logic: Linear probing loop.
                while (old != 0){
                        // Race Condition: This check and subsequent write happen in the *next*
                        // iteration, using a stale `last_idx_in_hashtable`.
                        if (old != -1 && ht_keys[last_idx_in_hashtable] == keys[idx]) {
                                ht_values[last_idx_in_hashtable] = values[idx];
                                return;
                        }
                        
                        last_idx_in_hashtable = idx_in_hashtable;
                        // Atomic Operation: Attempt to claim the key slot.
                        old = atomicCAS(&ht_keys[idx_in_hashtable], 0, keys[idx]);
                        
                        idx_in_hashtable++;
                        idx_in_hashtable %= capacity;
                }
                
                // Race Condition: The value is written here, but another thread could have
                // modified the slot at `last_idx_in_hashtable` in the meantime.
                ht_values[last_idx_in_hashtable] = values[idx];
        }
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @warning The search logic is inefficient. It does not terminate when an empty
 * slot is found, which should indicate that the key is not present. Instead, it
 * only terminates upon finding the key or probing the entire table.
 */
__global__ void getKeys(int *ht_keys, int *ht_values, int capacity, int *keys, int *values, int n){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx < n) {
                if (keys[idx] <= 0){
                        return;
		}
                int idx_in_hashtable = ((long)abs(keys[idx]) * 6421llu) % 110477914016779llu % capacity;
		int fst_tried_idx = idx_in_hashtable;
                // Block Logic: Linear probing loop to find the key.
                while (ht_keys[idx_in_hashtable] != keys[idx]){
                        idx_in_hashtable++;
                        idx_in_hashtable %= capacity;
			
			// Pre-condition: If we've probed the entire table, the key is not present.
			if (fst_tried_idx == idx_in_hashtable) {
				return;
			}
                }
                values[idx] = ht_values[idx_in_hashtable];
        }
        
}


/**
 * @brief Constructs a GpuHashTable object.
 */
GpuHashTable::GpuHashTable(int size) {
        this->load = 0;
        this->capacity = size;
        
        // Memory Hierarchy (SoA): Allocate separate device arrays for keys and values.
        cudaMalloc(&this->keys, size * sizeof(int));
        cudaMalloc(&this->values, size * sizeof(int));
        DIE(this->keys == 0, "cudaMalloc keys Init()");
	DIE(this->values == 0, "cudaMalloc values Init()");
        
        cudaMemset(this->keys, 0, size * sizeof(int));
        cudaMemset(this->values, 0, size * sizeof(int));
}

/**
 * @brief Destroys the GpuHashTable object.
 */
GpuHashTable::~GpuHashTable() {
        cudaFree(this->keys);
        this->keys = NULL;
        cudaFree(this->values);
        this->values = NULL;
        this->load = 0;
        this->capacity = 0;
}

/**
 * @brief Resizes the hash table and rehashes all existing elements.
 * @details This implementation cleverly reuses the `put` kernel to perform a
 * device-side rehash, avoiding expensive host-device data transfers.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
        
        int noBlocks = (this->capacity + block_size - 1) / block_size;
        
        // Allocate new, larger arrays for keys and values.
        int *to_replace_keys;
        int *to_replace_values;
        cudaMalloc(&to_replace_keys, numBucketsReshape * sizeof(int));
        cudaMalloc(&to_replace_values, numBucketsReshape * sizeof(int));
        DIE(to_replace_keys == 0, "cudaMalloc ReshapeHash()");
        DIE(to_replace_values == 0, "cudaMalloc ReshapeHash()");
        cudaMemset(to_replace_keys, 0, numBucketsReshape * sizeof(int));
        cudaMemset(to_replace_values, 0, numBucketsReshape * sizeof(int));
        
        // Functional Utility: Re-insert all elements from the old arrays into the new ones
        // by calling the existing `put` kernel.
        put<<<noBlocks, block_size>>>( to_replace_keys, to_replace_values, numBucketsReshape, this->keys, this->values, this->capacity);
        cudaDeviceSynchronize();
        
        // Free old storage and swap to the new arrays.
        cudaFree(this->keys);
        cudaFree(this->values);
	this->keys = to_replace_keys;
        this->values = to_replace_values;
	this->capacity = numBucketsReshape;
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
        int i;
	int ctr = 0;
        
	// Count valid pairs on the host first.
	for (i = 0; i < numKeys; i++){
		if (keys[i] > 0 && values[i] > 0)
			ctr++;
	}
        if (ctr == 0)
                return false;
        this->load += ctr;

        int noBlocks = (numKeys + block_size - 1) / block_size;
        
        // Pre-condition: Resize if the load factor exceeds the maximum threshold.
        if (loadFactor() > maxLoadFactor || this->load > this->capacity) {
                 reshape((int)(this->load / minLoadFactor) + 1);
        }
        
        // Allocate temporary device buffers for the input batch.
        int *kernel_keys;
        int *kernel_values;
        cudaMalloc(&kernel_keys, numKeys * sizeof(int));
        cudaMalloc(&kernel_values, numKeys * sizeof(int));
        DIE(kernel_keys == 0 || kernel_values == 0, "cudaMalloc kernel space");
        cudaMemcpy(kernel_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
        
        // Launch the insertion kernel.
        put<<<noBlocks, block_size>>>(this->keys, this->values, this->capacity, kernel_keys, kernel_values, numKeys);
	cudaDeviceSynchronize();
        
        cudaFree(kernel_keys);
	cudaFree(kernel_values);

	return true;
}

/**
 * @brief Retrieves values for a batch of keys.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
        
        int noBlocks = (numKeys + block_size - 1) / block_size;
        
        int *kernel_keys;
        int *kernel_values;
        cudaMalloc(&kernel_keys, numKeys * sizeof(int));
        cudaMalloc(&kernel_values, numKeys * sizeof(int));
        DIE(kernel_keys == 0 || kernel_values == 0, "cudaMalloc kernel space");
        cudaMemcpy(kernel_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
        
        // Launch the retrieval kernel.
        getKeys<<<noBlocks, block_size>>>(this->keys, this->values, this->capacity, kernel_keys, kernel_values, numKeys);
	cudaDeviceSynchronize();
        
        int *values = (int*)malloc(numKeys * sizeof(int));
        DIE(values == 0, "malloc values getBatch()");

        // Data Transfer: Copy results from device back to host.
        cudaMemcpy(values, kernel_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	
        cudaFree(kernel_keys);
	cudaFree(kernel_values);

	return values;
}

/**
 * @brief Calculates the current load factor.
 */
float GpuHashTable::loadFactor() {
	return (float)this->load / this->capacity;
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

#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)

const size_t block_size = 256;
const float minLoadFactor = 0.85;
const float maxLoadFactor = 0.95;
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
                
        private:
                int load;
                int capacity;
                int *keys;
                int *values;
};

#endif
