
#include "gpu_hashtable.hpp"
#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * @file gpu_hashtable-nu_merge.cu
 * @brief Thread-safe parallel hash table using CUDA Unified Memory and Linear Probing.
 * 
 * Functional Intent: Provides a high-concurrency hash table optimized for parallel 
 * execution on NVIDIA GPUs. It leverages CUDA Managed Memory (Unified Memory) to 
 * simplify host-device data sharing and employs `atomicCAS` for thread-safe 
 * bucket reservation. The implementation features a dynamic `reshape` mechanism 
 * to scale capacity while maintaining O(1) average lookup performance.
 * 
 * Domain: HPC, Parallel Data Structures, CUDA Unified Memory.
 */

/**
 * hash_function - Computes a deterministic bucket index using large prime coefficients.
 */
__host__ __device__ int hash_function(int data, int limit) {
	return ((long)abs(data) * 718326812383316683llu) % 8699590588571llu % limit;
}

/**
 * GpuHashTable constructor - Initializes unified memory buffers.
 */
GpuHashTable::GpuHashTable(int size) {
	this->HTcontor = 0;
	this->HTmarime = size;

	// Optimization: Uses Managed Memory to allow transparent access from both CPU and GPU.
	cudaMallocManaged((void **) &this->device_valori, this->HTmarime * sizeof(int));
	cudaMallocManaged((void **) &this->device_chei, this->HTmarime * sizeof(int));
	// Invariant: Keys are initialized to KEY_INVALID (0) to mark buckets as empty.
	cudaMemset(this->device_chei, KEY_INVALID, this->HTmarime * sizeof(int));
}

GpuHashTable::~GpuHashTable() {
	cudaFree(this->device_valori);
	cudaFree(this->device_chei);
}

/**
 * reshape - Increases the hash table capacity to maintain search density.
 * 
 * Algorithm: Full Table Migration.
 * 1. Backs up active entries into host-side temporary buffers.
 * 2. Re-allocates device memory with 1.06x factor for the new size.
 * 3. Re-inserts backed-up entries into the newly initialized table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int **ht_aux = (int **)malloc(sizeof(int *) * 2);
	ht_aux[0] = (int *)malloc(this->HTcontor * sizeof(int));
	ht_aux[1] = (int *)malloc(this->HTcontor * sizeof(int));

	int idx = 0, numKeys = this->HTcontor;

	// Block Logic: Active entry extraction loop.
	int i = 0;
	while (i < HTmarime){
		if (this->device_chei[i] != KEY_INVALID) {
			ht_aux[0][idx] = this->device_chei[i];
			ht_aux[1][idx] = this->device_valori[i];
			idx++;
		}
		i++;
	}

	this->HTcontor = 0;
	this->HTmarime = numBucketsReshape * 1.06f;

	cudaFree(this->device_chei);
	cudaFree(this->device_valori);

	cudaMallocManaged((void **) &this->device_valori, this->HTmarime * sizeof(int));
	cudaMallocManaged((void **) &this->device_chei, this->HTmarime * sizeof(int));
	cudaMemset(this->device_chei, KEY_INVALID, this->HTmarime * sizeof(int));

	// Functional Utility: Populates the new structure using parallel insertion logic.
	insertBatch(ht_aux[0], ht_aux[1], numKeys);

	for (int j = 0; j < 2; j++)
		free(ht_aux[j]);
	free(ht_aux);
}

/**
 * @kernel kernel_insert
 * @brief Performs atomic insertion and collision resolution using linear probing.
 * 
 * Algorithm: Open Addressing.
 * 1. Computes initial hash index.
 * 2. Attempts to claim slot using `atomicCAS`.
 * 3. If slot is occupied by a different key, increments index (linear probe) 
 *    until an empty or matching slot is located.
 * 4. Updates value using `atomicExch`.
 */
__global__ void kernel_insert(int *keys, int *values, int numKeys, int *htchei, int *htvalori, int HTmarime, int *HTcontor) {
	int h, h_aux, aux, check = 0;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < numKeys) {
		int add_cheie = keys[idx], valueToAdd = values[idx];
		h = hash_function(add_cheie, HTmarime);
		
		// Synchronization: atomicCAS ensures monotonic bucket ownership transition.
		aux = atomicCAS(&htchei[h], KEY_INVALID, add_cheie);

		if (aux == KEY_INVALID || aux == add_cheie)
			check = 1;

		if (check == 1) {
			if (aux == KEY_INVALID) atomicAdd(HTcontor, 1);
			atomicExch(&htvalori[h], valueToAdd);
			return;
		}
		
		h_aux = h;
		h = (h + 1) % HTmarime;
		
		/**
		 * Block Logic: Linear Probing Spin.
		 * Invariant: Probes every available bucket in a circular fashion until 
		 * the key is successfully placed.
		 */
		while (h != h_aux) {
			aux = atomicCAS(&htchei[h], KEY_INVALID, add_cheie);
			if (aux == KEY_INVALID || aux == add_cheie) {
				if (aux == KEY_INVALID) atomicAdd(HTcontor, 1);
				atomicExch(&htvalori[h], valueToAdd);
				return;
			}
			h = (h + 1) % HTmarime;
		}
	}
}

/**
 * insertBatch - Public interface for parallel data ingestion from host.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *device_chei = NULL;
	int *device_valori = NULL;
	int *device_HTcontor = NULL;

	if (numKeys == 0) return true;

	// Pre-condition: Automatic resize if incoming batch exceeds table headroom.
	if (this->HTcontor + numKeys > this->HTmarime)
		reshape(this->HTmarime + numKeys);

	cudaMalloc((void **) &device_chei, numKeys * sizeof(int));
	cudaMalloc((void **) &device_valori, numKeys * sizeof(int));
	cudaMallocManaged((void **) &device_HTcontor, sizeof(int));

	cudaMemcpy(device_chei, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_valori, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_HTcontor, &this->HTcontor, sizeof(int), cudaMemcpyHostToDevice);
	
	const int block_size = 1024;
	int blocks_no = (numKeys + block_size - 1) / block_size;

	kernel_insert<<<blocks_no, block_size>>>(device_chei,device_valori,
		numKeys,this->device_chei,this->device_valori,this->HTmarime,device_HTcontor);
	
	cudaDeviceSynchronize();
	cudaMemcpy(&this->HTcontor, device_HTcontor, sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(device_chei);
	cudaFree(device_valori);
	cudaFree(device_HTcontor);
	return true;
}

/**
 * @kernel kernel_get
 * @brief Retrieval kernel searching buckets via linear probing.
 */
__global__ void kernel_get(int *keys, int *values, int numKeys, int *hashTableKeys, int *hashTableValues, int HTmarime) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int new_key, h, h_aux;
	if (idx < numKeys) {
		new_key = keys[idx];
		h = hash_function(new_key, HTmarime);
		h_aux = h;

		if (hashTableKeys[h] == new_key) {
			values[idx] = hashTableValues[h];
			return;
		}
		h = (h + 1) % HTmarime;

		while (h != h_aux) {
			if (hashTableKeys[h] == new_key) {
				values[idx] = hashTableValues[h];
				return;
			}
			if (hashTableKeys[h] == KEY_INVALID) return; // Invariant: linear probe chain stops at empty slot.
			h = (h + 1) % HTmarime;
		}
	}
}

int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_chei = NULL;
	int *device_valori = NULL;

	cudaMalloc((void **) &device_chei, numKeys * sizeof(int));
	cudaMalloc((void **) &device_valori, numKeys * sizeof(int));
	cudaMemcpy(device_chei, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int *valori = (int *)calloc(numKeys, sizeof(int));

	const int block_size = 1024;
	int blocks_no = (numKeys + block_size - 1) / block_size;

	kernel_get<<<blocks_no, block_size>>>(device_chei,device_valori,numKeys,
		this->device_chei,this->device_valori,this->HTmarime);
	
	cudaDeviceSynchronize();
	cudaMemcpy(valori, device_valori, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(device_chei);
	cudaFree(device_valori);

	return valori;
}

float GpuHashTable::loadFactor() {
	return HTcontor / (float)HTmarime;
}

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <ctime>
#include <random>

using namespace std;

#define KEY_INVALID 0

#define DIE(assertion, call_description)              \
	do                                            \
	{                                             \
		if (assertion)                        \
		{                                     \
			fprintf(stderr, "(%s, %d): ", \
				__FILE__, __LINE__);  \
			perror(call_description);     \
			exit(errno);                  \
		}                                     \
	} while (0)

const size_t primeList[] = {
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
	11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu};

__host__ __device__ int hash_function_alt(int data, int limit) {
	return ((long)abs(data) * 2675975881llu) % 431554351609llu % limit;
}

class GpuHashTable
{
public:
	GpuHashTable(int size);
	void reshape(int sizeReshape);

	bool insertBatch(int *keys, int *values, int numKeys);
	int *getBatch(int *key, int numItems);

	float loadFactor();
	void occupancy();
	void print(string info);

	~GpuHashTable();

	int HTcontor, HTmarime;
	int *device_chei, *device_valori;
};

#endif
