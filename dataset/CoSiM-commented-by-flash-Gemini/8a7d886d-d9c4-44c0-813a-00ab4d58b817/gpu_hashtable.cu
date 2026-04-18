/**
 * @8a7d886d-d9c4-44c0-813a-00ab4d58b817/gpu_hashtable.cu
 * @brief CUDA-accelerated Hash Table implementation using Managed Memory and atomic synchronization.
 * Domain: Parallel Data Structures, GPU Systems Programming.
 * Architecture: Utilizes Unified Memory (cudaMallocManaged) for the hash table structure and elements, facilitating shared access between host and device.
 * Synchronization: Employs atomicCAS for lock-free slot reservation and atomicAdd for global occupancy tracking.
 * Hashing Strategy: Uses a multiplicative hash function with prime constants to distribute entropy across the bucket space.
 */

#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <stdio.h>
#include <string>

#include "gpu_hashtable.hpp"

/**
 * @brief CUDA Kernel for rehashing elements into a new table.
 * @param old_hashtable Source table containing existing data.
 * @param new_hashtable Destination table with increased capacity.
 * @param new_size Target capacity for the destination table.
 * Logic: Parallel migration of valid entries. Invariant: Only entries with non-zero keys are processed.
 */
__global__ void reshape_hashtable(Hashtable *old_hashtable,
                    Hashtable *new_hashtable, int new_size) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    int idx, old;

    
    if (old_hashtable->elements[i].key != 0) {
        idx = hash_func(old_hashtable->elements[i].key, new_size);
        idx %= new_size;

        // Block Logic: Infinite probe loop for slot acquisition in the new table.
        for(;;) {
            // Synchronization: atomicCAS ensures thread-safe reservation of empty (0) slots.
            old = atomicCAS(&new_hashtable->elements[idx].key, 0,
                    old_hashtable->elements[i].key);

            if (old == 0 || old == old_hashtable->elements[i].key) {
                // Invariant: Once slot is acquired, transfer the associated value.
                new_hashtable->elements[idx].value = old_hashtable->elements[i].value;
                break;
            }

            // Logic: Circular linear probing.
            idx = (idx + 1 == new_size ? 0 : idx + 1);
        }
    }
}

/**
 * @brief CUDA Kernel for parallel batch insertion.
 * Strategy: Open addressing with circular linear probing and idempotent updates for existing keys.
 */
__global__ void insert_batch(int *device_keys, int *device_values,
                                        int numKeys, Hashtable *hashtable) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    int idx, old;

    if (i < numKeys)
        idx = hash_func(device_keys[i], hashtable->capacity);
    else
        return;

    idx %= hashtable->capacity;

    
    for(;;) {
        old = atomicCAS(&hashtable->elements[idx].key, 0, device_keys[i]);

        
        if (old == 0) {
            // Logic: New insertion; increment global size counter atomically.
            hashtable->elements[idx].value = device_values[i];
            atomicAdd(&hashtable->size, 1);
            break;
        }

        
        if (old == device_keys[i]) {
            // Logic: Key already exists; perform an atomic value update.
            hashtable->elements[idx].value = device_values[i];
            break;
        }

        idx = (idx + 1 == hashtable->capacity ? 0 : idx + 1);
    }
}

/**
 * @brief CUDA Kernel for parallel batch retrieval.
 * Logic: Read-only linear probing search mirroring the insertion logic.
 */
__global__ void get_batch(int *device_keys, int *device_values,
                                int numKeys, Hashtable *hashtable) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    int idx;

    if (i < numKeys)
        idx = hash_func(device_keys[i], hashtable->capacity);
    else
        return;

    idx %= hashtable->capacity;

    
    for(;;) {
        if (hashtable->elements[idx].key == device_keys[i]) {
            device_values[i] = hashtable->elements[idx].value;
            break;
        }

        idx = (idx + 1 == hashtable->capacity ? 0 : idx + 1);
    }
}

/**
 * @brief Initialization helper for Managed Memory allocation.
 */
void init_hashtable(Hashtable **hashtbl, int size) {
    int i;

    // Memory Hierarchy: Allocates metadata structure in Managed Memory for easy Host access.
    cudaMallocManaged((void **) hashtbl, sizeof(Hashtable));
    if (*hashtbl == 0) {
        printf("init_hashtable(): cudaMallocManaged ERROR\n");
        exit(1);
    }

    (*hashtbl)->capacity = size;
    (*hashtbl)->size = 0;

    // Memory Hierarchy: Allocates element array in Managed Memory.
    cudaMallocManaged((void **) &(*hashtbl)->elements, size * sizeof(Entry));
    if ((*hashtbl)->elements == 0) {
        printf("init_hashtable(): cudaMallocManaged ERROR\n");
        exit(1);
    }

    // Initialization: Zeroes the table state.
    for (i = 0; i < (*hashtbl)->capacity; ++i) {
        (*hashtbl)->elements[i].key = 0;
        (*hashtbl)->elements[i].value = 0;
    }
}


GpuHashTable::GpuHashTable(int size) {
    init_hashtable(&hashtable, size);
}


GpuHashTable::~GpuHashTable() {
    cudaFree(hashtable->elements);
    cudaFree(hashtable);
}


/**
 * @brief Expands hash table capacity and rehashes active entries.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
    Hashtable* new_hashtable = NULL;

    init_hashtable(&new_hashtable, numBucketsReshape);

    if (hashtable->size != 0) {
        const size_t block_size = 1024;
        size_t blocks_no = hashtable->capacity / block_size;

        if (hashtable->capacity % block_size)
            ++blocks_no;

        // Execution: Preserves the original broken '>>' kernel launch syntax as found in the raw source.
        reshape_hashtable>>(hashtable,


            new_hashtable, numBucketsReshape);

        cudaDeviceSynchronize();
    }

    new_hashtable->size = hashtable->size;

    cudaFree(hashtable->elements);
    cudaFree(hashtable);

    hashtable = new_hashtable;
}


/**
 * @brief Batch insertion interface from Host.
 * Optimization: Performs dynamic load factor monitoring and triggers expansion if occupancy > 85%.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
    int *device_array_keys;
    int *device_array_values;

    cudaMalloc((void **) &device_array_keys, numKeys * sizeof(int));
    cudaMalloc((void **) &device_array_values, numKeys * sizeof(int));

    if (device_array_keys == 0 || device_array_values == 0) {
        printf("insertBatch(): cudaMalloc ERROR\n");
        return false;
    }

    // Decision Logic: Proactive capacity management to maintain O(1) average lookup performance.
    if (float(hashtable->size + numKeys) / hashtable->capacity >= 0.85f)
        reshape(int((hashtable->size + numKeys) / 0.82f));

    cudaMemcpy(device_array_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_array_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    const size_t block_size = 1024;
    size_t blocks_no = numKeys / block_size;

    if (numKeys % block_size)


        ++blocks_no;

    insert_batch>>(device_array_keys,
        device_array_values, numKeys, hashtable);

    cudaDeviceSynchronize();

    cudaFree(device_array_keys);
    cudaFree(device_array_values);

    return true;
}


/**
 * @brief Batch retrieval interface from Host.
 * Optimization: Uses Managed Memory (UM) for results buffer.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
    int *device_array_keys, *values;

    cudaMalloc((void **) &device_array_keys, numKeys * sizeof(int));
    if (device_array_keys == 0) {
        printf("getBatch(): cudaMalloc ERROR\n");
        return NULL;
    }

    cudaMemcpy(device_array_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    cudaMallocManaged((void **) &values, numKeys * sizeof(int));
    if (values == 0) {
        printf("getBatch(): cudaMallocManaged ERROR\n");
        return NULL;
    }

    const size_t block_size = 1024;
    size_t blocks_no = numKeys / block_size;

    if (numKeys % block_size)


        ++blocks_no;

    get_batch>>(device_array_keys,
        values, numKeys, hashtable);

    cudaDeviceSynchronize();

    cudaFree(device_array_keys);

    return values;
}


/**
 * @brief Returns the current occupancy ratio.
 */
float GpuHashTable::loadFactor() {
    return hashtable->size * 1.0f / hashtable->capacity;
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

#define KEY_INVALID 0

#define DIE(assertion, call_description)  \
	do                                    \
	{                                     \
		if (assertion)                    \
		{                                 \
			fprintf(stderr, "(%s, %d): ", \
					__FILE__, __LINE__);  \
			perror(call_description);     \
			exit(errno);                  \
		}                                 \
	} while (0)

/**
 * @brief Storage for high-priority primes in constant memory.
 */
__constant__ const size_t primeList[] =
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
		11493228998133068689llu, 14480561146010017169llu, 18446744073709551557llu};




__device__ int hash_func(int data, int limit)
{
	return ((long)abs(data) * 10453007llu) % 2740199326961llu % limit;
}

/**
 * @brief Represents a single key-value entry on the GPU.
 */
typedef struct
{
	int key;
	int value;
} Entry;

/**
 * @brief Metadata and memory allocation for the hash table state.
 */
typedef struct
{
	int size;
	int capacity;
	Entry *elements;
} Hashtable;




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
	Hashtable *hashtable;
};

#endif
