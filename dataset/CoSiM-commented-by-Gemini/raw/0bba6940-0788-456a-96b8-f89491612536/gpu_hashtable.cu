/**
 * @file gpu_hashtable.cu
 * @brief A self-contained implementation of a GPU-accelerated hash table.
 * @details This file provides both the host-side management class (`GpuHashTable`)
 * and the device-side CUDA kernels for a complete hash table solution. The
 * implementation uses an open-addressing, linear-probing collision resolution
 * strategy, optimized for parallel batch operations on the GPU.
 *
 * @note The header `gpu_hashtable.hpp` is expected to contain the class
 * and struct definitions.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_hashtable.hpp"

/**
 * @brief Computes a hash value for a given integer key on the device.
 * @param data The integer key to hash.
 * @param limit The capacity of the hash table, used for the modulo operation.
 * @return An integer hash value within the range [0, limit-1].
 * @details This is a simple multiplicative hash function designed for speed on the GPU.
 */
__device__ int computeHash(int data, int limit)
{
    long hash = 51437; // A prime number seed.
    hash += data * (data << 17) * hash;

    return hash % limit;
}

/**
 * @brief CUDA kernel to rehash all entries from an old hash table to a new one.
 * @param old_hashmap The source hash table to read from.
 * @param new_hashmap The destination hash table to write to.
 * @details Each thread is responsible for migrating one entry. It reads a key-value
 * pair from the old table, re-computes its hash for the new table's size, and
 * inserts it into the new table using linear probing and atomicCAS to find a slot.
 */
__global__ void map_resize(hash_table old_hashmap, hash_table new_hashmap) {
    // Thread Indexing: Each thread targets one entry from the old map.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Pre-condition: Ensure the thread index is within the bounds of the old map's size.
    if (idx >= old_hashmap.size) {
        return;
    }

    int oldKey;
    int newKey;
    int newHash;

    newKey = old_hashmap.entries[idx].key;
    // Hashing: Re-compute the hash for the new table size.
    newHash = computeHash(newKey, new_hashmap.size);


    // Block Logic: Perform linear probing to find an empty slot in the new map.
    // Invariant: Probing continues until an empty slot is atomically secured.
    for (int j = newHash; j < new_hashmap.size; j++) {
        // Atomic Slot Acquisition: Use atomicCAS to try and claim an empty slot (KEY_INVALID).
        oldKey = atomicCAS(&new_hashmap.entries[j].key, KEY_INVALID, newKey);

        // Pre-condition: If the slot was empty, the swap was successful.
        if (oldKey == KEY_INVALID) {
            new_hashmap.entries[j].value = old_hashmap.entries[idx].value;
            break; // Exit loop after successful insertion.
        } else if (j + 1 == new_hashmap.size) {
            // Inline: Handle wrap-around for the linear probe.
            j = 0;
        }
    }

}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param hashmap The target hash table.
 * @param numKeys The number of keys in the batch.
 * @param keys A device pointer to the array of keys to insert.
 * @param values A device pointer to the array of values to insert.
 * @details Each thread handles one key-value pair. It finds a slot using linear
 * probing and uses `atomicCAS` to either claim an empty slot or update an
 * existing key. This ensures thread-safe concurrent insertions.
 */
__global__ void map_put(hash_table hashmap, int numKeys, int *keys, int *values) {
    // Thread Indexing: Each thread targets one key-value pair from the input batch.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Pre-condition: Ensure thread index is within the bounds of the batch size.
    if (idx >= numKeys) {
        return;
    }

    int oldKey;
    int newKey;
    int newHash;

    newKey = keys[idx];
    // Hashing: Compute the initial hash for the new key.
    newHash = computeHash(newKey, hashmap.size);


    // Block Logic: Perform linear probing to find or create an entry.
    // Invariant: The loop continues until the key is successfully inserted or updated.
    for (int i = newHash; i < hashmap.size; i++) {
        // Atomic Operation: Attempt to claim an empty slot or overwrite an existing key.
        oldKey = atomicCAS(&hashmap.entries[i].key, KEY_INVALID, newKey);

        // Pre-condition: If the slot was empty or held the same key, the update is safe.
        if (oldKey == KEY_INVALID || oldKey == newKey) {
            hashmap.entries[i].value = values[idx];
            return; // Insertion/update complete.
        } else if (i + 1 == hashmap.size) {
            // Inline: Handle wrap-around for the linear probe.
            i = 0;
        }
    }
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param hashmap The hash table to search in.
 * @param numKeys The number of keys to look up.
 * @param keys A device pointer to the array of keys to find.
 * @param values A device pointer to the output array for the found values.
 * @details Each thread searches for one key. It follows the linear probing path
 * to find a matching key. This operation is read-only and does not require atomic
 * operations.
 */
__global__ void map_get(hash_table hashmap, int numKeys, int *keys, int *values) {
    // Thread Indexing: Each thread targets one key from the input batch.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Pre-condition: Ensure thread index is within the bounds of the batch size.
    if (idx >= numKeys) {
        return;
    }

    int key;
    int newHash;

    key = keys[idx];
    // Hashing: Compute the initial hash to start the search.
    newHash = computeHash(key, hashmap.size);


    // Block Logic: Perform linear probing to find the key.
    // Invariant: The key, if present, must be in the probed sequence of non-empty slots.
    for (int i = newHash; i < hashmap.size; i++) {
        // Pre-condition: Check if the key at the current slot matches.
        if (hashmap.entries[i].key == key) {
            values[idx] = hashmap.entries[i].value;
            return; // Key found.
        } else if (i + 1 == hashmap.size) {
            // Inline: Handle wrap-around for the linear probe.
            i = 0;
        }
    }


}



/**
 * @brief Constructs a GpuHashTable object.
 * @param size The initial capacity (number of buckets) for the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
    entriesNumber = 0;
    hashmap.size = size;

    // Memory Hierarchy: Allocate hash table entries in global device memory.
    cudaMalloc(&hashmap.entries, size * sizeof(entry));
    // Initialize all keys to KEY_INVALID (0), marking them as empty.
    cudaMemset(hashmap.entries, 0, size * sizeof(entry));
}

/**
 * @brief Destroys the GpuHashTable object, freeing device memory.
 */
GpuHashTable::~GpuHashTable() {
    cudaFree(hashmap.entries);
}

/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The new capacity.
 * @details This function orchestrates a complete rehash of all existing elements
 * into a new, larger device memory allocation by launching the `map_resize` kernel.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
    hash_table new_hashmap;
    new_hashmap.size = numBucketsReshape;

    // Allocate memory for the new, larger hash table.
    cudaMalloc(&new_hashmap.entries, numBucketsReshape * sizeof(entry));
    cudaMemset(new_hashmap.entries, 0, numBucketsReshape * sizeof(entry));

    // Kernel Launch Configuration: Determine grid size for rehashing all elements.
    int blocks_no = hashmap.size / NUM_THREADS;
    if (hashmap.size % NUM_THREADS) {
        blocks_no++;
    }

    // Launch the kernel to migrate data from the old map to the new map.
    map_resize<<<blocks_no, NUM_THREADS>>>(hashmap, new_hashmap);
    cudaDeviceSynchronize();

    // Free the old table and update the instance to point to the new one.
    cudaFree(hashmap.entries);
    hashmap.entries = new_hashmap.entries;
    hashmap.size = new_hashmap.size;
}

/**
 * @brief Inserts a batch of keys and values into the hash table.
 * @param keys A host pointer to an array of keys.
 * @param values A host pointer to an array of values.
 * @param numKeys The number of key-value pairs in the batch.
 * @return Returns true on successful completion.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
    int *deviceKeys;
    int *deviceValues;

    // Block Logic: Check if the load factor will exceed a threshold after insertion.
    // Pre-condition: If the table is about to become too full, resize it first.
    if ((float)(entriesNumber + numKeys) / hashmap.size >= 0.7) {
        reshape(hashmap.size * 2);
    }

    // Memory Hierarchy: Allocate temporary buffers on the device for the input batch.
    cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    cudaMalloc(&deviceValues, numKeys * sizeof(int));

    // Data Transfer: Copy the batch from host to device.
    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel Launch Configuration: Determine grid size for the insertion batch.
    int blocks_no = numKeys / NUM_THREADS;
    if (numKeys % NUM_THREADS) {
        blocks_no++;
    }

    // Launch the kernel to perform the parallel insertion.
    map_put<<<blocks_no, NUM_THREADS>>>(hashmap, numKeys, deviceKeys, deviceValues);
    cudaDeviceSynchronize();

    entriesNumber += numKeys;
    cudaFree(deviceKeys);
    cudaFree(deviceValues);

	return true;
}

/**
 * @brief Retrieves a batch of values corresponding to a given set of keys.
 * @param keys A host pointer to an array of keys to look up.
 * @param numKeys The number of keys in the batch.
 * @return A host pointer to an array of values. The memory for this array is
 *         managed by CUDA (unified memory) and should be freed by the caller
 *         using `cudaFree`.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *deviceKeys;
	int *values;

    // Memory Hierarchy: Allocate device memory for keys and unified memory for values.
	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
    // Using cudaMallocManaged allows the 'values' pointer to be accessed
    // directly from the host after the kernel completes, simplifying data retrieval.
	cudaMallocManaged(&values, numKeys * sizeof(int));

    // Data Transfer: Copy the lookup keys from host to device.
    cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel Launch Configuration: Determine grid size for the lookup batch.
    int blocks_no = numKeys / NUM_THREADS;
    if (numKeys % NUM_THREADS) {
        blocks_no++;
    }

    // Launch the kernel to perform the parallel lookup.
    map_get<<<blocks_no, NUM_THREADS>>>(hashmap, numKeys, deviceKeys, values);
    cudaDeviceSynchronize(); // Ensure kernel completion before host accesses results.

    cudaFree(deviceKeys);

    return values;
}

/**
 * @brief A non-functional or debugging implementation of load factor.
 * @return A hardcoded value of 0.8.
 * @warning This function does not calculate the actual load factor. Instead, it
 *          triggers a table resize (shrink) and returns a constant. Its purpose
 *          is unclear and likely incorrect for production use.
 */
float GpuHashTable::loadFactor() {
    reshape(int(hashmap.size * 5 / 8));
	return 0.8f;
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
#define NUM_THREADS     1024

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

struct entry {
    int key;
    int value;
};

struct hash_table {
    int size;
    entry *entries;
};




class GpuHashTable
{
    int entriesNumber;
    hash_table hashmap;

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
