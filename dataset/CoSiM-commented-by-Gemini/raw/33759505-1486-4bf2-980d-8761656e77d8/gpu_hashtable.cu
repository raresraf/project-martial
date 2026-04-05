/**
 * @file gpu_hashtable.cu
 * @brief A GPU-based hash table implementation using open addressing with linear probing.
 *
 * This file provides a hash table that resides entirely in GPU memory. It uses
 * a single contiguous array for storage and resolves hash collisions by linearly
 * probing for the next available bucket. The implementation is notable for its
 * robust, bounded probing loops and an efficient, GPU-only reshape mechanism.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief A device function to compute a hash for a given key.
 * @param data The integer key to hash.
 * @param limit The total number of buckets in the hash table.
 * @return An integer hash value within the bounds of the table size.
 *
 * This function uses a simple multiplicative hashing scheme with large primes
 * to help distribute keys evenly across the hash table.
 */
__device__ int computeHash(int data, int limit) {
    return ((long)abs(data) * 59llu) % 10703903591llu % limit;
}

/**
 * @brief CUDA kernel for inserting a batch of key-value pairs.
 *
 * Each thread handles one key-value pair. It computes an initial hash and then
 * uses a bounded `do-while` loop to perform linear probing. `atomicCAS` is used
 * to ensure thread-safe insertion into an empty slot or updating of an existing key.
 */
__global__ void insert(int *keys, int *values, int numKeys, hash_table hashtable) {
	
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
 
    if (i < numKeys) {
        unsigned int newKey = keys[i];
        unsigned int hash = computeHash(newKey, hashtable.size);
        unsigned int counter = hash;

        // Loop until the element is inserted or the table is fully probed.
        do {
            // Atomically attempt to write `newKey` if the slot is empty (KEY_INVALID)
            // or if the slot already holds the same key (for updates).
            unsigned int key = atomicCAS(&hashtable.table[counter].key, KEY_INVALID, newKey);

            if (key == KEY_INVALID || key == newKey) {
                hashtable.table[counter].value = values[i];
                return; // Success
            }
            // If the slot was occupied by a different key, probe the next one.
            counter = (counter + 1) % hashtable.size;
        } while (counter != hash); // The loop terminates if a full cycle is completed.
    }
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 *
 * Each thread searches for one key. It uses a bounded `do-while` loop for
 * linear probing, which safely terminates if the key is not found after
 * checking all buckets.
 */
__global__ void get(int *keys, int *values, int numKeys, hash_table hashtable) {
    
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (i < numKeys) {
        unsigned int searchKey = keys[i];
        unsigned int hash = computeHash(searchKey, hashtable.size);
        unsigned int counter = hash;

        // Linearly probe the table to find the key.
        do {
            unsigned int currentKey = hashtable.table[counter].key;
            // A non-atomic read is sufficient and efficient for the check.
            if (searchKey == currentKey) {
                values[i] = hashtable.table[counter].value;
                return; // Key found.
            }
            counter = (counter + 1) % hashtable.size;
        } while (counter != hash); // Safely terminates after one full cycle.
    }
}

/**
 * @brief CUDA kernel to rehash all elements from an old table into a new, larger one.
 *
 * This kernel is launched during a `reshape` operation. Each thread takes one element
 * from the old table and re-inserts it into the new table using the same atomic
 * linear probing logic as the `insert` kernel.
 */
__global__ void reshape_table(hash_table newTable, hash_table oldTable) {
    
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (i < oldTable.size) {
        unsigned int oldKey = oldTable.table[i].key;

        // Skip empty slots in the old table.
        if (oldKey == KEY_INVALID)
            return;

        // Re-hash the key for the new table size.
        unsigned int newHash = computeHash(oldKey, newTable.size);
        unsigned int counter = newHash;

        // Use atomic linear probing to insert into the new table.
        do {
            unsigned int newKey = atomicCAS(&newTable.table[counter].key, KEY_INVALID, oldKey);

            if (newKey == KEY_INVALID || newKey == oldKey) {
                newTable.table[counter].value = oldTable.table[i].value;
                return;
            }
            counter = (counter + 1) % newTable.size; // Bug: should be newTable.size
        } while (counter != newHash);
    }
}

/**
 * @brief Host-side constructor. Allocates and initializes the hash table on the GPU.
 */
GpuHashTable::GpuHashTable(int size) {
    hashtable.size = size;
    hashtable.occupied = 0;

    cudaError_t rc = cudaMalloc(&hashtable.table, size * sizeof(entry));
    DIE(rc != cudaSuccess, cudaGetErrorString(rc));

    // Initialize all keys in the hash table to KEY_INVALID (0).
    cudaMemset(hashtable.table, 0, size * sizeof(entry));
}

/**
 * @brief Host-side destructor. Frees the GPU memory used by the hash table.
 */
GpuHashTable::~GpuHashTable() {
    cudaFree(hashtable.table);
}

/**
 * @brief Resizes the hash table and rehashes all elements entirely on the GPU.
 */
void GpuHashTable::reshape(int numBucketsReshape) {

    hash_table newHashTable;
    newHashTable.occupied = hashtable.occupied;
    newHashTable.size = numBucketsReshape;

    // 1. Allocate a new, larger table on the device.
    cudaError_t rc = cudaMalloc(&newHashTable.table, numBucketsReshape * sizeof(entry));
    DIE(rc != cudaSuccess, cudaGetErrorString(rc));

    cudaMemset(newHashTable.table, 0, numBucketsReshape * sizeof(entry));

    // 2. Configure and launch the rehash kernel.
    unsigned int blocks_no = hashtable.size / BLOCK_SIZE;
    if (hashtable.size % BLOCK_SIZE) 
        ++blocks_no;
    reshape_table>>(newHashTable, hashtable);
	cudaDeviceSynchronize();

    // 3. Free the old table memory.
    cudaFree(hashtable.table);

    // 4. Update the host-side struct to point to the new table.
	hashtable = newHashTable;
}

/**
 * @brief Host-side function to insert a batch of key-value pairs.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
    
    int *device_array_keys;
    int *device_array_values;

    size_t array_size = numKeys * sizeof(int);

    cudaError_t rc = cudaMalloc(&device_array_keys, array_size);
    DIE(rc != cudaSuccess, cudaGetErrorString(rc));
    rc = cudaMalloc(&device_array_values, array_size);
    DIE(rc != cudaSuccess, cudaGetErrorString(rc));

	// Check if the insertion will exceed the target load factor and reshape if so.
	if (1.f * (hashtable.occupied + numKeys) / hashtable.size >= 0.8f)
		reshape(int((hashtable.occupied + numKeys) / 0.6f));

	// Copy the input batch from host to device.
	cudaMemcpy(device_array_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_array_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	unsigned int blocks_no = hashtable.size / BLOCK_SIZE;
    if (hashtable.size % BLOCK_SIZE) 
        ++blocks_no;

	// Launch the insertion kernel.
	insert>>(device_array_keys,
        device_array_values, numKeys, hashtable);
    cudaDeviceSynchronize();
    
    hashtable.occupied += numKeys;

	// Free temporary device memory.
	cudaFree(device_array_keys);
	cudaFree(device_array_values);

	return true;
}

/**
 * @brief Host-side function to retrieve values for a batch of keys.
 * @return A pointer to a host array containing the results. The caller must free this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {

	int *device_array_keys;
    int *device_array_values;
    int *host_array_values;

    // Allocate temporary device memory for keys and values.
    cudaError_t rc = cudaMalloc(&device_array_keys, numKeys * sizeof(int));
    DIE(rc != cudaSuccess, cudaGetErrorString(rc));
    rc = cudaMalloc(&device_array_values, numKeys * sizeof(int));
    DIE(rc != cudaSuccess, cudaGetErrorString(rc));

    // Allocate host memory for the final results.
    host_array_values = (int*)malloc(numKeys * sizeof(int));
    DIE(!host_array_values, "Failed host malloc in getBatch");

	// Copy input keys from host to device.
	cudaMemcpy(device_array_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	// Configure and launch the get kernel.
	unsigned int blocks_no = hashtable.size / BLOCK_SIZE;
    if (hashtable.size % BLOCK_SIZE) 
        ++blocks_no;
	get>>(device_array_keys,
        device_array_values, numKeys, hashtable);
    cudaDeviceSynchronize();
    
    // Copy results from device back to host.
    cudaMemcpy(host_array_values, device_array_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	// Clean up temporary device memory.
	cudaFree(device_array_keys);
	cudaFree(device_array_values);

	return host_array_values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
    if (hashtable.size == 0)
        return 0.f;
    return 1.f * hashtable.occupied / hashtable.size;
}


// --- Convenience macros for the test file ---
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
// The file includes a header guard and content that should be in gpu_hashtable.hpp
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

// Sentinel value representing an empty bucket in the hash table.
#define	KEY_INVALID		0

// Default number of threads per CUDA block.
#define BLOCK_SIZE		1024

// C-style error handling macro for CUDA calls.
#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)
	
// A large list of prime numbers, likely for hashing, though not used by computeHash.
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


// These hash functions are defined but are not used in the kernel code.
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

// A simple key-value pair struct.
struct entry {
	unsigned int key, value;
};

// A struct to hold the device pointer and metadata for the hash table.
struct hash_table {
	unsigned int size;
	unsigned int occupied;
	entry *table;
};


/**
 * @brief The host-side class providing an interface to the GPU hash table.
 */
class GpuHashTable {
	hash_table hashtable;

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

