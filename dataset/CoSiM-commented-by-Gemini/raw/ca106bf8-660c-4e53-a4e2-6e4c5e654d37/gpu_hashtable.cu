
/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table using a flawed open addressing scheme.
 * @details This file implements a hash table on the GPU using CUDA. It attempts
 * to use open addressing with linear probing. It correctly uses atomic operations
 * for claiming empty slots during insertion and has a proper re-hashing strategy
 * for resizing.
 *
 * @warning CRITICAL ALGORITHMIC FLAW: The linear probing implementation in all
 * kernels (`kernel_insert`, `kernel_get`, `kernel_rehash`) is incorrect. The probing
 * sequence does not cover all slots in the hash table, which will cause both
 * insertions and lookups to fail even when slots are available or keys are present.
 *
 * @warning RACE CONDITION: The `kernel_insert` function contains a race condition
 * on the non-atomic value update for existing keys, which can lead to data loss.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief Computes a hash value for a given key.
 * @param data The key to hash.
 * @param limit The size of the hash table (number of slots).
 * @return An integer hash value within the range [0, limit-1].
 */
__device__ int myHash(int data, int limit) {
	// A simple multiply-and-modulo hashing scheme using large prime numbers.
	return ((long) abs(data) * 1306601) % 274019932696 % limit; 
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs.
 *
 * @warning FLAWED PROBING LOGIC: The linear probing loops do not cover all
 * buckets. For example, bucket 0 is missed in the second loop, and the probe
 * does not correctly wrap around the end of the array. This will cause
 * insertions to fail incorrectly on a partially full table.
 */
__global__ void kernel_insert(int *values, int *keys, hash_table ht, int numEntries) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < numEntries) {
		int key;
		int hash = myHash(keys[idx], ht.size);
		
		// --- Flawed Linear Probing (Part 1) ---
		// Probes from the hash value to the end of the table.
		for (int i = hash; i < ht.size; i++) {
			// Atomically claim a slot if it's empty or already holds the correct key.
			key = atomicCAS(&ht.table[i].key, 0, keys[idx]);
			bool invalid = key == keys[idx] || key == 0;
			if (invalid) {
				// RACE CONDITION: Non-atomic write. Can lead to lost updates if multiple
				// threads are updating the same existing key.
				ht.table[i].value = values[idx];
				return;
			}
		}
		
		// --- Flawed Linear Probing (Part 2 - Incorrect Wrap-around) ---
		// This loop incorrectly probes backwards from hash-1, failing to cover all slots.
		for (int i = hash - 1; i > 0; i--) {
			key = atomicCAS(&ht.table[i].key, 0, keys[idx]);
			bool invalid = key == keys[idx] || key == 0;
			if (invalid) {
				ht.table[i].value = values[idx];
				return;
			}
		}
	}
	return;
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 *
 * @warning FLAWED PROBING LOGIC: This kernel uses the same incorrect linear
 * probing strategy as `kernel_insert`. As a result, it may fail to find keys
 * that are present in the table.
 */
__global__ void kernel_get(int *values, int *keys, hash_table ht, int numEntries) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < numEntries) {
		int hash = myHash(keys[idx], ht.size);
		
		// --- Flawed Linear Probing Search (Part 1) ---
		for (int i = hash; i < ht.size; i++) {
			if (ht.table[i].key == keys[idx]) {
				values[idx] = ht.table[i].value;
				return;
			}
		}
		
		// --- Flawed Linear Probing Search (Part 2) ---
		for (int i = hash - 1; i > 0; i--) {
			if (ht.table[i].key == keys[idx]) {
				values[idx] = ht.table[i].value;
				return;
			}
		}
	}
}

/**
 * @brief CUDA kernel to resize the hash table by re-hashing all elements.
 * @warning FLAWED PROBING LOGIC: This kernel also uses the incorrect linear probing
 * strategy, which may cause it to fail during the re-hashing process.
 */
__global__ void kernel_rehash(hash_table newHt, hash_table oldHt) {
	// Each thread processes one slot from the old hash table.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= oldHt.size || oldHt.table[idx].key == 0)
		return;
	
	int key;
	int hash = myHash(oldHt.table[idx].key, newHt.size);
	
	// --- Re-insertion with Flawed Linear Probing ---
	for (int i = hash; i < newHt.size; i++) {
		key = atomicCAS(&newHt.table[i].key, 0, oldHt.table[idx].key);
		if (key == 0) {
			newHt.table[i].value = oldHt.table[idx].value;
			return;
		} 
	}
	
	for (int i = hash - 1; i > 0; i--) {
		key = atomicCAS(&newHt.table[i].key, 0, oldHt.table[idx].key);
		if (key == 0) {
			newHt.table[i].value = oldHt.table[idx].value;
			return;
		} 
	}
	return;
}


/**
 * @brief Host-side constructor. Allocates memory on the GPU.
 */
GpuHashTable::GpuHashTable(int size) {
	inserted = 0;
	ht.size = size;
	ht.table = nullptr;
	
	if (cudaMalloc(&ht.table, size * sizeof(entry)) != cudaSuccess) {
		std::cerr << "Memory allocation failed! 
";
		return;
	}
	cudaMemset(ht.table, 0, size * sizeof(entry));
}

/**
 * @brief Host-side destructor. Frees GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(ht.table);
}

/**
 * @brief Resizes the hash table to a new, larger capacity.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hash_table newHt;
	newHt.size = numBucketsReshape;
	
	// 1. Allocate memory for the new table.
	if (cudaMalloc(&newHt.table, numBucketsReshape * sizeof(entry)) != cudaSuccess) {
		std::cerr << "Memory allocation failed! 
";
		return;
	}
	cudaMemset(newHt.table, 0, numBucketsReshape * sizeof(entry));
	
	// 2. Launch the re-hashing kernel.
	unsigned int blocks = ht.size / NUM_THREADS + 1;
	kernel_rehash>>(newHt, ht);
	cudaDeviceSynchronize();
	
	// 3. Free the old table and update the pointer.
	cudaFree(ht.table);
	ht = newHt;
}

/**
 * @brief Inserts a batch of key-value pairs from the host.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *deviceKeys;
	int *deviceValues;
	
	// Use Unified Memory, accessible from both host and device.
	cudaMallocManaged(&deviceKeys, numKeys * sizeof(int));
	cudaMallocManaged(&deviceValues, numKeys * sizeof(int));

	if (deviceValues == nullptr || deviceKeys == nullptr) {
		std::cerr << "Memory allocation failed! 
";
		return false;
	}
	
	// 1. Check load factor and resize if it exceeds the threshold.
	float loadFactor = float(inserted + numKeys) / ht.size;
	if (loadFactor >= 0.95)
		reshape(int((inserted + numKeys) / 0.85));
	
	// 2. Copy data from host to device.
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	// 3. Launch the insertion kernel.
	unsigned int blocks = numKeys / NUM_THREADS + 1;
	kernel_insert>>(deviceValues, 
	        deviceKeys, ht, numKeys);
	cudaDeviceSynchronize();
	
	// 4. Free temporary buffers.
	cudaFree(deviceKeys);
	cudaFree(deviceValues);
	
	inserted = inserted + numKeys;

	return true;
}

/**
 * @brief Retrieves values for a batch of keys from the host.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *deviceKeys;
	int *values; // This will be a Unified Memory pointer.
	
	// 1. Allocate Unified Memory for keys and result values.
	cudaMallocManaged(&deviceKeys, numKeys * sizeof(int));
	cudaMallocManaged(&values, numKeys * sizeof(int));
	
	if (values == nullptr || deviceKeys  == nullptr) {
		std::cerr << "Memory allocation failed!
";
		return nullptr;
	}
	
	// 2. Copy keys to device.
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	// 3. Launch the get kernel.
	unsigned int blocks = numKeys / NUM_THREADS + 1;
	kernel_get>>(values, deviceKeys, ht, numKeys);
	cudaDeviceSynchronize();
	
	// 4. Free the temporary key buffer.
	cudaFree(deviceKeys);

	// 5. Return the Unified Memory pointer to the values. The caller on the host can
	// access this directly but is responsible for freeing it with cudaFree.
	return values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
	if (ht.size != 0)
		return (float(inserted) / ht.size);
	return 0;
}


// --- Test harness and boilerplate ---
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"

#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#include 
#include 

#define KEY_INVALID 0
#define NUM_THREADS 256

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


// Unused hash functions.
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

// Data structure for a single hash table entry (key-value pair).
struct entry {
	int key, value;
};

// Data structure representing the hash table's state.
struct hash_table {
	int size;
	entry *table;
};

class GpuHashTable
{
	int inserted;
	hash_table ht;

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
