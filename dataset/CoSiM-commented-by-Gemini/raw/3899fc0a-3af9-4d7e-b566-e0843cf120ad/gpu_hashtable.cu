/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA with managed memory.
 *
 * This file provides a hash table data structure optimized for parallel execution on
 * NVIDIA GPUs. It utilizes an open addressing scheme with a specific linear probing
 * strategy (probing forward, then backward from the initial hash location).
 * Concurrency is managed via atomic Compare-And-Swap (CAS) operations.
 * The implementation uses `cudaMallocManaged` to simplify host-device memory
 * synchronization.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief Computes a hash value for a given key on the GPU.
 * @param data The input key to hash.
 * @param limit The size of the hash table, used for the final modulo operation.
 * @return The computed hash index within the range [0, limit-1].
 *
 * This is a device function, executed by each thread on the GPU. It uses a
 * multiplicative hashing scheme with large prime numbers.
 */
__device__ int myHash(int data, int limit) {


	return ((long) abs(data) * 1306601) % 274019932696 % limit; 
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param values Pointer to the device memory holding the values to insert.
 * @param keys Pointer to the device memory holding the keys to insert.
 * @param ht A struct containing the hash table's device pointer and size.
 * @param numEntries The number of key-value pairs in the batch.
 *
 * Each thread handles one key-value pair. It uses a linear probing strategy
 * that first probes forward to the end of the table, and then backward from the
 * initial hash position. It uses `atomicCAS` to claim an empty slot (key == 0).
 */
__global__ void kernel_insert(int *values, int *keys, hash_table ht, int numEntries) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Each thread processes one entry within the batch size.
	if (idx < numEntries) {

		int key;
		
		// Step 1: Calculate the initial hash position.
		int hash = myHash(keys[idx], ht.size);
		
		// Step 2: Linearly probe forward from the hash position to the end of the table.
		for (int i = hash; i < ht.size; i++) {
			// Attempt to atomically claim an empty slot.
			key = atomicCAS(&ht.table[i].key, 0, keys[idx]);
			// Pre-condition: Check if the slot was empty (key == 0) or already contained our key.
			bool invalid = key == keys[idx] || key == 0;
			if (invalid) {
				ht.table[i].value = values[idx];
				return; // Insertion successful.
			}
		}
		
		// Step 3: Handle wrap-around by probing backward from the hash position.
		for (int i = hash - 1; i > 0; i--) {
			key = atomicCAS(&ht.table[i].key, 0, keys[idx]);
			bool invalid = key == keys[idx] || key == 0;
			if (invalid) {
				ht.table[i].value = values[idx];
				return; // Insertion successful.
			}
		}
	}


	return;
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param values Pointer to device memory where retrieved values will be stored.
 * @param keys Pointer to device memory holding the keys to look up.
 * @param ht A struct containing the hash table's device pointer and size.
 * @param numEntries The number of keys to look up.
 *
 * Each thread searches for one key using the same forward-then-backward probing
 * strategy as the insert kernel.
 */
__global__ void kernel_get(int *values, int *keys, hash_table ht, int numEntries) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numEntries) {
		
		// Step 1: Calculate the initial hash position.
		int hash = myHash(keys[idx], ht.size);
		
		// Step 2: Linearly probe forward to find the key.
		for (int i = hash; i < ht.size; i++) {
			if (ht.table[i].key == keys[idx]) {
				values[idx] = ht.table[i].value;
				return;
			}
		}
		
		// Step 3: Probe backward if not found in the forward pass.
		for (int i = hash - 1; i > 0; i--) {
			if (ht.table[i].key == keys[idx]) {
				values[idx] = ht.table[i].value;
				return;
			}
		}
	}
}

/**
 * @brief CUDA kernel to rehash all elements from an old table into a new, larger table.
 * @param newHt The new, larger hash table.
 * @param oldHt The old hash table to rehash from.
 *
 * Each thread is responsible for rehashing one entry from the old table into the new one.
 */
__global__ void kernel_rehash(hash_table newHt, hash_table oldHt) {
	// Skip threads corresponding to empty slots in the old table.
	if (oldHt.table[blockIdx.x * blockDim.x + threadIdx.x].key == 0)
			return;
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < oldHt.size) {
		int key;
		
		// Step 1: Re-calculate the hash for the key in the context of the new table size.
		int hash = myHash(oldHt.table[idx].key, newHt.size);
		
		// Step 2: Insert into the new table using the same probing logic as kernel_insert.
		for (int i = hash; i < newHt.size; i++) {
			key = atomicCAS(&newHt.table[i].key, 0, oldHt.table[idx].key);
			if (key == 0) { // Successfully claimed an empty slot.
				newHt.table[i].value = oldHt.table[idx].value;
				return;
			} 
		}
		
		// Step 3: Handle wrap-around by probing backward.
		for (int i = hash - 1; i > 0; i--) {
			key = atomicCAS(&newHt.table[i].key, 0, oldHt.table[idx].key);
			if (key == 0) {
				newHt.table[i].value = oldHt.table[idx].value;
				return;
			} 
		}
	}
	return;
}


/**
 * @brief Host-side constructor for the GpuHashTable class.
 * @param size The initial capacity of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	inserted = 0;
	ht.size = size;
	ht.table = nullptr;
	
	// Allocate GPU memory for the hash table array.
	if (cudaMalloc(&ht.table, size * sizeof(entry)) != cudaSuccess) {
		std::cerr << "Memory allocation failed! 
";
		return;
	}
	// Initialize the allocated GPU memory to zero.
	cudaMemset(ht.table, 0, size * sizeof(entry));
}

/**
 * @brief Host-side destructor for the GpuHashTable class.
 */
GpuHashTable::~GpuHashTable() {
	// Free the GPU memory.
	cudaFree(ht.table);
}

/**
 * @brief Resizes the hash table on the GPU.
 * @param numBucketsReshape The new capacity for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	
	hash_table newHt;
	newHt.size = numBucketsReshape;
	
	// Allocate memory for the new table.
	if (cudaMalloc(&newHt.table, numBucketsReshape * sizeof(entry)) != cudaSuccess) {
		std::cerr << "Memory allocation failed! 
";
		return;
	}
	cudaMemset(newHt.table, 0, numBucketsReshape * sizeof(entry));
	
	// Calculate kernel launch parameters for the rehash operation.
	unsigned int blocks;
	blocks = ht.size / NUM_THREADS + 1;

	// Launch the kernel to copy and rehash elements.
	kernel_rehash>>(newHt, ht);

	// Block until the rehash is complete.
	cudaDeviceSynchronize();
	
	// Free the old table memory.
	cudaFree(ht.table);
	
	// Point to the new table.
	ht = newHt;
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys Host pointer to an array of keys.
 * @param values Host pointer to an array of values.
 * @param numKeys The number of pairs to insert.
 * @return True on success, false on failure.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	
	int *deviceKeys;
	// Use managed memory for automatic data migration between host and device.
	cudaMallocManaged(&deviceKeys, numKeys * sizeof(int));
	
	int *deviceValues;
	cudaMallocManaged(&deviceValues, numKeys * sizeof(int));

	float loadFactor;
	
	if (deviceValues == nullptr || deviceKeys == nullptr) {
		std::cerr << "Memory allocation failed! 
";
		return false;
	}
	
	// Invariant: Check if the load factor will exceed the maximum threshold.
	loadFactor = float(inserted + numKeys) / ht.size;
	
	// If so, trigger a reshape to maintain performance.
	if (loadFactor >= 0.95)
		reshape(int((inserted + numKeys) / 0.85));
	
	// Copy data from host arrays to the managed device arrays.
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	// Calculate kernel launch parameters.
	unsigned int blocks = numKeys / NUM_THREADS + 1;
	// Launch the insertion kernel.
	kernel_insert>>(deviceValues, 
	        deviceKeys, ht, numKeys);

	// Wait for the kernel to finish.
	cudaDeviceSynchronize();
	
	// Free the temporary managed buffers.
	cudaFree(deviceKeys);
	cudaFree(deviceValues);
	
	inserted = inserted + numKeys;

	return true;
}

/**
 * @brief Retrieves a batch of values corresponding to a batch of keys.
 * @param keys Host pointer to an array of keys to look up.
 * @param numKeys The number of keys to retrieve.
 * @return A pointer to a managed memory array with the results. The caller must eventually free this with `cudaFree`.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	
	int *deviceKeys;
	cudaMallocManaged(&deviceKeys, numKeys * sizeof(int));
	int *values;
	cudaMallocManaged(&values, numKeys * sizeof(int));
	
	if (values == nullptr || deviceKeys  == nullptr) {
		std::cerr << "Memory allocation failed!
";
		return nullptr;
	}
	
	// Copy input keys to managed memory.
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	unsigned int blocks = numKeys / NUM_THREADS + 1;
	
	// Launch the retrieval kernel.
	kernel_get>>(values, deviceKeys, ht, numKeys);
	
	// Wait for the kernel to complete so the results are visible to the host.
	cudaDeviceSynchronize();
	
	cudaFree(deviceKeys);

	// Return the pointer to managed memory containing the results.
	return values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float (number of items / capacity).
 */
float GpuHashTable::loadFactor() {
	if (ht.size != 0)
		return (float(inserted) / ht.size);
	return 0;
}


// Testing framework macros.
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"

// Redefinition of header content.
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#include 
#include 

#define KEY_INVALID 0
#define NUM_THREADS 256

// Macro for basic error handling.
#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)
	
// Pre-computed list of prime numbers.
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


// Alternative hashing functions.
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

// C-style struct for a hash table entry.
struct entry {
	int key, value;
};

// C-style struct to hold the hash table data and size.
struct hash_table {
	int size;
	entry *table;
};



// C++ class providing an API for the GPU hash table.
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
