
/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table using open addressing and linear probing.
 * @details This file implements a hash table on the GPU using CUDA. It employs
 * an open addressing strategy where collisions are resolved via linear probing.
 * The implementation correctly uses atomic operations to handle race conditions during
 * the claiming of new slots, making it largely thread-safe for parallel insertions.
 * However, it contains a subtle race condition on value updates.
 *
 * NOTE: The file appears to have a header (`gpu_hashtable.hpp`) concatenated at the end.
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
 * @param data The key to hash.
 * @param limit The size of the hash table, used for the modulo operation.
 * @return An integer hash value within the range [0, limit-1].
 * @note This is a __device__ function, executed on the GPU by each thread.
 */
__device__ int myhashKernel(int data, int limit) {
	return ((long long)abs(data) * PRIMENO) % PRIMENO2 % limit;
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param keys Device pointer to an array of keys to insert.
 * @param values Device pointer to an array of corresponding values.
 * @param limit The total number of pairs to insert.
 * @param myTable The GPU hash table data structure.
 *
 * @warning RACE CONDITION: While this function correctly uses `atomicCAS` to claim
 * empty slots, the value update (`myTable.list[i].value = values[index]`) is
 * non-atomic. If multiple threads attempt to update the value for the *same existing key*
 * simultaneously, some updates may be lost. A safer implementation would use `atomicExch`.
 */
__global__ void insert_kerFunc(int *keys, int *values, int limit, hashtable myTable) {
	int newIndex;
	// Determine the unique index for this thread.
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int old, i, j;
	int low, high;

	if (index < limit) {
		newIndex = myhashKernel(keys[index], myTable.size);
		
		// --- Linear Probing with Wrap-around ---
		// This structure probes from the initial hash index to the end of the table,
		// then wraps around to probe from the beginning to the hash index.
		low = newIndex;
		high = myTable.size;
		for(j = 0; j < 2; j++) { // j=0 for first part, j=1 for wrap-around
			for (i = low; i < high; i++) {
				// Synchronization: Atomically check if the slot contains our key or is empty,
				// and if empty, claim it with our key.
				old = atomicCAS(&myTable.list[i].key, 0, keys[index]);
				if (old == keys[index] || old == 0) {
					// BUG: Non-atomic write. This can cause lost updates if multiple
					// threads are inserting/updating the same key concurrently.
					myTable.list[i].value = values[index];
					return;
				}
			}
			// Setup for the wrap-around probe.
			low = 0;
			high = newIndex;
		}

	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param myTable The GPU hash table data structure.
 * @param keys Device pointer to an array of keys to look up.
 * @param values Device pointer to an array to store found values.
 * @param limit The number of keys to retrieve.
 */
__global__ void get_kerFunc(hashtable myTable, int *keys, int *values, int limit) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j;
	int newIndex;
	int low, high;

	if (index < limit) {
		newIndex = myhashKernel(keys[index], myTable.size);
		
		// --- Search with Linear Probing and Wrap-around ---
		low = newIndex;
		high = myTable.size;
		for(j = 0; j < 2; j++) {
			for (i = low; i < high; i++) { 
				// A simple, non-atomic read is sufficient and correct for a 'get' operation.
				if (myTable.list[i].key == keys[index]) {
					values[index] = myTable.list[i].value;
					return;
				}
			}
			low = 0;
			high = newIndex;	
		}
	}
}

/**
 * @brief CUDA kernel to resize the hash table by re-hashing all elements.
 * @param oldTable The original hash table.
 * @param newTable The new, larger hash table to populate.
 * @param size The size of the new hash table.
 */
__global__ void reshapeKerFunc(hashtable oldTable, hashtable newTable, int size) {
	int newIndex, i, j;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int low, high;

	// Guard against out-of-bounds access on the old table.
	if (index >= oldTable.size)
		return;

	// Only process valid entries from the old table.
	if (oldTable.list[index].key == 0)
	    return;

	// Re-hash the key for the new table size.
	newIndex = myhashKernel(oldTable.list[index].key, size);
	
	// --- Re-insertion with Linear Probing ---
	low = newIndex;
	high = size;
	for(j = 0; j < 2; j++) {
		for (i = low; i < high; i++) {
			// Atomically claim an empty slot in the new table.
			if (atomicCAS(&newTable.list[i].key, 0,
				oldTable.list[index].key) == 0) {
				newTable.list[i].value = oldTable.list[index].value;
				return;
			}
		}
		low = 0;
		high = newIndex;
	}
}

/**
 * @brief Host-side constructor. Allocates memory on the GPU.
 */
GpuHashTable::GpuHashTable(int size) {
	myTable.size = size;
	myTable.slotsTaken = 0;
	// Allocate GPU memory for the hash table array.
	DIE(cudaMalloc(&myTable.list, size * sizeof(node)) != cudaSuccess, "Could not allocate memory for table list on init.
"); 
	// Initialize GPU memory to zero.
	cudaMemset(myTable.list, 0, size * sizeof(node));
}

/**
 * @brief Host-side destructor. Frees GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(myTable.list);
}

/**
 * @brief Resizes the hash table to a new, larger capacity.
 * @param numBucketsReshape The new number of buckets.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hashtable newTable;
	int noBlocks;
	
	newTable.size = numBucketsReshape;
	newTable.slotsTaken = myTable.slotsTaken;

	// 1. Allocate memory for the new, larger table.
	DIE(cudaMalloc(&newTable.list, numBucketsReshape * sizeof(node)) != cudaSuccess, "Could not allocate memory for new table list.
"); 	
	cudaMemset(newTable.list, 0, numBucketsReshape * sizeof(node));

	// 2. Calculate kernel launch parameters.
	noBlocks = (myTable.size % BLOCK_SIZE != 0) ?
		(myTable.size / BLOCK_SIZE + 1) : (myTable.size / BLOCK_SIZE);
	
	// 3. Launch the kernel to perform parallel re-hashing.
	reshapeKerFunc>>(myTable, newTable, numBucketsReshape);
	
	cudaDeviceSynchronize();
	
	// 4. Free the old table and update the pointer.
	cudaFree(myTable.list);
	myTable = newTable;
}

/**
 * @brief Inserts a batch of key-value pairs from the host.
 * @return True on success.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *device_keys, *device_values;
	int noBlocks;
	myTable.slotsTaken += numKeys;

	size_t memSize = numKeys * sizeof(int);
	// 1. Allocate temporary device buffers.
	DIE(cudaMalloc(&device_keys, numKeys * sizeof(int)) != cudaSuccess, "Could not allocate memory for device keys.
");
	DIE(cudaMalloc(&device_values, numKeys * sizeof(int)) != cudaSuccess, "Could not allocate memory for device values.
");

	// 2. Check load factor and resize if it exceeds the threshold.
	if ((float(myTable.slotsTaken) / myTable.size) >= 0.95)
		reshape(int((myTable.slotsTaken) / 0.8));

	// 3. Copy data from host to device.
	cudaMemcpy(device_keys, keys, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, memSize, cudaMemcpyHostToDevice);

	// 4. Calculate launch configuration and launch the insertion kernel.
	noBlocks = (numKeys % BLOCK_SIZE != 0) ?
		(numKeys / BLOCK_SIZE + 1) : (numKeys / BLOCK_SIZE);
	insert_kerFunc>>(device_keys, device_values, numKeys, myTable);

	cudaDeviceSynchronize();

	// 5. Free temporary buffers.
	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}

/**
 * @brief Retrieves values for a batch of keys from the host.
 * @return A device pointer to an array with the results.
 * @note Unconventional API: Returns a device pointer allocated with `cudaMallocManaged`.
 * The caller can access this pointer from the host due to Unified Memory, but is
 * responsible for freeing it with `cudaFree`.
 */
 int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_values;
    int *device_keys;
    int noBlocks;

	// Allocate Unified Memory for the results, accessible by both host and device.
	DIE(cudaMallocManaged(&device_values, numKeys * sizeof(int)) != cudaSuccess, "Could not allocate memory for device values.
");
	DIE(cudaMalloc(&device_keys, numKeys * sizeof(int)) != cudaSuccess, "Could not allocate memory for device keys.
");

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	noBlocks = (numKeys % BLOCK_SIZE != 0) ?
		(numKeys / BLOCK_SIZE + 1) : (numKeys / BLOCK_SIZE);

	get_kerFunc >> (myTable, device_keys, device_values, numKeys);
	
	cudaDeviceSynchronize();
	
	// Free the temporary key buffer. The value buffer is returned to the caller.
	cudaFree(device_keys);
	return device_values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
 float GpuHashTable::loadFactor() {
	return (myTable.size > 0) ? float(myTable.slotsTaken) / myTable.size : 0;
}


// --- Test harness and C-style definitions ---
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0 // Defines the value for an empty slot.

// Macro for simple error checking and exiting.
#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)
	
// A list of prime numbers for hashing.
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
// Primes used in the hash function.
#define PRIMENO 4148279llu
#define PRIMENO2 53944293929llu
// Default kernel launch block size.
#define BLOCK_SIZE 1024


// Alternative hash functions (unused).
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}


// Data structure for a single key-value pair.
struct node {
	int key;
	int value;
};
// Data structure representing the hash table's state.
struct hashtable {
	int size;
	int slotsTaken;
	node *list;
};
class GpuHashTable
{	
	hashtable myTable;
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
