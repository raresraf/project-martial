/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA and linear probing.
 *
 * @details
 * This file implements a hash table that operates on the GPU. It uses an array of
 * `HashItem` structs (key-value pairs) and resolves collisions using linear probing.
 * The implementation includes kernels for initialization, batch insertion, batch
 * retrieval, and dynamic resizing (reshaping) to manage the load factor.
 *
 * @algorithm
 * The core algorithm is open-addressing with linear probing.
 * - Hashing: A multiplicative hash function (`createHash`) determines the initial bucket.
 * - Insertion: `insertOneItem` attempts to place a key-value pair. If the initial
 *   bucket is occupied, it probes subsequent buckets linearly (with wrap-around).
 *   Concurrency is managed with `atomicCAS` to safely claim an empty slot or update
 *   an existing key.
 * - Retrieval: `get` uses the same linear probing sequence to locate a key.
 * - Reshaping: When the load factor exceeds a threshold, a new, larger table is
 *   allocated, and all elements from the old table are rehashed into the new one.
 *
 * @note The file follows an unconventional structure, appearing to concatenate the
 * implementation with its own header definitions.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief Computes a hash value for a given key on the device.
 * @param data The key to hash.
 * @param limit The capacity of the hash table, used for the modulo operation.
 * @return The calculated hash index.
 */
__device__ int createHash(int data, int limit) {
	return ((long long) abs(data) * PARAM_1_HASH) % PARAM_2_HASH % limit;
}

/**
 * @brief CUDA kernel to initialize all keys in the hash table to an invalid state.
 * @param hashTable The hash table structure containing the map and capacity.
 * @details Each thread initializes one bucket.
 */
__global__ void initHashT(HashTable hashTable)
{
	// Thread Indexing: Each thread is assigned one bucket to initialize.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= hashTable.capacity)
		return;

	// Set the key to KEY_INVALID (0) to mark the slot as empty.
	hashTable.map[idx].key = KEY_INVALID;
}

/**
 * @brief Device function to insert a single key-value pair using linear probing.
 * @param hashTable The hash table to insert into.
 * @param newKey The key to insert.
 * @param newValue The value to associate with the key.
 * @return 1 if a new element was inserted, 0 if an existing element was updated.
 *
 * @details This function is called by the `insert` and `reshapeHT` kernels.
 */
__device__ int insertOneItem(HashTable hashTable, int newKey, int newValue)
{
	int n = 0;
	int existingKey;
	int startPos = createHash(newKey, hashTable.capacity);

	// Block Logic: Linear probing loop. Iterates through the table to find a slot.
	for (int i = 0; i < hashTable.capacity; ++i) {
    	
		// Synchronization: Atomically attempt to claim an empty slot.
    	existingKey = atomicCAS(&hashTable.map[startPos].key, KEY_INVALID, newKey);
    	
		// Case 1: Slot was empty (KEY_INVALID) and we successfully claimed it.
		// Case 2: Slot already contained our key.
		if (existingKey == KEY_INVALID || existingKey == newKey) {
    		hashTable.map[startPos].value = newValue; // Set/update the value.
    		if (existingKey == KEY_INVALID) {
    			n = 1; // Mark that a new key was inserted.
    		}
    		break; // Exit the loop on success.
    	}
    	
		// Collision: Probe the next slot with wrap-around.
		if (startPos == hashTable.capacity - 1)
			startPos = 0;
		else
			startPos++;
	}
	return n;
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs.
 * @param deviceExistKeys A counter for the number of newly inserted keys (vs. updated).
 * @details Each thread calls the `insertOneItem` device function to perform the insertion.
 */
__global__ void insert(HashTable hashTable, int *deviceKeys, int *deviceValues,
											 int numKeys, int *deviceExistKeys)
{
	// Thread Indexing: Each thread handles one key-value pair.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numKeys)
		return;

    int ret = insertOneItem(hashTable, deviceKeys[idx], deviceValues[idx]);
    // Atomically update the counter of newly added keys.
	atomicAdd(&(*deviceExistKeys), ret);
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @details Each thread searches for one key using linear probing.
 */
__global__ void get(HashTable hashTable, int *deviceKeys, int *deviceValues, int numKeys)
{
	// Thread Indexing: Each thread is assigned one key to search for.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numKeys)
		return;
    
    int searchedKey = deviceKeys[idx];
    int startPos = createHash(searchedKey, hashTable.capacity);

    // Block Logic: Linear probing search.
    for (int i = 0; i < hashTable.capacity; ++i) {
    	if (hashTable.map[startPos].key == searchedKey) {
    		// Key found, write the value to the output array.
			deviceValues[idx] = hashTable.map[startPos].value;
    		break;
    	}
		// If an empty slot is found (key == KEY_INVALID), the key is not in the table.
		// This check is implicitly handled by the loop terminating after 'capacity' steps
		// if the key is not found.
		if (startPos == hashTable.capacity - 1)
			startPos = 0;
		else
			startPos++;
	}
}

/**
 * @brief CUDA kernel to rehash all elements from an old table to a new one.
 * @details Each thread takes one item from the old table and inserts it into the new
 * table using the `insertOneItem` device function.
 */
__global__ void reshapeHT(HashTable hashTable, HashTable newHashTable)
{
	// Thread Indexing: Each thread is responsible for one slot in the old table.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= hashTable.capacity)
		return;
	
	// Pre-condition: Only rehash valid items.
	if (hashTable.map[idx].key != KEY_INVALID)
		insertOneItem(newHashTable, hashTable.map[idx].key, hashTable.map[idx].value);
}

/**
 * @brief Constructor for the GpuHashTable class.
 * @param size Initial capacity of the hash table.
 * @details Allocates memory on the GPU and initializes the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	cudaError_t err;
	int minGridSize, blocksize, gridSize;

	hashTable.capacity = size;
	
	hostExistKeys = (int*)calloc(1,sizeof(int));
	DIE(hostExistKeys == NULL, "calloc");
	
	// Memory Allocation: Allocate device memory for a counter and the main table.
	err = cudaMalloc(&deviceExistKeys, sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMemset(deviceExistKeys, 0, sizeof(int));
	DIE(err != cudaSuccess, "cudaMemset");
	
	err = cudaMalloc(&(hashTable.map), size * sizeof(HashItem));
	DIE(err != cudaSuccess, "cudaMalloc");

	// Performance: Use CUDA Occupancy API to determine optimal launch parameters for the init kernel.
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blocksize, initHashT, 0, 0);
	gridSize = hashTable.capacity / blocksize;
	if (hashTable.capacity % blocksize)
		gridSize++;
	
	// Launch the initialization kernel.
	initHashT>>(hashTable);
	
	err = cudaDeviceSynchronize();
	DIE(err != cudaSuccess, "cudaDeviceSynchronize");
}

/**
 * @brief Destructor for GpuHashTable. Frees allocated device memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaError_t err;

	free(hostExistKeys);

	err = cudaFree(deviceExistKeys);
	DIE(err != cudaSuccess, "cudaFree");

	err = cudaFree(hashTable.map);
	DIE(err != cudaSuccess, "cudaFree");
}

/**
 * @brief Resizes the hash table to a new capacity and rehashes all elements.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t err;
	HashTable newHashTable;
	int minGridSize, blocksize, gridSize;

	newHashTable.capacity = numBucketsReshape;

	// Memory Allocation: Allocate a new, larger table on the device.
	err = cudaMalloc(&newHashTable.map, numBucketsReshape * sizeof(HashItem));
	DIE(err != cudaSuccess, "cudaFree");

	// Initialize the new table.
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blocksize, initHashT, 0, 0);
	gridSize = newHashTable.capacity / blocksize; // Correctly use new capacity.
	if (newHashTable.capacity % blocksize)
		gridSize++;
	initHashT>>(newHashTable);
	err = cudaDeviceSynchronize();
	DIE(err != cudaSuccess, "cudaDeviceSynchronize");

	// Rehash elements from the old table into the new one.
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blocksize, reshapeHT, 0, 0);
	gridSize = hashTable.capacity / blocksize;
	if (hashTable.capacity % blocksize)
		gridSize++;
	reshapeHT>>(hashTable, newHashTable);

	err = cudaDeviceSynchronize();
	DIE(err != cudaSuccess, "cudaDeviceSynchronize");
	
	// Free the old table memory and update the main structure.
	err = cudaFree(hashTable.map);
	DIE(err != cudaSuccess, "cudaFree");

	hashTable = newHashTable;
}

/**
 * @brief Inserts a batch of keys and values into the hash table.
 * @return True on success, false on failure.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	cudaError_t err;
	int *deviceKeys; 
	int *deviceValues;
	int minGridSize, blocksize, gridSize;

	// Memory Transfer: Allocate temporary device memory for the batch.
	err = cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");
	err = cudaMalloc(&deviceValues, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");

	// Memory Transfer: Copy batch from host to device.
	err = cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");
	err = cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	// Block Logic: Check the load factor and reshape if it exceeds the threshold (0.9f).
	float auxLF = ((float)(*hostExistKeys + numKeys) / hashTable.capacity);
	if (auxLF > 0.9f) {
		// Calculate new capacity to bring load factor to ~0.85f.
		int newCapacity = (int)(((float)*hostExistKeys + numKeys) / 0.85f);
		reshape(newCapacity);
	}
	
	// Determine optimal launch parameters and launch the insertion kernel.
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blocksize, insert, 0, 0);
	gridSize = numKeys / blocksize; // Based on numKeys to insert.
	if (numKeys % blocksize)
		gridSize++;
	insert>>(hashTable, deviceKeys, deviceValues, numKeys, deviceExistKeys);

	err = cudaDeviceSynchronize();
	DIE(err != cudaSuccess, "cudaDeviceSynchronize");
	
	// Update host-side count of elements.
	err = cudaMemcpy(hostExistKeys, deviceExistKeys, sizeof(int), cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, "cudaMemcpy");
	
	// Free temporary device memory.
	err = cudaFree(deviceKeys);
	DIE(err != cudaSuccess, "cudaFree");
	err = cudaFree(deviceValues);
	DIE(err != cudaSuccess, "cudaFree");

	return true;
}

/**
 * @brief Retrieves values for a batch of keys.
 * @return A host pointer to an array of results. Caller must free this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	cudaError_t err;

	int *values;
	int *deviceKeys; 
	int *deviceValues;
	int minGridSize, blocksize, gridSize;

	values = (int *)calloc(numKeys,sizeof(int));
	DIE(values == NULL , "malloc");

	// Memory Allocation and Transfer for keys.
	err = cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");
	err = cudaMalloc(&deviceValues, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");
	err = cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");
	
	// Determine launch parameters and launch the get kernel.
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blocksize, get, 0, 0);
	gridSize = numKeys / blocksize;
	if (numKeys % blocksize)
		gridSize++;
	get>>(hashTable, deviceKeys, deviceValues, numKeys);
	
	err = cudaDeviceSynchronize();
	DIE(err != cudaSuccess, "cudaDeviceSynchronize");

	// Memory Transfer: Copy results from device to host.
	err = cudaMemcpy(values, deviceValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, "cudaMemcpy");

	// Free temporary device memory.
	err = cudaFree(deviceKeys);
	DIE(err != cudaSuccess, "cudaFree");
	err = cudaFree(deviceValues);
	DIE(err != cudaSuccess, "cudaFree");

	return values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
	return ((float)*hostExistKeys / hashTable.capacity); 
}


// --- Convenience macros for external API ---
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

/*
 * @note The following line includes a .cpp file, which is highly unconventional.
 */
#include "test_map.cpp"

/*
 * @note The following block appears to be a re-definition of the header file content.
 */
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

// Value for an empty/invalid key slot.
#define	KEY_INVALID		0
#define BLOCKSIZE	1024

// Primes used for the multiplicative hash function.
#define PARAM_1_HASH 4247846927llu
#define PARAM_2_HASH 215777175787llu

// Macro for fatal error checking.
#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)
	
// A list of large prime numbers.
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



// Redundant hash function definitions.
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
 * @struct HashItem
 * @brief Represents a key-value pair in the hash table.
 */
typedef struct
{
	int key;
	int value;
}HashItem;

/**
 * @struct HashTable
 * @brief The main control structure passed to kernels, containing table capacity and a pointer to the data.
 */
typedef struct
{
	int capacity;
	HashItem *map;
}HashTable;


/**
 * @class GpuHashTable
 * @brief Host-side class for managing the GPU hash table.
 */
class GpuHashTable
{
	int *hostExistKeys;     // Host-side counter for existing keys.
	int *deviceExistKeys;   // Device-side counter for existing keys.
	HashTable hashTable;    // The main hash table structure.

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
