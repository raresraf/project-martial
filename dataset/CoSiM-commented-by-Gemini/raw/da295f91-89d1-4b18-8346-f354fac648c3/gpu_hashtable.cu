
/**
 * @file gpu_hashtable.cu
 * @brief This file contains a CUDA implementation of a GPU-based hash table.
 * @raw/da295f91-89d1-4b18-8346-f354fac648c3/gpu_hashtable.cu
 *
 * This implementation uses open addressing with linear probing for collision resolution.
 * It stores keys and values in separate arrays. The main operations (insert, get, reshape)
 * are parallelized on the GPU using CUDA kernels.
 *
 * Algorithm: Open Addressing with Linear Probing
 * Time Complexity:
 *  - Insertion: Average O(1), Worst O(N)
 *  - Retrieval: Average O(1), Worst O(N)
 *  - Reshape: O(N)
 * Space Complexity: O(N)
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief Constructor for the GpuHashTable class.
 * @param size The initial capacity of the hash table.
 *
 * Allocates two separate arrays on the GPU for keys and values and initializes them.
 */
GpuHashTable::GpuHashTable(int size) {
	
	totalSize = size;
	currentSize = 0;
	cudaMalloc(&keysVector, size * sizeof(int));
	cudaMalloc(&valuesVector, size * sizeof(int));
	// Initialize all keys and values to an invalid state.
	cudaMemset(keysVector, KEY_INVALID, size * sizeof(int));
	cudaMemset(valuesVector, KEY_INVALID, size * sizeof(int));	
}

/**
 * @brief Destructor for the GpuHashTable class.
 *
 * Frees the GPU memory allocated for the keys and values vectors.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(keysVector);
	cudaFree(valuesVector);
}


/**
 * @brief CUDA kernel for resizing and re-hashing the hash table.
 * @param oldKeys Pointer to the old keys array on the device.
 * @param oldValues Pointer to the old values array on the device.
 * @param oldSize The size of the old hash table.
 * @param newKeys Pointer to the new keys array on the device.
 * @param newValues Pointer to the new values array on the device.
 * @param newSize The size of the new hash table.
 *
 * Each thread processes one entry from the old hash table. If the entry is valid,
 * it re-hashes the key and inserts the key-value pair into the new table using
 * linear probing and atomic operations for thread safety.
 */
__global__ void reshape_func(int *oldKeys, int *oldValues, int oldSize, int *newKeys, int* newValues, int newSize)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < oldSize)
    {
    	// Skip empty slots in the old table.
    	if(oldKeys[i] == KEY_INVALID)
    		return;
		
		// Re-hash the key for the new table size.
    	int index = hash_function(oldKeys[i], newSize);
		// Use atomicCAS to find an empty slot in the new table via linear probing.
    	int result = atomicCAS(&newKeys[index], KEY_INVALID, oldKeys[i]);
    	
    	while(result != 0)
    	{
    		index = (index + 1) % newSize;
    		result = atomicCAS(&newKeys[index], KEY_INVALID, oldKeys[i]);
    	}
    	
		// Place the value in the corresponding slot.
    	newValues[index] = oldValues[i];
    }
}

/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The new capacity of the hash table.
 *
 * This host function orchestrates the resize operation. It allocates new key and value
 * arrays on the GPU, then launches the `reshape_func` kernel to transfer and re-hash
 * the elements.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	
	int *oldKeys = keysVector;
	int *oldValues = valuesVector;
	int oldSize = totalSize;
	cudaMalloc(&keysVector, numBucketsReshape * sizeof(int));
	cudaMalloc(&valuesVector, numBucketsReshape * sizeof(int));
	cudaMemset(keysVector, KEY_INVALID, numBucketsReshape * sizeof(int));
	cudaMemset(valuesVector, KEY_INVALID, numBucketsReshape * sizeof(int));
	totalSize = numBucketsReshape;
	const size_t block_size = 8;
    size_t blocks_no = oldSize / block_size;
    if (oldSize % block_size)
    	blocks_no++;
    
    reshape_func>>(oldKeys, oldValues, oldSize, keysVector, valuesVector, totalSize);
    cudaDeviceSynchronize();
	cudaFree(oldKeys);
	cudaFree(oldValues);
}


/**
 * @brief CUDA kernel for inserting a batch of key-value pairs.
 * @param vramKeys Pointer to the batch of keys to insert (on the device).
 * @param vramValues Pointer to the batch of values to insert (on the device).
 * @param numKeys The number of key-value pairs in the batch.
 * @param keysVector The hash table's key array.
 * @param valuesVector The hash table's value array.
 * @param totalSize The total capacity of the hash table.
 *
 * Each thread handles one key-value pair, using linear probing and `atomicCAS`
 * to find an empty slot or update an existing key.
 */
__global__ void insert_func(int *vramKeys, int *vramValues, int numKeys, int *keysVector, int *valuesVector, int totalSize)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < numKeys)
	{
		int index = hash_function(vramKeys[i], totalSize);
		// Atomically try to insert the new key into an empty slot.
		int result = atomicCAS(&keysVector[index], KEY_INVALID, vramKeys[i]);
		
		// If the slot was not empty, probe linearly.
		while(result != 0)
		{
			// The original value is restored before moving to the next slot.
			keysVector[index] = result;
			
			// If the key already exists, update its value and return.
			if(keysVector[index] == vramKeys[i])
			{
				valuesVector[index] = vramValues[i];
				return;
			}
			index = (index + 1) % totalSize;
			result = atomicCAS(&keysVector[index], KEY_INVALID, vramKeys[i]);
		}
		// Found an empty slot, so place the value.
		valuesVector[index] = vramValues[i];
	}
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys Pointer to the host array of keys.
 * @param values Pointer to the host array of values.
 * @param numKeys The number of key-value pairs to insert.
 * @return `true` if the insertion process was initiated successfully.
 *
 * This function handles host-to-device memory transfers, triggers a `reshape`
 * if the load factor is too high, and launches the `insert_func` kernel.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	currentSize += numKeys;
	
	// If the load factor will exceed 1.0, reshape the table.
	if(currentSize >= totalSize)
		reshape(currentSize * 1.1 + 1);
	int *vramKeys = 0;
	int *vramValues = 0;
	cudaMalloc((void **) &vramKeys, numKeys * sizeof(int));


	cudaMalloc((void **) &vramValues, numKeys * sizeof(int));
	cudaMemcpy(vramKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(vramValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	const size_t block_size = 8;
	size_t blocks_no = numKeys / block_size;
	if (numKeys % block_size) 
  		blocks_no++;
  	
	insert_func>>(vramKeys, vramValues, numKeys, keysVector, valuesVector, totalSize);
	cudaDeviceSynchronize();
	return true;
}


/**
 * @brief CUDA kernel for searching for a batch of keys.
 * @param vramKeys Pointer to the batch of keys to search for (on the device).
 * @param resultValues Pointer to an array where the results will be stored.
 * @param numKeys The number of keys in the batch.
 * @param keysVector The hash table's key array.
 * @param valuesVector The hash table's value array.
 * @param totalSize The total capacity of the hash table.
 *
 * Each thread searches for one key using linear probing.
 * @warning This kernel has a potential issue: if a key is not present in the table,
 * the `while` loop will become an infinite loop. A check for reaching the original
 * hash index or an empty slot is needed to handle non-existent keys.
 */
__global__ void search_func(int *vramKeys, int *resultValues, int numKeys, int *keysVector, int *valuesVector, int totalSize)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < numKeys)
	{
		int index = hash_function(vramKeys[i], totalSize);
		
		// Probe linearly until the key is found.
		while(keysVector[index] != vramKeys[i])
		{
			index = (index + 1) % totalSize;
		}
		// Store the found value in the results array.
		resultValues[i] = valuesVector[index];
	}
}

/**
 * @brief Retrieves the values for a batch of keys.
 * @param keys Pointer to the host array of keys to search for.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to a host array containing the retrieved values.
 *
 * This function handles memory transfers and launches the `search_func` kernel.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *vramKeys = 0;
	int *resultValues = 0;
	cudaMalloc((void **) &vramKeys, numKeys * sizeof(int));


	cudaMalloc((void **) &resultValues, numKeys * sizeof(int));
	cudaMemcpy(vramKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	const size_t block_size = 8;
	size_t blocks_no = numKeys / block_size;
	if (numKeys % block_size) 
  		blocks_no++;
	search_func>>(vramKeys, resultValues, numKeys, keysVector, valuesVector, totalSize);
	cudaDeviceSynchronize();
	int *cpu_result = (int *) malloc(numKeys * sizeof(int));
	// Copy results from device to host.
	cudaMemcpy(cpu_result, resultValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	return cpu_result;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float.
 */
float GpuHashTable::loadFactor() {
	return ((float) currentSize) / totalSize;
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
	
// A list of prime numbers, likely for use in more complex hashing schemes.
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



// Alternative hash functions, not used in the current implementation.
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
 * @brief Computes the hash of a given key.
 * @param data The key to hash.
 * @param limit The size of the hash table, used to wrap the hash value.
 * @return The hash value for the given key.
 * @note This is a multiplicative hash function.
 */
__host__ __device__ int hash_function(int data, int limit)
{
	return ((long)abs(data) * 17399181177241llu) % 132745199llu % limit;
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

		
		int totalSize;
		int currentSize;
		int *keysVector;
		int *valuesVector;
};

#endif
