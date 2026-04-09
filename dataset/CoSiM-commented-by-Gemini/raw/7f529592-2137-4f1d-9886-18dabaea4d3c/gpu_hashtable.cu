/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA.
 *
 * This version of the hash table uses open addressing with a linear probing
 * collision resolution strategy. It relies on global variables for its state
 * (e.g., `hashTable`, `hashSize`), which means only a single instance of this
 * hash table can exist within the application context. Operations like insertion
 * and lookup are performed in parallel on the GPU.
 *
 * @note The design with global state is not scalable to multiple hash table
 * instances. The file also includes a C-style API wrapper and redundant
 * definitions at the end.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @struct myPair
 * @brief A simple key-value pair structure for storing entries in the hash table.
 */
typedef struct
{
    uint32_t key;
    uint32_t value;
} myPair;

// Global variable to track the number of elements in the hash table.
// Note: This is updated non-atomically on the host side.
int elementsInHash;

// Global variable for the current capacity of the hash table.
int hashSize;

// Global pointer to the hash table storage on the GPU device.
myPair* hashTable;

/**
 * @brief Computes the hash for a given key.
 * @param data The integer key to hash.
 * @param limit The size of the hash table, used for the final modulo.
 * @return The calculated hash index.
 *
 * This is a device function, callable from within CUDA kernels. It uses
 * multiplicative hashing with large prime numbers.
 */
__device__ int hashFunction(int data, int limit) {
	return ((long)abs(data) * 41812097llu) % 142534092204280003llu % limit;
}

/**
 * @brief Constructs the global GpuHashTable.
 * @param size The initial capacity of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	
	elementsInHash = 0;
	hashSize = size;
	cudaMalloc((void **) &hashTable, size*sizeof(myPair));
	if(hashTable != NULL)
		cudaMemset(hashTable, 0, size * sizeof(myPair));
	else
		return;
}



/**
 * @brief Destroys the global GpuHashTable, freeing its device memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashTable);
}

/**
 * @brief CUDA kernel to re-hash all elements from an old table into a new, larger one.
 * @param newHashTable Pointer to the new hash table's storage (device memory).
 * @param hashTable Pointer to the old hash table's storage (device memory).
 * @param oldSize The capacity of the old table.
 * @param hashSize The capacity of the new table.
 *
 * Each thread takes an element from the old table and re-inserts it into the
 * new table using linear probing and atomic operations.
 */
__global__ void kernel_reshape(myPair* newHashTable, myPair* hashTable, int oldSize, int hashSize) {
 	
 	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
 	
 	// Each thread processes one slot from the old table.
 	if (i >= oldSize || hashTable[i].key == 0) return;
 	
 	int startIndex = hashFunction(hashTable[i].key, hashSize);

	// Linear probing implementation: search from the start index to the end.
 	for (int index = startIndex; index < hashSize; index++){
  		
		// Attempt to atomically claim an empty slot (where key is 0).
  		int prev = atomicCAS(&newHashTable[index].key, 0, hashTable[i].key);
 		if (prev == 0){
 			newHashTable[index].key = hashTable[i].key;
 			newHashTable[index].value = hashTable[i].value;
 			return;
 		}
 	}
	// If the end is reached, wrap around and search from the beginning to the start index.
 	for (int index = 0; index < startIndex; index++){
  		int prev = atomicCAS(&newHashTable[index].key, 0, hashTable[i].key);		
 		if (prev == 0){
 			newHashTable[index].key = hashTable[i].key;
 			newHashTable[index].value = hashTable[i].value;
 			return;
 		}
 	}
}


void GpuHashTable::reshape(int numBucketsReshape) {
	int oldSize = hashSize;
	hashSize = numBucketsReshape;
	myPair *newHashTable;
	
	// Allocate a new table with a 20% buffer.
	cudaMalloc((void **) &newHashTable, (numBucketsReshape + ((int) 0.2*numBucketsReshape))*sizeof(myPair));
	if(hashTable != NULL)
		cudaMemset(newHashTable, 0, (numBucketsReshape + ((int) 0.2*numBucketsReshape))*sizeof(myPair));
	else
		return;

	const size_t block_size = 256;
    size_t blocks_no = hashSize / block_size;

	if (hashSize % block_size) 
		++blocks_no;
	
	kernel_reshape>>(newHashTable, hashTable, oldSize, hashSize);
	cudaDeviceSynchronize();
	
	cudaFree(hashTable);
	
	// Update the global pointer to the new table.
	hashTable = newHashTable;
}

/**
 * @brief CUDA kernel for inserting a batch of key-value pairs into the hash table.
 * @param hashTable Pointer to the hash table's storage (device memory).
 * @param keysArray Array of keys to insert (device memory).
 * @param valuesArray Array of values to insert (device memory).
 * @param numKeys The number of pairs to process.
 * @param hashSize The capacity of the hash table.
 * @param elementsInHash The current number of elements (passed by value, for information only).
 *
 * Each thread handles one key-value pair, using linear probing and `atomicCAS`
 * to find a slot and insert the data in a thread-safe manner.
 */
__global__ void kernel_insert(myPair* hashTable, int* keysArray, int* valuesArray, int numKeys, 
											int hashSize, int elementsInHash) {
 	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
 	if (i >= numKeys) return;

 	int startIndex = hashFunction(keysArray[i], hashSize);

	// Linear probing: search from the start index to the end.
 	for(int index = startIndex; index < hashSize; index++){
		// Atomically try to claim an empty slot or find an existing key.
  		int prev = atomicCAS(&hashTable[index].key, 0, keysArray[i]);		
 		if (prev == 0){ // Slot was empty, successful insertion.
 			hashTable[index].key = keysArray[i];
 			hashTable[index].value = valuesArray[i];
 			return;
 		}
 		
 		if (prev == keysArray[i]){ // Key already existed, update value.
 			hashTable[index].value = valuesArray[i];
 			return;
 		}
 	}
	// Wrap around and continue probing from the beginning.
 	for (int index = 0; index < startIndex; index++){
  		int prev = atomicCAS(&hashTable[index].key, 0, keysArray[i]);		
 		if (prev == 0){ // Slot was empty, successful insertion.
 			hashTable[index].key = keysArray[i];
 			hashTable[index].value = valuesArray[i];
 			return;
 		}
 		if(prev == keysArray[i]){ // Key already existed, update value.
 			hashTable[index].value = valuesArray[i];
 			return;
 		}
 	}
}


/**
 * @brief Inserts a batch of key-value pairs.
 * @param keys Pointer to a host array of keys.
 * @param values Pointer to a host array of values.
 * @param numKeys The number of pairs to insert.
 * @return True on success, false on memory allocation failure.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *keysArray;


	int *valuesArray;
	
	cudaMalloc((void **) &keysArray, numKeys*sizeof(int));
	cudaMalloc((void **) &valuesArray, numKeys*sizeof(int));
	if(keysArray == NULL || valuesArray == NULL)
		return false;
	
	cudaMemcpy(keysArray, keys, numKeys*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(valuesArray, values, numKeys*sizeof(int), cudaMemcpyHostToDevice);

	
	// If load factor will exceed 0.9, reshape to a size that makes the new load factor 0.8.
	if((elementsInHash + numKeys)/hashSize > 0.9)
		reshape((int) (elementsInHash + numKeys) / 0.8);
	
	const size_t block_size = 256;
    size_t blocks_no = numKeys / block_size;

	if (numKeys % block_size) 
		++blocks_no;

	kernel_insert>>(hashTable, keysArray, valuesArray, numKeys, hashSize, elementsInHash);
	cudaDeviceSynchronize();

	// Note: This update is not thread-safe if multiple host threads call insertBatch.
	elementsInHash += numKeys;
	cudaFree(keysArray);
	cudaFree(valuesArray);

	return true;
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param hashTable Pointer to the hash table storage (device memory).
 * @param keys Array of keys to look up (device memory).
 * @param values Array where found values will be stored (device memory).
 * @param numKeys The number of keys to look up.
 * @param hashSize The capacity of the hash table.
 *
 * Each thread searches for one key using the same linear probing logic as insertion.
 * This is a read-only operation and does not require atomic instructions.
 */
__global__ void kernel_get(myPair* hashTable, int* keys, int* values, int numKeys, int hashSize) {
 	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= numKeys) return;

	int startIndex = hashFunction(keys[i], hashSize);
	// Linear probing: search from the start index to the end.
  	for(int index = startIndex; index < hashSize; index++){
 		if (hashTable[index].key == keys[i]){
 			values[i] = hashTable[index].value;
 			return;
 		}
	}
	// Wrap around and continue probing from the beginning.
 	for (int index = 0; index < startIndex; index++){
  		if (hashTable[index].key == keys[i]){
 			values[i] = hashTable[index].value;
 			return;
 		}
  	}
}

/**
 * @brief Retrieves the values for a batch of keys.
 * @param keys Pointer to a host array of keys.
 * @param numKeys The number of keys to look up.
 * @return A host pointer to an array of found values. The caller must free this memory.
 *         Returns 0 on failure.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *storeKeys;
	int *deviceValues;
	int *hostValues;



	hostValues = (int*) malloc (numKeys * sizeof(int));
	cudaMalloc((void **) &storeKeys, numKeys * sizeof(int));
	cudaMalloc((void **) &deviceValues, numKeys * sizeof(int));

	if (storeKeys == NULL || deviceValues == NULL)
		return 0;
	cudaMemset(deviceValues, 0, numKeys * sizeof(int));
	cudaMemcpy(storeKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	const size_t block_size = 256;
    size_t blocks_no = numKeys / block_size;

	if (numKeys % block_size) 
		++blocks_no;

	kernel_get>>(hashTable, storeKeys, deviceValues, numKeys, hashSize);
	
	cudaDeviceSynchronize();
	cudaMemcpy(hostValues, deviceValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(storeKeys);
	cudaFree(deviceValues);

	return hostValues;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float (elements / capacity).
 */
float GpuHashTable::loadFactor() {
	if (hashSize == 0)
		return 0.f; 
	return (float)elementsInHash/(float)hashSize;
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

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
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
		
};

#endif

