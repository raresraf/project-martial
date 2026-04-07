
/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA with double hashing.
 * @details This file defines a hash table that resides in GPU memory and is manipulated
 * through CUDA kernels. It supports batch insertion and retrieval operations. The collision
 * resolution strategy is double hashing, and thread safety is ensured using atomic operations.
 * The hash table can also be dynamically resized.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief Handles CUDA errors by printing an error message and exiting.
 * @param err The CUDA error code.
 * @param file The name of the file where the error occurred.
 * @param line The line number where the error occurred.
 */
static void HandleError(cudaError_t err, const char *file, int line ) {
	if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << " in " 
			<< file << " at line " << line << endl;
		exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


/**
 * @brief Constructs a GpuHashTable object.
 * @param size The initial desired capacity of the hash table.
 * @details Allocates memory on the GPU for the hash table. The actual capacity is adjusted
 * to be a prime number to improve hash distribution. It initializes the table to zero
 * and selects random prime numbers for the double hashing function.
 */
GpuHashTable::GpuHashTable(int size)
{
	int flag = 0, i;
	int primeSize = sizeof(primeList) / sizeof(*primeList);

	srand(time(NULL));

	HANDLE_ERROR( cudaMallocManaged((void **) &hashTable, sizeof(struct hashT)) );

	
	// Find the next prime number greater than or equal to size for the capacity.
	while (1) {
		for (i = 2; i < size; i++)
			if (size % i == 0) {
				flag = 1;
				break;
			}

		if (!flag)
			break;
		flag = 0;
		size++;
	}
	if (size == 0)
		size = 1;

	hashTable->capacity = size;
	hashTable->size = 0;
	hashTable->subHash = (size % 2 == 0) ? size / 2 : (size + 1) / 2;
	HANDLE_ERROR( cudaMalloc((void **) &(hashTable->data),
		size * sizeof(struct node)) );
	HANDLE_ERROR( cudaMemset(hashTable->data, KEY_INVALID,
		size * sizeof(struct node)) );
	// Select random prime numbers for the hash functions.
	hashTable->firstPrime = primeList[rand() % primeSize];
	hashTable->secondPrime = primeList[rand() % primeSize];
}


/**
 * @brief Destroys the GpuHashTable object.
 * @details Frees the GPU memory allocated for the hash table data and the table structure itself.
 */
GpuHashTable::~GpuHashTable()
{
	HANDLE_ERROR( cudaFree(hashTable->data) );
	HANDLE_ERROR( cudaFree(hashTable) );
}


/**
 * @brief CUDA kernel to resize the hash table by re-hashing all elements.
 * @param currHash The new, larger hash table.
 * @param prevHash The old hash table.
 * @details Each thread takes an element from the old table and inserts it into the new table
 * using a double hashing scheme.
 */
__global__ void reshapeKernel(HashTable currHash, HashTable prevHash)
{
	uint32_t prevKey = KEY_INVALID, currKey, value, hashInd, hashVal;
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id capacity) {


		currKey = prevHash->data[id].key;
		if (currKey == KEY_INVALID)
			return;
		value = prevHash->data[id].value;
		// Calculate the new hash position in the new table.
		hashInd = h1(currKey, currHash->capacity, currHash->firstPrime,
			currHash->secondPrime);
		hashVal = h2(currKey, currHash->subHash);

		// Probe for an empty slot using double hashing.
		do {
			hashInd = (hashInd + hashVal) % currHash->capacity;
			prevKey = atomicCAS(&(currHash->data[hashInd].key),
				KEY_INVALID, currKey);
		} while (prevKey != KEY_INVALID);

		currHash->data[hashInd].value = value;
	}
}


/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The number of new buckets to add.
 * @details Allocates a new, larger hash table on the GPU and launches a kernel to re-hash
 * all elements from the old table into the new one. The new size is chosen to be a prime number.
 */
void GpuHashTable::reshape(int numBucketsReshape)
{
	HashTable newHash;
	int flag = 0, i;
	unsigned int numBlocks;
	int primeSize = sizeof(primeList) / sizeof(*primeList);
	unsigned int sizeForReshape = (hashTable->size + numBucketsReshape
		+ DEFAULTLOAD - 1) / DEFAULTLOAD;

	HANDLE_ERROR( cudaMallocManaged((void **) &newHash, sizeof(struct hashT)) );

	
	// Find the next prime number for the new capacity.
	while (1) {
		for (i = 2; i < sizeForReshape; i++)
			if (sizeForReshape % i == 0) {
				flag = 1;
				break;
			}

		if (!flag)
			break;
		flag = 0;
		sizeForReshape++;
	}
	if (sizeForReshape == 0)
		sizeForReshape = 1;

	newHash->capacity = sizeForReshape;
	newHash->size = hashTable->size;
	newHash->subHash = (sizeForReshape % 2 == 0) ?
		sizeForReshape / 2 : (sizeForReshape + 1) / 2;
	HANDLE_ERROR( cudaMalloc((void **) &(newHash->data),
		sizeForReshape * sizeof(struct node)) );
	HANDLE_ERROR( cudaMemset(newHash->data, KEY_INVALID,
		sizeForReshape * sizeof(struct node)) );
	newHash->firstPrime = primeList[rand() % primeSize];
	newHash->secondPrime = primeList[rand() % primeSize];

	numBlocks = (hashTable->capacity + BLOCKSIZE - 1) / BLOCKSIZE;
	reshapeKernel>>(newHash, hashTable);

	HANDLE_ERROR( cudaDeviceSynchronize() );

	HANDLE_ERROR( cudaFree(hashTable->data) );
	HANDLE_ERROR( cudaFree(hashTable) );
	
	hashTable = newHash;
}


/**
 * @brief CUDA kernel to insert a batch of key-value pairs using double hashing.
 * @param hashTable The hash table structure.
 * @param keys Pointer to an array of keys to insert.
 * @param values Pointer to an array of values to insert.
 * @param numKeys The number of keys to insert.
 * @details Each thread handles one key-value pair. It computes two hash values and uses them
 * for probing. `atomicCAS` is used to ensure thread-safe insertion.
 */
__global__ void insertKernel(HashTable hashTable, int *keys, int *values, int numKeys)
{
	uint32_t prevKey = KEY_INVALID, currKey, value, hashInd, hashVal;
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < numKeys) {
		currKey = keys[id];
		value = values[id];
		if (currKey <= KEY_INVALID || value <= KEY_INVALID)
			return;
		// Compute primary and secondary hashes.
		hashInd = h1(currKey, hashTable->capacity, hashTable->firstPrime,
			hashTable->secondPrime);
		hashVal = h2(currKey, hashTable->subHash);

		// Probe for an empty slot using double hashing.
		do {
			hashInd = (hashInd + hashVal) % hashTable->capacity;
			prevKey = atomicCAS(&(hashTable->data[hashInd].key),
				KEY_INVALID, currKey);
		} while (prevKey != currKey && prevKey != KEY_INVALID);

		hashTable->data[hashInd].value = value;
		
		// Atomically increment the size of the hash table if a new element was inserted.
		if (prevKey == KEY_INVALID)
			atomicAdd(&(hashTable->size), 1);
	}
}


/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys An array of keys to insert.
 * @param values An array of values to insert.
 * @param numKeys The number of keys and values in the batch.
 * @return True if the insertion was successful.
 * @details If the load factor exceeds a threshold, the table is resized. Keys and values
 * are copied to the GPU, and the insertion kernel is launched.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys)
{
	int *devKeys, *devValues;
	float currLoad = (float)(hashTable->size + numKeys) / hashTable->capacity;
	unsigned int numBlocks, size = numKeys * sizeof(int);

	if (currLoad > MAXIMUMLOAD)
		reshape((currLoad - MAXIMUMLOAD) * hashTable->capacity);

	HANDLE_ERROR( cudaMalloc((void **) &devKeys, size) );
	HANDLE_ERROR( cudaMalloc((void **) &devValues, size) );

	HANDLE_ERROR( cudaMemcpy(devKeys, keys, size,
		cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(devValues, values, size,


		cudaMemcpyHostToDevice) );

	numBlocks = (numKeys + BLOCKSIZE - 1) / BLOCKSIZE;
	insertKernel>>(hashTable, devKeys,
		devValues, numKeys);

	HANDLE_ERROR( cudaDeviceSynchronize() );

	HANDLE_ERROR( cudaFree(devKeys) );
	HANDLE_ERROR( cudaFree(devValues) );

	return true;
}


/**
 * @brief CUDA kernel to retrieve values for a batch of keys using double hashing.
 * @param hashTable The hash table structure.
 * @param keys Pointer to an array of keys to look up.
 * @param values Pointer to an array to store the retrieved values.
 * @param numKeys The number of keys to look up.
 * @details Each thread handles one key. It computes the primary and secondary hashes and
 * uses them to probe the table until the key is found.
 */
__global__ void getKernel(HashTable hashTable, int *keys, int *values, int numKeys)
{
	uint32_t key, hashInd, hashVal;
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < numKeys) {
		key = keys[id];
		// Compute primary and secondary hashes.
		hashInd = h1(key, hashTable->capacity, hashTable->firstPrime,
			hashTable->secondPrime);
		hashVal = h2(key, hashTable->subHash);

		// Probe for the key using double hashing.
		do {
			hashInd = (hashInd + hashVal) % hashTable->capacity;
		} while (hashTable->data[hashInd].key != key);

		values[id] = hashTable->data[hashInd].value;
	}
}


/**
 * @brief Retrieves values for a batch of keys.
 * @param keys An array of keys to look up.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to an array in managed GPU memory containing the retrieved values.
 * @details This function copies the keys to the GPU and launches a kernel to retrieve the values.
 */
int *GpuHashTable::getBatch(int* keys, int numKeys)
{
	unsigned int numBlocks, size = numKeys * sizeof(int);
	int *devKeys, *devValues;

	HANDLE_ERROR( cudaMalloc((void **) &devKeys, size) );
	HANDLE_ERROR( cudaMallocManaged((void **) &devValues, size) );

	HANDLE_ERROR( cudaMemcpy(devKeys, keys, size,
		cudaMemcpyHostToDevice) );



	numBlocks = (numKeys + BLOCKSIZE - 1) / BLOCKSIZE;
	getKernel>>(hashTable, devKeys,
		devValues, numKeys);

	HANDLE_ERROR( cudaDeviceSynchronize() );

	HANDLE_ERROR( cudaFree(devKeys) );

	return devValues;
}


/**
 * @brief Calculates the load factor of the hash table.
 * @return The load factor (ratio of occupied slots to total capacity).
 */
float GpuHashTable::loadFactor()
{
	float currLoad = (float)hashTable->size / hashTable->capacity;
	return currLoad;
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
#define	BLOCKSIZE		1024
#define	DEFAULTLOAD		0.8
#define	MAXIMUMLOAD		0.95


typedef struct node {
	uint32_t key;
	uint32_t value;
} Node;


typedef struct hashT {
	uint32_t capacity;
	uint32_t size;
	uint32_t subHash;
	size_t firstPrime;
	size_t secondPrime;
	Node *data;
} *HashTable;

	
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


int hash1(int data, int limit)
{
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit)
{
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit)
{
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}


/**
 * @brief Computes the primary hash for double hashing.
 * @param data The key to hash.
 * @param limit The capacity of the hash table.
 * @param firstPrime A large prime number for the hash function.
 * @param secondPrime Another large prime number for the hash function.
 * @return The primary hash value.
 */
__device__ uint32_t h1(uint32_t data, uint32_t limit, size_t firstPrime, size_t secondPrime)
{
	return ((long)data * firstPrime) % secondPrime % limit;
}
/**
 * @brief Computes the secondary hash (step size) for double hashing.
 * @param data The key to hash.
 * @param subHash A value derived from the table capacity, typically a prime number smaller than the capacity.
 * @return The step size for probing.
 */
__device__ uint32_t h2(uint32_t data, uint32_t subHash)
{
	return subHash - ((long)data % subHash);
}


class GpuHashTable
{
	HashTable hashTable;

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
