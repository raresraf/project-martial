/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA.
 *
 * This implementation utilizes an open addressing scheme with linear probing for
 * collision resolution. It features a simple multiplicative hashing function and
 * uses atomic operations to manage concurrent insertions. The resizing logic
 * is handled by creating a new, larger table and migrating existing elements
 * via a dedicated CUDA kernel.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


#define HASH_VAL1 13
#define HASH_VAL2 56564691976601587

// --- Kernel and Device Function Prototypes ---

/// Kernel to rehash elements from an old table to a new one.
__global__ void kernel_reshape(elem *dest, elem *source, unsigned int source_size, unsigned int dest_size);
/// Kernel to insert a batch of key-value pairs.
__global__ void insertFunc(int *keys, int *values, int numKeys, int dictSize, elem* dictElements);
/// Kernel to retrieve a batch of values corresponding to a set of keys.
__global__ void getFunc(int *keys, int *values, int numKeys, int dictSize, elem* dictElements);
/// Device-side hash function.
__device__ int myHash(int data, int limit);


/**
 * @brief Host-side constructor for the GpuHashTable class.
 * @param size The initial capacity of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	DIE(size < 0, "invalid size");

	cudaError_t rc;

	this->inserted = 0;
	this->size = size;

	// Allocate memory on the device for the hash table array.
	rc = cudaMalloc(&this->elements, size * sizeof(*elements));
	DIE(rc != cudaSuccess || this->elements == NULL, "cudaMalloc failed");

	// Initialize the allocated memory to zero.
	rc = cudaMemset(this->elements, 0, size * sizeof(*elements));
	DIE(rc != cudaSuccess, "cudaMemset failed");
}

/**
 * @brief Host-side destructor for the GpuHashTable class.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(this->elements);
}

/**
 * @brief Resizes the hash table on the GPU, preserving existing data.
 * @param numBucketsReshape The new capacity for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t rc;
	elem *aux;

	// Allocate a new, larger table on the device.
	rc = cudaMalloc(&aux, numBucketsReshape * sizeof(elem));
	DIE(rc != cudaSuccess || aux == NULL, "cudaMalloc failed");

	rc = cudaMemset(aux, 0, numBucketsReshape * sizeof(*aux));
	DIE(rc != cudaSuccess, "cudaMemset failed");

	// Calculate kernel launch parameters for the reshape operation.
	size_t num_blocks;

	num_blocks = this->size / block_size;
	if (this->size % block_size)
		num_blocks++;

	elem *old_elements = this->elements;
	
	// Launch the kernel to copy and rehash all elements from the old table to the new one.
	kernel_reshape>>(aux, this->elements,
		this->size, numBucketsReshape);
	cudaDeviceSynchronize();

	// Free the old table memory.
	rc = cudaFree(old_elements);
	DIE(rc != cudaSuccess, "free failed");

	// Update the host-side pointers and size to reflect the new table.
	this->elements = aux;
	this->size = numBucketsReshape;
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
	int *deviceValues;
	size_t num_blocks;
	cudaError_t rc;

	// Allocate temporary device memory for the batch.
	rc = cudaMalloc(&deviceKeys, numKeys * sizeof(*deviceKeys));
	DIE(rc != cudaSuccess, "cudaMalloc failed");
	if (deviceKeys == NULL)
		return false;

	rc = cudaMalloc(&deviceValues, numKeys * sizeof(*deviceValues));
	DIE(rc != cudaSuccess, "cudaMalloc failed");
	if (deviceValues == NULL)
		return false;

	// Copy keys and values from host to device.
	rc = cudaMemcpy(deviceKeys, keys, numKeys * sizeof(*deviceKeys), cudaMemcpyHostToDevice);
	DIE(rc != cudaSuccess, "cudaCopy failed");

	rc = cudaMemcpy(deviceValues, values, numKeys * sizeof(*deviceValues), cudaMemcpyHostToDevice);
	DIE(rc != cudaSuccess, "cudaCopy failed");

	// Update the element count and check if a reshape is needed.
	this->inserted += numKeys;
	if (loadFactor() > 0.8) {
		// Reshape the table to a new size based on the current element count.
		reshape(this->inserted / 0.8);
	}
	
	// Calculate kernel launch parameters.
	num_blocks = numKeys / block_size;
	if (numKeys % block_size)
		num_blocks++;
	
	int dictSize = this->size;

	// Launch the insertion kernel.
	insertFunc>>(deviceKeys, deviceValues, numKeys, dictSize, this->elements);
	cudaDeviceSynchronize();

	// Free temporary device buffers.
	rc = cudaFree(deviceKeys);
	DIE(rc != cudaSuccess, "free failed");
	
	rc = cudaFree(deviceValues);
	DIE(rc != cudaSuccess, "free failed");

	return true;
}

/**
 * @brief Retrieves a batch of values for a given set of keys.
 * @param keys Host pointer to an array of keys.
 * @param numKeys The number of keys to retrieve.
 * @return A host pointer to an array with the results. Caller is responsible for freeing this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *deviceKeys;
	cudaError_t rc;
	int *deviceValues;
	int *hostValues;
	size_t num_blocks;

	// Allocate temporary device buffers for keys and values.
	rc = cudaMalloc(&deviceKeys, numKeys * sizeof(*deviceKeys));
	DIE(rc != cudaSuccess, "malloc failed");
	if (deviceKeys == NULL)
		return NULL;

	rc = cudaMalloc(&deviceValues, numKeys * sizeof(*deviceValues));
	DIE(rc != cudaSuccess, "malloc failed");
	if (deviceValues == NULL)
		return NULL;

	// Allocate host memory for the final results.
	hostValues = (int *)malloc(numKeys * sizeof(*hostValues));
	if (hostValues == NULL)
		return NULL;

	rc = cudaMemset(deviceValues, 0, numKeys * sizeof(*deviceValues));
	DIE(rc != cudaSuccess, "cudaMemset failed");

	// Copy input keys from host to device.
	rc = cudaMemcpy(deviceKeys, keys, numKeys * sizeof(*deviceKeys), cudaMemcpyHostToDevice);
	DIE(rc != cudaSuccess, "copy failed");

	// Calculate kernel launch parameters.
	num_blocks = numKeys / block_size;
	if (numKeys % block_size)
		num_blocks++;

	int dictSize = this->size;

	// Launch the retrieval kernel.
	getFunc>>(deviceKeys, deviceValues, numKeys, dictSize, this->elements);
	cudaDeviceSynchronize();


	// Copy results from device back to host.
	rc = cudaMemcpy(hostValues, deviceValues, numKeys * sizeof(*hostValues), cudaMemcpyDeviceToHost);
	DIE(rc != cudaSuccess, "cudaCopy failed");

	// Free temporary device buffers.
	rc = cudaFree(deviceKeys);
	DIE(rc != cudaSuccess, "free failed");

	rc = cudaFree(deviceValues);
	DIE(rc != cudaSuccess, "free failed");

	return hostValues;

}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float (number of items / capacity).
 */
float GpuHashTable::loadFactor() {
	if (size == 0)
		return 0;
	return (float)inserted / size;
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param keys Pointer to device memory holding the keys to look up.
 * @param values Pointer to device memory where retrieved values will be stored.
 * @param numKeys The number of keys to look up.
 * @param dictSize The capacity of the hash table.
 * @param dictElems The hash table data.
 */
__global__ void getFunc(int *keys, int *values, int numKeys, int dictSize, elem *dictElems) {
	unsigned int id;
	int hash_pos;

	id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numKeys)
		return;

	// Step 1: Calculate the initial hash position.
	hash_pos = myHash(keys[id], dictSize);

	// Step 2: Linearly probe forward to find the key.
	for (int i = hash_pos; i < dictSize; ++i) {
		if (dictElems[i].key == keys[id]) {
			values[id] = dictElems[i].value;
			return;
		}
	}
	// Step 3: Handle wrap-around if not found in the forward pass.
	for (int i = 0; i < hash_pos; ++i) {
		if (dictElems[i].key == keys[id]) {
			values[id] = dictElems[i].value;
			return;
		}
	}
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs.
 * @param keys Pointer to device memory holding the keys.
 * @param values Pointer to device memory holding the values.
 * @param numKeys The number of pairs to insert.
 * @param dictSize The capacity of the hash table.
 * @param dictElems The hash table data.
 */
__global__ void insertFunc(int *keys, int *values, int numKeys, int dictSize, elem* dictElems) {
	unsigned int id;
	unsigned int old_key;
	int hash_pos;

	id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numKeys)
		return;
	
	// Step 1: Calculate the initial hash position.
	hash_pos = myHash(keys[id], dictSize);

	// Step 2: Linearly probe forward to find an empty slot or an existing key.
	for (int i = hash_pos; i < dictSize; ++i) {
		old_key = atomicCAS(&dictElems[i].key, 0, keys[id]);
		
		// If the slot was empty (old_key == 0) or already held the same key, the write is safe.
		if (old_key == 0 || old_key == keys[id]) {
			dictElems[i].value = values[id];
			return;
		}
	}

	// Step 3: Handle wrap-around if no slot was found.
	for (int i = 0; i < hash_pos; ++i) {
		old_key = atomicCAS(&dictElems[i].key, 0, keys[id]);
		if (old_key == 0 || old_key == keys[id]) {
			dictElems[i].value = values[id];
			return;
		}
	}
}

/**
 * @brief CUDA kernel to copy and rehash all elements from an old table to a new one.
 * @param dest The destination (new) hash table.
 * @param source The source (old) hash table.
 * @param source_size The size of the old table.
 * @param dest_size The size of the new table.
 */
__global__ void kernel_reshape(elem *dest, elem *source, unsigned int source_size, unsigned int dest_size){
	int hash_pos;
	unsigned int id;

	id = threadIdx.x + blockDim.x * blockIdx.x;

	if (id >= source_size)
		return;
	// Skip empty slots in the source table.
	if (source[id].key == 0)
		return;

	// Re-calculate the hash for the key in the context of the new table size.
	hash_pos = myHash(source[id].key, dest_size);

	// Linearly probe and insert into the new table.
	for (int i = hash_pos; i < dest_size; ++i) {
		// Use atomicCAS to safely claim an empty slot.
		if (atomicCAS(&dest[i].key, 0, source[id].key) == 0) {
			dest[i].value = source[id].value;
			return;
		}
	}
	// Handle wrap-around.
	for (int i = 0; i < hash_pos; ++i) {
		if (atomicCAS(&dest[i].key, 0, source[id].key) == 0) {
			dest[i].value = source[id].value;
			return;
		}
	}
}

/**
 * @brief Computes a hash value for a given key on the GPU.
 * @param data The input key to hash.
 * @param limit The size of the hash table, used for the final modulo operation.
 * @return The computed hash index.
 */
__device__ int myHash(int data, int limit) {
	// Simple multiplicative hashing.
	return ((long)abs(data) * HASH_VAL1) % HASH_VAL2 % limit;
}


// Macros for a simplified testing interface.
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
// Header guard and definitions.
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0

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

const size_t block_size = 256;

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



// Alternative hash functions.
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
typedef struct {
	unsigned int key;
	unsigned int value;
} elem;

// The C++ class that provides a high-level API for the GPU hash table.
class GpuHashTable
{

	public:
		unsigned int inserted;
		unsigned int size;
		elem *elements;

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
