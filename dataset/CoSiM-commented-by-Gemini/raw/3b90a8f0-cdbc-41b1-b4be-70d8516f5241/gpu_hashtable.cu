/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA.
 *
 * This implementation uses an open addressing scheme with linear probing. It features
 * a MurmurHash-style hashing function for good key distribution. A notable feature
 * is its ability to distinguish between new key insertions and updates to existing
 * keys using an atomic counter.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief Host-side constructor for the GpuHashTable class.
 * @param size The initial capacity of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	
	// Allocate memory on the device for the hash table array and initialize it to zero.
	cudaError_t __cudaCalloc_err = cudaMalloc(&this->hashtable, size * sizeof(gpu_hashtable));
	if (__cudaCalloc_err == cudaSuccess) 
		cudaMemset(this->hashtable, 0, size * sizeof(gpu_hashtable));


	// Allocate a single integer on the device to be used as an atomic counter for tracking key updates.
	__cudaCalloc_err = cudaMalloc(&this->update_device, sizeof(int));
	DIE(__cudaCalloc_err != cudaSuccess, "cudaMalloc error");

	// Allocate host-side memory to track the table's capacity and current number of elements.
	this->maxim_host = (int *)malloc(sizeof(int));
	DIE(this->maxim_host == NULL, "malloc error");
	this->current_host = (int *)malloc(sizeof(int));
	DIE(this->current_host == NULL, "malloc error");

	*this->maxim_host = size;
	*this->current_host = 0;

	// Initialize the device-side update counter to 0.
	cudaMemcpy(this->update_device, this->current_host, sizeof(int), cudaMemcpyHostToDevice);
}

/**
 * @brief Host-side destructor for the GpuHashTable class.
 *
 * Frees all allocated device and host memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(this->hashtable);
	cudaFree(this->update_device);
	free(this->maxim_host);
	free(this->current_host);
}



/**
 * @brief Computes a hash value for a given key on the GPU.
 * @param key The input key to hash.
 * @param maxim The size of the hash table, used for the final modulo operation.
 * @return The computed hash index.
 *
 * This function uses a series of bitwise operations (XORs, shifts) and
 * multiplications with magic constants, characteristic of MurmurHash variants,
 * to produce a well-distributed hash value.
 */
__device__ int hashFunction(int key, int maxim){
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;


    return key % maxim;
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param hashtable Pointer to the device memory representing the hash table.
 * @param keys Pointer to device memory holding the keys to insert.
 * @param values Pointer to device memory holding the values to insert.
 * @param numKeys The number of pairs in the batch.
 * @param maxim The capacity of the hash table.
 * @param update A device pointer to an atomic counter for tracking updates to existing keys.
 */
__global__ void gpu_hashtable_insert_kernel(gpu_hashtable *hashtable, int *keys, int *values, int numKeys, int maxim, int *update)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (threadId < numKeys)
    {
		int i, key = keys[threadId], value =  values[threadId];
		
		// Step 1: Calculate the initial hash position.
		int slot = hashFunction(key, maxim);
		
		// Step 2: Linearly probe forward from the hash position.
		for(i = slot; i < maxim; i++)
		{
			// Atomically attempt to claim an empty slot (KEY_INVALID) or find the existing key.
			int prev = atomicCAS(&hashtable[i].key, KEY_INVALID, key);
			
			// Pre-condition: If the slot was empty, we have inserted a new key.
			if (prev == KEY_INVALID) {
				hashtable[i].value = value;
				return;
			}
			// Pre-condition: If the slot already contained our key, this is an update.
			else if (prev == key) {
				// Atomically increment the update counter.
				atomicAdd(update, 1);
				hashtable[i].value = value;
				return;
			}
		}
		// Step 3: Handle wrap-around if no slot was found.
		for(i = 0; i < slot; i++){
			int prev = atomicCAS(&hashtable[i].key, KEY_INVALID, key);
			
			if (prev == KEY_INVALID) {
				hashtable[i].value = value;
				return;
			}
			else if (prev == key) {
				atomicAdd(update, 1);
				hashtable[i].value = value;
				return;
			}
		}
    }
}



/**
 * @brief Overloaded CUDA kernel to rehash elements from an old table to a new one.
 * @param hashtable The destination (new) hash table.
 * @param hash The source (old) hash table.
 * @param numKeys The capacity of the old hash table.
 * @param maxim The capacity of the new hash table.
 */
__global__ void gpu_hashtable_insert_kernel(gpu_hashtable *hashtable, gpu_hashtable *hash, int numKeys, int maxim)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	
    // Process one slot from the old table per thread, skipping empty slots.
    if (threadId < numKeys && hash[threadId].key != KEY_INVALID)
    {
		int i, key = hash[threadId].key, value =  hash[threadId].value;
		
		int slot = hashFunction(key, maxim);
		
		// Linearly probe and insert into the new table. This version does not
		// need to track updates, as it's just moving existing data.
		for(i = slot; i < maxim; i++)
		{
			int prev = atomicCAS(&hashtable[i].key, KEY_INVALID, key);
			if (prev == KEY_INVALID || prev == key) {
				hashtable[i].value = value;
				return;
			}
		}
		for(i = 0; i < slot; i++){
			int prev = atomicCAS(&hashtable[i].key, KEY_INVALID, key);
			if (prev == KEY_INVALID || prev == key) {
				hashtable[i].value = value;
				return;
			}
		}
    }
}

/**
 * @brief Resizes the hash table on the GPU.
 * @param numBucketsReshape The desired new number of elements, used to calculate new capacity.
 */
void GpuHashTable::reshape(int numBucketsReshape) {

	gpu_hashtable *old_hash = this->hashtable;
	
	// Add a 5% buffer to the new size to reduce future resizes.
	int new_size = numBucketsReshape + numBucketsReshape / 20;

	// Allocate and initialize the new, larger table.
	cudaError_t __cudaCalloc_err = cudaMalloc(&this->hashtable, new_size * sizeof(gpu_hashtable));
	if (__cudaCalloc_err == cudaSuccess)
		cudaMemset(this->hashtable, 0, new_size * sizeof(gpu_hashtable));
	
	// Calculate kernel launch configuration.
	int block_no = (*this->maxim_host + THREAD_BLOCKSIZE - 1) / THREAD_BLOCKSIZE;
	
	// Launch the rehashing kernel.
	gpu_hashtable_insert_kernel>>(this->hashtable, old_hash, *this->maxim_host, new_size);
	
	cudaDeviceSynchronize();

	// Update the host-side capacity tracker.
	*this->maxim_host = new_size;

	// Free the old table memory.
	cudaFree(old_hash);
}

/**
 * @brief Inserts a batch of key-value pairs.
 * @param keys Host pointer to an array of keys.
 * @param values Host pointer to an array of values.
 * @param numKeys The number of pairs to insert.
 * @return True on success.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *keys_device, *values_device;

	// Copy the device-side update counter to the host.
	// This is needed to calculate the current number of unique elements.
	int update;
	cudaMemcpy(&update, this->update_device, sizeof(int), cudaMemcpyDeviceToHost);

	// Check if a reshape is needed.
	// Logic appears to be based on the total number of insertion *attempts* rather than
	// the actual number of unique keys. This might lead to premature reshaping.
	if (this->current_host - update + numKeys > this->maxim_host)
		reshape(*this->current_host + numKeys);

	// Increment the total number of insertion attempts.
	*this->current_host += numKeys;

	// Allocate temporary device buffers and copy data from host.
	cudaError_t __cudaCalloc_err = cudaMalloc(&keys_device, numKeys * sizeof(int));
	if (__cudaCalloc_err == cudaSuccess)
		cudaMemcpy(keys_device, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	__cudaCalloc_err = cudaMalloc(&values_device, numKeys * sizeof(int));
	if (__cudaCalloc_err == cudaSuccess)
		cudaMemcpy(values_device, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Launch the insertion kernel.
	int block_no = (numKeys + THREAD_BLOCKSIZE - 1) / THREAD_BLOCKSIZE;
	gpu_hashtable_insert_kernel>>(this->hashtable, keys_device, values_device, numKeys, *this->maxim_host, this->update_device);
	cudaDeviceSynchronize();

	// Clean up temporary buffers.
	cudaFree(keys_device);
	cudaFree(values_device);
	return true;
}

/**
 * @brief Device function to retrieve a single value for a key.
 * @param hashtable The hash table to search.
 * @param key The key to look up.
 * @param maxim The capacity of the hash table.
 * @return The value associated with the key, or -1 if not found.
 */
__device__ int gpu_hashtable_get(gpu_hashtable *hashtable, int key, int maxim)
{
    int slot = hashFunction(key, maxim);
	int i;
	
	// Linearly probe forward from the hash slot.
	for(i = slot; i < maxim; i++)
		if (hashtable[i].key == key)
			return hashtable[i].value;
	// Handle wrap-around.
	for(i = 0; i < slot; i++)
		if (hashtable[i].key == key)
			return hashtable[i].value;

	// Key not found.
	return -1;
}



/**
 * @brief CUDA kernel to retrieve a batch of values.
 * @param hashtable The hash table to search.
 * @param keys Pointer to device memory with keys to look up.
 * @param values Pointer to device memory where results will be stored.
 * @param numKeys The number of keys in the batch.
 * @param maxim The capacity of the hash table.
 */
__global__ void gpu_hashtable_get_kernel(gpu_hashtable *hashtable, int *keys, int *values, int numKeys, int maxim)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (threadId < numKeys)
    {
		// Each thread calls the device-side get function for one key.
		values[threadId] = gpu_hashtable_get(hashtable, keys[threadId], maxim);
    }
}

/**
 * @brief Retrieves a batch of values for a given set of keys.
 * @param keys Host pointer to an array of keys.
 * @param numKeys The number of keys to retrieve.
 * @return A host pointer to an array with the results. Caller is responsible for freeing this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *values_host, *values_device, *keys_device;

	// Allocate device buffer and copy keys from host.
	cudaError_t __cudaCalloc_err = cudaMalloc(&keys_device, numKeys * sizeof(int));
	if (__cudaCalloc_err == cudaSuccess)
		cudaMemcpy(keys_device, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Allocate device buffer for the results.
	__cudaCalloc_err = cudaMalloc(&values_device, numKeys * sizeof(int));
	DIE(__cudaCalloc_err != cudaSuccess, "cudaMalloc error");

	// Launch the retrieval kernel.
	int block_no = (numKeys + THREAD_BLOCKSIZE - 1) / THREAD_BLOCKSIZE;
	gpu_hashtable_get_kernel>>(this->hashtable, keys_device, values_device, numKeys, *this->maxim_host);
	cudaDeviceSynchronize();

	// Allocate host memory for the final results and copy from device.
	values_host = (int *)malloc(numKeys * sizeof(int));
	DIE(values_host == NULL, "malloc error");
	cudaMemcpy(values_host, values_device, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	// Free temporary device buffers.
	cudaFree(keys_device);
	cudaFree(values_device);
	return values_host;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float.
 *
 * Note: This operation requires a device-to-host memory copy to get the
 * number of updated keys, which can impact performance if called frequently.
 */
float GpuHashTable::loadFactor() {
	int update;
	cudaMemcpy(&update, this->update_device, sizeof(int), cudaMemcpyDeviceToHost);

	// Load factor is calculated based on the number of insertion attempts minus the number of updates.
	return 1.0f * (*this->current_host - update) / *this->maxim_host;
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

#define	KEY_INVALID	0
#define THREAD_BLOCKSIZE 512

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



// Struct definition for a hash table entry.
struct gpu_hashtable
{
	int key, value;
};

// C++ class providing a high-level API for the GPU hash table.
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
	private:
		gpu_hashtable *hashtable;
		int *maxim_host, *current_host;
		int *update_device;
};

#endif
