/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA with linear probing.
 * @details This file defines the device-side kernels and host-side methods for a hash table
 * that operates on the GPU. The implementation uses two separate arrays for keys and values
 * and employs linear probing for collision resolution. It supports batch insertion,
 * batch retrieval, and dynamic resizing (reshaping) based on a load factor.
 *
 * @note This file has an unusual structure, appearing to concatenate the implementation,
 * a header file, and an include of a .cpp file. The documentation will follow this
 * structure without modifying the code. The collision resolution strategy is linear probing.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief Constructor for the GpuHashTable.
 * @param size The initial capacity of the hash table.
 * @details Allocates two separate arrays on the GPU: one for keys and one for values.
 */
GpuHashTable::GpuHashTable(int size) {
	// For multi-GPU systems, this explicitly selects GPU device 1.
	cudaSetDevice(1);
	this->size = size;
	nr_elems = 0; 
	
	// Memory Allocation: Allocate global device memory for the key and value arrays.
	cudaMalloc((void **) &hash_keys, size*sizeof(int));
	cudaMalloc((void **) &hash_values, size*sizeof(int));
}

/**
 * @brief Destructor for the GpuHashTable.
 * @details Frees the device memory allocated for the key and value arrays.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hash_keys);
	cudaFree(hash_values);
}


/**
 * @brief CUDA kernel to rehash elements from an old table to a new, larger table.
 * @param hash_keys Pointer to the new key array.
 * @param hash_values Pointer to the new value array.
 * @param size The capacity of the new hash table.
 * @param numKeys The capacity of the old hash table.
 * @param device_keys Pointer to the old key array.
 * @param device_values Pointer to the old value array.
 * @details Each thread is responsible for re-inserting one element from the old arrays
 * into the new ones. It uses linear probing and atomicCAS to find an empty slot.
 */
__global__ void resh(int *hash_keys, int *hash_values, int size, int numKeys, 
	int *device_keys, int *device_values) {
	// Thread Indexing: Each thread gets a unique index to process one element.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	// Pre-condition: Process only valid elements from the old arrays.
	if (i < numKeys && device_keys[i] != 0) {
		// Calculate the initial hash position in the new table.
		int hash_result = hash_func(device_keys[i], size);
		// Block Logic: Linear probing loop to find an empty slot.
		// Invariant: Keep probing until an empty slot is claimed.
		while(true){
			// Synchronization: Use atomicCAS to safely claim an empty slot (where key is 0).
			// This prevents race conditions if multiple old keys hash to the same new slots.
			if (atomicCAS(&(hash_keys[hash_result]), 0, device_keys[i]) == 0) {
				hash_values[hash_result] = device_values[i];
				break; // Slot found, exit loop.
			} else {
				// Collision: Move to the next slot.
				hash_result++;
				// Wrap around if the end of the table is reached.
				hash_result = hash_result % size;
			}
		}
	}
}

/**
 * @brief Resizes and rehashes the hash table to a new, larger capacity.
 * @param numBucketsReshape The new target capacity for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int *new_device_keys, *new_device_values;

	// Memory Allocation: Allocate new, larger arrays on the device for keys and values.
	cudaMalloc((void **) &new_device_keys, numBucketsReshape*sizeof(int));
	cudaMalloc((void **) &new_device_values, numBucketsReshape*sizeof(int));
	// Initialize the new arrays to zero.
	cudaMemset(new_device_keys, 0,numBucketsReshape*sizeof(int));
	cudaMemset(new_device_values, 0,numBucketsReshape*sizeof(int));
	
	// Kernel Launch Configuration.
	const size_t block_size = 64;
	size_t blocks_no = size / block_size; // Note: This should use the old 'size'.
	
	if (size % block_size)
		++blocks_no;
	
	// Launch the rehash kernel.
	resh>>(new_device_keys, new_device_values, numBucketsReshape, size, hash_keys, hash_values);
	// Synchronization: Wait for the rehash kernel to complete.
	cudaDeviceSynchronize();
	
	// Update the host-side capacity.
	size = numBucketsReshape;

	// Free the old device memory.
	cudaFree(hash_keys);
	cudaFree(hash_values); 
	
	// Point the class members to the new device arrays.
	hash_keys = new_device_keys;
	hash_values = new_device_values;
}


/**
 * @brief CUDA kernel to insert a batch of key-value pairs.
 * @details Each thread handles one key-value pair. It calculates the hash and uses
 * linear probing with atomicCAS to find an empty slot or update an existing key.
 */
__global__ void insert(int *hash_keys, int *hash_values, int size, int numKeys, 
			int *device_keys, int *device_values) {
	// Thread Indexing: Each thread is assigned one key-value pair to insert.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (i < numKeys) {
		int hash_result = hash_func(device_keys[i], size);
		// Block Logic: Linear probing loop.
		// Invariant: Continue until the key is inserted or updated.
		while(true){
			// Synchronization: Use atomicCAS. This handles two cases:
			// 1. If slot is empty (0), it inserts the new key.
			// 2. If slot contains the same key, `atomicCAS` fails but the subsequent
			//    check `hash_keys[hash_result] == device_keys[i]` handles the update.
			if (atomicCAS(&(hash_keys[hash_result]), 0, device_keys[i]) == 0 || hash_keys[hash_result] == device_keys[i]) {
				// Update the value associated with the key.
				hash_values[hash_result] = device_values[i];
				break; // Exit loop.
			} else {
				// Collision: Probe the next slot.
				hash_result++;
				hash_result = hash_result % size;
			}
		}
	}
}

/**
 * @brief Inserts a batch of keys and values into the hash table.
 * @param keys Host pointer to the array of keys.
 * @param values Host pointer to the array of values.
 * @param numKeys The number of key-value pairs to insert.
 * @return Always returns true.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	
	// Block Logic: Check the load factor and trigger a reshape if it exceeds a threshold (80%).
	if (nr_elems + numKeys > 0.8*size) {
		// The new size is chosen to be 125% of the required elements.
		reshape((nr_elems + numKeys)*1.25);
	}

	int *device_keys, *device_values;
	
	// Kernel Launch Configuration.
	const size_t block_size = 64;
	size_t blocks_no = numKeys / block_size;
	
	if (numKeys % block_size)
		++blocks_no;

	// Memory Transfer: Allocate device memory and copy the batch from host to device.
	cudaMalloc((void **) &device_keys, numKeys*sizeof(int));
	cudaMalloc((void **) &device_values, numKeys*sizeof(int));
	cudaMemcpy(device_keys, keys, numKeys*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys*sizeof(int), cudaMemcpyHostToDevice);
	
	// Launch the insertion kernel.
	insert>>(hash_keys, hash_values, size,
			numKeys, device_keys, device_values);
	
	// Synchronization: Wait for the insertion to complete.
	cudaDeviceSynchronize();
	nr_elems += numKeys;

	// Free the temporary device memory for the batch.
	cudaFree(device_keys);
	cudaFree(device_values);
	
	return true;
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @details Each thread looks up one key. It uses linear probing to find the key.
 * The search stops if the key is found or an empty slot (key=0) is encountered.
 */
__global__ void get(int *hash_keys, int *hash_values, int size, int numKeys, 
	int *device_keys, int *device_values) {

	// Thread Indexing: Each thread is assigned one key to look up.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (i < numKeys) {
		int hash_result = hash_func(device_keys[i], size);
		// Block Logic: Linear probing loop for searching.
		// Invariant: Continue until key is found or an empty slot proves it's not present.
		while (true) {
			// Memory Access: Read from global device memory.
			if (hash_keys[hash_result] == device_keys[i] || hash_keys[hash_result] == 0) {
				// If key matches or an empty slot is found, store the corresponding value.
				// If the slot was empty, `hash_values[hash_result]` will be 0.
				device_values[i] = hash_values[hash_result];
				break; // Exit loop.
			} else {
				// Collision: Probe the next slot.
				hash_result++;
				hash_result = hash_result % size;	
			}
		}
	}
}

/**
 * @brief Retrieves values for a batch of keys.
 * @param keys Host pointer to an array of keys to look up.
 * @param numKeys The number of keys to retrieve.
 * @return A host pointer to an array with the results. Caller must free this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	// Allocate host memory for the final results.
	int *result = (int *) calloc(numKeys, sizeof(int));
	int *device_keys, *device_values;

	// Memory Transfer: Allocate temporary device memory for keys and results.
	cudaMalloc((void **) &device_keys, numKeys*sizeof(int));
	cudaMalloc((void **) &device_values, numKeys*sizeof(int));

	cudaMemset(device_values, 0,numKeys*sizeof(int));
	cudaMemcpy(device_keys, keys, numKeys*sizeof(int), cudaMemcpyHostToDevice);
	
	// Kernel Launch Configuration.
	const size_t block_size = 64;
	size_t blocks_no = numKeys / block_size;
	
	if (numKeys % block_size)
		++blocks_no;

	// Launch the retrieval kernel.
	get>>(hash_keys, hash_values, size, numKeys, device_keys, device_values);
	// Synchronization: Wait for the kernel to complete.
	cudaDeviceSynchronize();

	// Memory Transfer: Copy results from device back to host.
	cudaMemcpy(result, device_values, numKeys*sizeof(int), cudaMemcpyDeviceToHost);

	// Free temporary device memory.
	cudaFree(device_keys);
	cudaFree(device_values);
	return result;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float.
 */
float GpuHashTable::loadFactor() {
	return (float)nr_elems/(float)size; 
}


// Convenience macros for a simplified external API.
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

/*
 * @note The following line includes a .cpp file, which is highly unconventional
 * and can lead to build and maintenance issues. The content of "test_map.cpp"
 * is not available for documentation.
 */
#include "test_map.cpp"

/*
 * @note The following block appears to be a re-definition of the header file
 * content, which is redundant in this context but is being documented as is.
 */
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
	
/**
 * @brief A list of large prime numbers for hashing.
 */
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
 * @brief The hash function used by the kernels.
 * @param data The key to hash.
 * @param limit The capacity of the hash table.
 * @return The calculated hash index.
 * @note This is a device function, callable from kernel code.
 */
__device__ int hash_func(int data, int limit) {
	return ((long)abs(data) * 13169977llu) % limit;
}



/**
 * @class GpuHashTable
 * @brief Host-side class definition for the GPU hash table.
 */
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

		// Public members holding the state of the hash table.
		int size; // Current capacity.
		int *hash_keys, *hash_values; // Pointers to device memory.
		int nr_elems; // Number of elements.
};

#endif
