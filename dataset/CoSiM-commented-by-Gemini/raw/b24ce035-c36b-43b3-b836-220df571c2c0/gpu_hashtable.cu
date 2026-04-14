/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 *
 * This file provides the `GpuHashTable` class, which implements a hash table
 * that resides in GPU memory. It uses open addressing with linear probing to
 * handle hash collisions. The implementation is designed for batch operations
 * (insert and get) to leverage the parallel processing power of the GPU.
 * It also supports dynamic resizing (rehashing) when the load factor exceeds
 * a certain threshold.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief Constructs a GpuHashTable.
 * @param size The initial number of buckets in the hash table.
 *
 * Allocates memory for the hash table on the GPU and initializes it to zero.
 */
GpuHashTable::GpuHashTable(int size)
{
	cudaError_t error;

	// Allocate memory on the GPU for the hash table array.
	error = cudaMalloc(&(GpuHashTable::hashtable), size * sizeof(hT));
	DIE(error != cudaSuccess || GpuHashTable::hashtable == NULL, "cudaMalloc hashtable error");
	// Initialize the allocated memory to all zeros.
	error = cudaMemset(GpuHashTable::hashtable, 0, size * sizeof(hT));
	DIE(error != cudaSuccess, "cudaMemset hashtable error");

	
	GpuHashTable::currentTableSize = 0;
	GpuHashTable::tableSize = size;
}



/**
 * @brief Destroys the GpuHashTable.
 *
 * Frees the GPU memory allocated for the hash table.
 */
GpuHashTable::~GpuHashTable()
{
	cudaError_t error;

	
	error = cudaFree(GpuHashTable::hashtable);
	DIE(error != cudaSuccess, "cudaFree hashtable error");

	
	GpuHashTable::currentTableSize = 0;
	GpuHashTable::tableSize = 0;
}


/**
 * @brief CUDA kernel to copy existing key-value pairs from the old hash table
 *        into compact, temporary arrays during a reshape operation.
 * @param hashtable The source hash table on the GPU.
 * @param tableSize The size of the source hash table.
 * @param device_keys Destination array on the GPU for keys.
 * @param device_values Destination array on the GPU for values.
 * @param counter An atomic counter to get a unique index for each element.
 *
 * Parallelism: Each thread is responsible for checking one slot in the hash table.
 * If the slot is not empty, the thread atomically gets an index and copies the
 * key-value pair to the temporary arrays.
 */
__global__ void copyForReshape(hT *hashtable, int tableSize,
								int *device_keys, int *device_values,
								int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if(idx < tableSize) {
		// Check if the slot is occupied (key is non-zero).
		if (hashtable[idx].key != 0) {
			// Atomically increment the counter to get a unique destination index.
			int index = atomicAdd(counter, 1);

			device_keys[index] = hashtable[idx].key;
			device_values[index] = hashtable[idx].value;
		}
	}
}



/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The desired number of buckets for the new table.
 *
 * Algorithm (Rehashing):
 * 1. Allocates a new, larger hash table on the GPU.
 * 2. If the current table is not empty, it copies all existing key-value pairs
 *    from the old GPU table to temporary arrays on the host (CPU).
 * 3. Frees the old GPU hash table.
 * 4. Replaces the old table with the new one.
 * 5. Re-inserts all the key-value pairs from the host arrays into the new table
 *    using `insertBatch`.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t error;
	hT *newHashTable = NULL;
	int new_size = 1.2f * numBucketsReshape;

	// Step 1: Allocate a new, larger table on the GPU.
	error = cudaMalloc(&newHashTable, new_size * sizeof(hT));
	DIE(error != cudaSuccess || newHashTable == NULL, "cudaMalloc new hashtable error");
	error = cudaMemset(newHashTable, 0, new_size * sizeof(hT));
	DIE(error != cudaSuccess, "cudaMemset new hashtable error");


	// Step 2: If the current table has elements, copy them out.
	if(GpuHashTable::currentTableSize != 0) {
		const size_t block_size = 1024;
		size_t blocks_no = GpuHashTable::tableSize / block_size;
 
		if (GpuHashTable::tableSize % block_size) 
			++blocks_no;

		// Allocate temporary storage on the GPU and host.
		int *device_keys = NULL;
		int *counter = NULL;
		int *device_values = NULL;


		error = cudaMalloc(&device_keys, GpuHashTable::currentTableSize * sizeof(int));
		DIE(error != cudaSuccess || device_keys == NULL, "cudaMalloc device_keys error");

		error = cudaMalloc(&device_values, GpuHashTable::currentTableSize * sizeof(int));
		DIE(error != cudaSuccess || device_values == NULL, "cudaMalloc device_values error");

		
		error = cudaMalloc(&counter, sizeof(int));
		DIE(error != cudaSuccess || counter == NULL, "cudaMalloc counter error");
		error = cudaMemset(counter, 0, sizeof(int));
		DIE(error != cudaSuccess, "cudaMemset counter error");


		int *host_keys = (int *)malloc(GpuHashTable::currentTableSize * sizeof(int));
		DIE(host_keys == NULL, "malloc host_keys error");

		int *host_values = (int *)malloc(GpuHashTable::currentTableSize * sizeof(int));
		DIE(host_values == NULL, "malloc host_values error");

		
		// Launch kernel to copy all existing elements to compact GPU arrays.
		copyForReshape>>(GpuHashTable::hashtable,
												GpuHashTable::tableSize,
												device_keys, device_values,
												counter);
		error = cudaDeviceSynchronize();
		DIE(error != cudaSuccess, "cudaDeviceSynchronize error");


		// Copy the compacted arrays from GPU to host memory.
		error = cudaMemcpy(host_keys, device_keys,
							GpuHashTable::currentTableSize * sizeof(int),
							cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_keys error");

		error = cudaMemcpy(host_values, device_values,
							GpuHashTable::currentTableSize * sizeof(int),
							cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_values error");


		// Step 3 & 4: Destroy the old table and assign the new one.
		GpuHashTable::~GpuHashTable();

		
		GpuHashTable::tableSize = new_size;
		GpuHashTable::hashtable = newHashTable;
		GpuHashTable::currentTableSize = 0;
		

		int numKeys = 0;
		error = cudaMemcpy(&numKeys, counter, sizeof(int), cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy numKeys error");
		
		// Step 5: Re-insert all old elements into the new table.
		insertBatch(host_keys, host_values, numKeys);


		// Clean up temporary allocations.
		error = cudaFree(device_keys);
		DIE(error != cudaSuccess, "cudaFree device_keys error");

		error = cudaFree(device_values);
		DIE(error != cudaSuccess, "cudaFree device_values error");

		error = cudaFree(counter);
		DIE(error != cudaSuccess, "cudaFree counter error");

		free(host_keys);
		free(host_values);

		return;

	}

	// If the table was empty, just replace it with the new empty table.
	GpuHashTable::~GpuHashTable();

	
	GpuHashTable::tableSize = new_size;

	GpuHashTable::hashtable = newHashTable;
}


/**
 * @brief CUDA kernel to compute hash codes for a batch of keys.
 * @param keys The input keys.
 * @param hashcodes The output array for computed hash codes.
 * @param numkeys The number of keys in the batch.
 * @param tablesize The current size of the hash table.
 *
 * Parallelism: Each thread computes the hash for a single key.
 * Hash Function: A simple multiplicative hashing scheme.
 */
__global__ void getHashCode(int *keys, int *hashcodes, int numkeys, int tablesize)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numkeys)
		hashcodes[idx] = (((long)keys[idx]) * 1402982055436147llu)
								% 452517535812813007llu % tablesize;
}



/**
 * @brief CUDA kernel to insert or update key-value pairs in the hash table.
 * @param hashcodes The current hash code for each key.
 * @param keys The keys to insert. A key is set to 0 by the kernel when processed.
 * @param values The corresponding values to insert.
 * @param numKeys The total number of keys in the batch.
 * @param hashtable The GPU hash table.
 * @param currentTableSize An atomic counter for the number of elements in the table.
 * @param tablesize The size of the hash table.
 * @param counter An atomic counter for the number of successfully processed keys in this pass.
 *
 * Parallelism: Each thread attempts to insert one key-value pair.
 * Collision Handling:
 * - If a slot is empty, the thread uses `atomicCAS` to claim it.
 * - If the key already exists, the value is updated.
 * - If a different key occupies the slot (a collision), the thread increments
 *   its hash code (linear probing) to try the next slot in a subsequent kernel launch.
 */
__global__ void insertKeysandValues(int *hashcodes, int *keys, int *values,
									int numKeys, hT *hashtable,
									int *currentTableSize, int tablesize,
									int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numKeys) {
		if (keys[idx] == 0) // Key has already been processed in a previous pass.
			return;

		// Attempt to atomically claim an empty slot (where key is 0).
		int key = atomicCAS(&hashtable[hashcodes[idx]].key, 0, keys[idx]);
		
		if (key == 0) { // Success: claimed an empty slot.
			hashtable[hashcodes[idx]].value = values[idx];
			
			keys[idx] = 0; // Mark this key as processed.
			atomicAdd(currentTableSize, 1);
			atomicAdd(counter, 1);
		} else if (key == keys[idx]) { // Success: key already existed.
			// Update the value (upsert).
			hashtable[hashcodes[idx]].value = values[idx];
			atomicAdd(counter, 1);
			
			keys[idx] = 0; // Mark this key as processed.
		} else { // Collision: another key was in the slot.
			// Apply linear probing by incrementing the hash code.
			// The while loop in insertBatch will re-launch this kernel to retry.
			hashcodes[idx] += 1;
			hashcodes[idx] %= tablesize;
		}
	}
}



/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys An array of keys on the host.
 * @param values An array of values on the host.
 * @param numKeys The number of pairs to insert.
 * @return True on success.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys = NULL;
	int *device_values = NULL;
	cudaError_t error;


	// Check if the table needs to be resized before insertion.
	if ((GpuHashTable::currentTableSize + numKeys) > GpuHashTable::tableSize)
		reshape((GpuHashTable::currentTableSize + numKeys));


	// Copy host data (keys and values) to the GPU.
	error = cudaMalloc(&device_keys, numKeys * sizeof(int));
	DIE(error != cudaSuccess || device_keys == NULL, "cudaMalloc device_keys error");

	error = cudaMalloc(&device_values, numKeys * sizeof(int));
	DIE(error != cudaSuccess || device_values == NULL, "cudaMalloc device_values error");


	
	error = cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(error != cudaSuccess, "cudaMemcpy keys error");

	error = cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(error != cudaSuccess, "cudaMemcpy values error");


	// Compute initial hash codes for all keys in parallel.
	int *hashcodes = NULL;
	error = cudaMalloc(&hashcodes, numKeys * sizeof(int));
	DIE(error != cudaSuccess || hashcodes == NULL, "cudaMalloc hashcodes error");

	const size_t block_size = 1024;
	size_t blocks_no = numKeys / block_size;
 
	if (numKeys % block_size) 
		++blocks_no;

	getHashCode>>(device_keys, hashcodes,
											numKeys, GpuHashTable::tableSize);
	error = cudaDeviceSynchronize();
	DIE(error != cudaSuccess, "cudaDeviceSynchronize error");



	// Allocate counters on the device.
	int *device_current = NULL;
	error = cudaMalloc(&device_current, sizeof(int));
	DIE(error != cudaSuccess, "cudaMalloc device_current error");
	error = cudaMemset(device_current, 0, sizeof(int));
	DIE(error != cudaSuccess, "cudaMemset device_current error");



	
	int *device_counter = NULL;
	error = cudaMalloc(&device_counter, sizeof(int));
	DIE(error != cudaSuccess || device_counter == NULL, "cudaMalloc device_counter error");
	error = cudaMemset(device_counter, 0, sizeof(int));
	DIE(error != cudaSuccess, "cudaMemset device_counter error");


	int host_counter = 0;
	int old_counter = 0;
	// Kernel-in-a-loop: Repeatedly launch the insertion kernel until all keys
	// have been successfully inserted. This loop handles collisions, as keys that
	// collided in one pass will have their hash codes updated for the next pass.
	while(1) {
		
		insertKeysandValues>>(hashcodes, device_keys,
													device_values, numKeys,
													GpuHashTable::hashtable,
													device_current,
													GpuHashTable::tableSize,
													device_counter);
		error = cudaDeviceSynchronize();
		DIE(error != cudaSuccess, "cudaDeviceSynchronize error");

		old_counter = host_counter;
		error = cudaMemcpy(&host_counter, device_counter, sizeof(int), cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_counter error");

		// Update the total size based on newly inserted elements.
		GpuHashTable::currentTableSize += host_counter - old_counter;

		// Exit loop if the counter shows all keys have been processed.
		if(host_counter == numKeys)
			break;
	}
	

	// Free all temporary GPU memory.
	error = cudaFree(device_keys);
	DIE(error != cudaSuccess, "cudaFree device_keys error");

	error = cudaFree(device_values);
	DIE(error != cudaSuccess, "cudaFree device_values error");

	error = cudaFree(hashcodes);
	DIE(error != cudaSuccess, "cudaFree hashcodes error");

	error = cudaFree(device_current);
	DIE(error != cudaSuccess, "cudaFree device_current error");

	error = cudaFree(device_counter);
	DIE(error != cudaSuccess, "cudaFree device_counter error");

	return true;
}


/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param values The output array for retrieved values.
 * @param hashcodes The current hash code for each key being searched.
 * @param keys The keys to search for. A key is set to 0 by the kernel when found.
 * @param numKeys The number of keys in the batch.
 * @param hashtable The GPU hash table.
 * @param tablesize The size of the hash table.
 * @param counter An atomic counter for the number of successfully found keys.
 *
 * Parallelism: Each thread searches for one key.
 * Collision Handling: Uses linear probing. If the key at the current hash index
 * does not match, the hash code is incremented to check the next slot in a
 * subsequent kernel launch.
 */
__global__ void getbatch(int *values, int *hashcodes, int *keys, int numKeys,
						hT *hashtable, int tablesize, int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numKeys) {
		if (keys[idx] == 0) // Key already found.
			return;

		if(hashtable[hashcodes[idx]].key == keys[idx]) {
			// Key found, retrieve value and mark as processed.
			values[idx] = hashtable[hashcodes[idx]].value;
			atomicAdd(counter, 1);
			keys[idx] = 0;
		} else {
			// Collision or empty slot, apply linear probing for the next iteration.
			hashcodes[idx] += 1;
			hashcodes[idx] %= tablesize;
		}
	}
}



/**
 * @brief Retrieves values for a batch of keys from the hash table.
 * @param keys An array of keys on the host to search for.
 * @param numKeys The number of keys to retrieve.
 * @return A host pointer to an array containing the retrieved values. The caller
 *         is responsible for freeing this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *ret_values = (int *)malloc(numKeys * sizeof(int));
	cudaError_t error;

	// Allocate GPU memory for keys and return values.
	int *device_ret_values = NULL;
	error = cudaMalloc(&device_ret_values, numKeys * sizeof(int));
	DIE(error != cudaSuccess || device_ret_values == NULL, "cudaMalloc device_ret_values error");

	
	int *device_keys = NULL;
	error = cudaMalloc(&device_keys, numKeys * sizeof(int));
	DIE(error != cudaSuccess || device_keys == NULL, "cudaMalloc device_keys error");
	error = cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(error != cudaSuccess, "cudaMemcpy keys error");


	// Compute initial hash codes for all keys.
	int *hashcodes = NULL;
	error = cudaMalloc(&hashcodes, numKeys * sizeof(int));
	DIE(error != cudaSuccess || hashcodes == NULL, "cudaMalloc hashcodes error");

	const size_t block_size = 1024;
	size_t blocks_no = numKeys / block_size;
 
	if (numKeys % block_size) 
		++blocks_no;

	getHashCode>>(device_keys, hashcodes, numKeys, GpuHashTable::tableSize);
	error = cudaDeviceSynchronize();
	DIE(error != cudaSuccess, "cudaDeviceSynchronize error");



	int *device_counter = NULL;
	error = cudaMalloc(&device_counter, sizeof(int));
	DIE(error != cudaSuccess || device_counter == NULL, "cudaMalloc device_counter error");
	error = cudaMemset(device_counter, 0, sizeof(int));
	DIE(error != cudaSuccess, "cudaMemset device_counter error");

	// Kernel-in-a-loop: Repeatedly launch the get kernel until all keys are found.
	while(1) {
		getbatch>>(device_ret_values, hashcodes,
											device_keys, numKeys,
											GpuHashTable::hashtable,
											GpuHashTable::tableSize, device_counter);
		error = cudaDeviceSynchronize();
		DIE(error != cudaSuccess, "cudaDeviceSynchronize error");

		int host_counter = 0;
		error = cudaMemcpy(&host_counter, device_counter, sizeof(int), cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_counter error");

		if(host_counter == numKeys)
			break;
	}


	
	// Copy the retrieved values from GPU back to host memory.
	error = cudaMemcpy(ret_values, device_ret_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	DIE(error != cudaSuccess, "cudaMemcpy ret_values error");


	// Free all temporary GPU memory.
	error = cudaFree(device_keys);
	DIE(error != cudaSuccess, "cudaFree device_keys error");

	error = cudaFree(device_ret_values);
	DIE(error != cudaSuccess, "cudaFree device_ret_values error");

	error = cudaFree(hashcodes);
	DIE(error != cudaSuccess, "cudaFree hashcodes error");

	error = cudaFree(device_counter);
	DIE(error != cudaSuccess, "cudaFree device_counter error");

	return ret_values;
}



/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float.
 */
float GpuHashTable::loadFactor() {
	return ((float)GpuHashTable::currentTableSize) / ((float)GpuHashTable::tableSize);
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

/**
 * @brief A macro for error checking CUDA and C library calls.
 * If the assertion is true, it prints an error message and exits.
 */
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




int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

typedef struct hashtableCell{
	int key;
	int value;
} hT;




class GpuHashTable
{
	public:
		hT *hashtable;
		int tableSize;
		int currentTableSize;
		

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
