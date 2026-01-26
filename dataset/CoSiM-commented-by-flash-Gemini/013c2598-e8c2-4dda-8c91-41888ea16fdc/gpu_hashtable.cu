/**
 * @file gpu_hashtable.cu
 * @brief This file implements a GPU-accelerated hash table using CUDA.
 *        It contains both CUDA device code (kernels and device functions)
 *        and C++ host code to manage the hash table on the GPU, including
 *        initialization, insertion, retrieval, and dynamic resizing (rehashing).
 *        The underlying hash table uses quadratic probing for collision resolution.
 * @note This file includes the content of `gpu_hashtable.hpp` at the end,
 *       which defines common structures, constants, and the host-side class interface.
 */

#include <iostream>  // For debugging output (printf on host).
#include <cstdio>    // For printf.
#include <cstdlib>   // For exit(), calloc(), free().
#include <cuda_runtime.h> // CUDA runtime API for memory management and kernel launching.
#include <cuda.h>    // CUDA driver API (less common for direct use in host code).
#include <math.h>    // For math functions.
#include <errno.h>   // For errno.

#include "gpu_hashtable.hpp" // Header for hash table structures and host class definition.

/**
 * @brief Device function to compute a hash value for a given key.
 *
 * This hash function uses modular arithmetic with large prime numbers
 * to distribute keys across the hash table's capacity, aiming for a
 * uniform distribution.
 *
 * @param key The integer key to hash.
 * @param table_capacity The current capacity of the hash table.
 * @return An integer hash value within the range [0, table_capacity - 1].
 */
__device__ int hash_function(int key, unsigned long long table_capacity) {
	// A combination of multiplication and modulo operations with large primes for hashing.
	return (((key * 350745513859007llu) % 3452434812973llu) % table_capacity);
}


/**
 * @brief CUDA kernel for initializing the hash table structure on the device.
 *
 * This kernel is launched once to set up the `my_hashTable` struct.
 * It links the `hashtable` descriptor to the pre-allocated `table` array
 * and initializes all elements in the `table` to an empty state
 * (`elem_key = 0`, `elem_value = 0`).
 *
 * @param hashtable A device pointer to the `my_hashTable` descriptor.
 * @param table A device pointer to the array of `my_hashElem` representing the hash table buckets.
 * @param size The total capacity of the hash table.
 */
__global__ void kernel_init_table(my_hashTable* hashtable, my_hashElem* table, int size) {
	hashtable->table = table;       // Link the descriptor to the allocated table memory.
	hashtable->max_items = size;    // Set the maximum capacity.

	printf("SUS DE TOT MAX : %d \n", hashtable->max_items); // Debug print on device.
	// Initialize all hash table elements to be empty.
	for (int i = 0; i < size; i++) {
		hashtable->table[i].elem_key = 0;   // KEY_INVALID is 0.
		hashtable->table[i].elem_value = 0;
	}

	hashtable->curr_nr_items = 0; // Initialize current number of items to 0.
}


/**
 * @brief Constructor for the GpuHashTable class.
 *
 * This host-side function initializes a GPU-accelerated hash table.
 * It allocates device memory for the hash table elements and the hash table
 * descriptor, then launches a CUDA kernel to perform device-side initialization.
 *
 * @param size The desired initial capacity (number of buckets) for the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	my_hashElem* table = NULL;
	// Allocate device memory for the hash table elements (buckets).
	cudaMalloc((void**)&table, size * sizeof(my_hashElem));
	// Initialize all allocated device memory to zeros (marking all buckets as empty).
	cudaMemset(table, 0, size * sizeof(my_hashElem));

	// Allocate device memory for the hash table descriptor structure.
	cudaMalloc((void **)&dev_hash, sizeof(my_hashTable));
	// Launch kernel to initialize the hash table descriptor on the device.
	kernel_init_table > > (dev_hash, table, size);
	cudaDeviceSynchronize(); // Synchronize to ensure kernel completes before host continues.
}


/**
 * @brief Destructor for the GpuHashTable class.
 *
 * This host-side function frees all device memory allocated for the hash table
 * elements and the hash table descriptor when the GpuHashTable object is destroyed.
 */
GpuHashTable::~GpuHashTable() {
	// Free device memory for the hash table descriptor (which implicitly includes the table).
	cudaFree(dev_hash);
}

/**
 * @brief CUDA kernel for rehashing elements from an old hash table to a new, larger one.
 *
 * Each thread processes one element from the old hash table. It computes the
 * initial hash for this element in the new table, then uses quadratic probing
 * and `atomicCAS` to find an empty slot and insert the element.
 *
 * @param hash A device pointer to the old `my_hashTable` descriptor.
 * @param new_hash A device pointer to the newly allocated `my_hashElem` array (new table).
 * @param new_max_items The capacity of the new hash table.
 */
__global__ void kernel_rehash(my_hashTable* hash, my_hashElem* new_hash, unsigned long long new_max_items) {
	// Calculate global thread index.
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	// Check if the index is within the bounds of the old hash table.
	if (index >= hash->max_items) return; // Guard against out-of-bounds access.

	my_hashElem item = hash->table[index]; // Get the item from the old hash table.
	int key = item.elem_key;

	// Only rehash valid keys (KEY_INVALID is 0).
	if (key == KEY_INVALID) return;

	int hashed = hash_function(key, new_max_items); // Compute initial hash for the new table.
	int i = new_max_items; // Probing counter, initialized with a large value.

	// Quadratic probing loop to find an empty slot in the new table.
	while (i != 0) {
		// Attempt to atomically insert the key.
		// If `new_hash[hashed].elem_key` is 0 (empty), it gets replaced by `item.elem_key`.
		int aux = atomicCAS(&new_hash[hashed].elem_key, 0, item.elem_key);
		if (aux == 0) { // Successfully inserted into an empty slot.
			new_hash[hashed].elem_value = item.elem_value; // Set value.
			return;
		}
		// If collision, compute next probe location using quadratic probing.
		hashed = (hashed + i * i) % new_max_items;
		i--; // Decrement probe counter.
	}
	// If the loop finishes without insertion, it implies the new table is full or probing failed.
}


/**
 * @brief Resizes the hash table on the GPU and rehashes existing elements.
 *
 * This host-side function orchestrates the dynamic resizing of the hash table.
 * It calculates a new size, allocates a new device table, launches a kernel
 * to rehash elements from the old table to the new one, and then updates
 * the hash table descriptor on the device.
 *
 * @param numBucketsReshape The suggested new number of buckets (capacity).
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int new_size;
	// Determine the new size based on MIN_LOAD and numBucketsReshape.
	if (OK == 1) // OK seems to be a flag indicating a previous load factor issue.
		new_size = numBucketsReshape * 100 / MIN_LOAD;
	else
		new_size = numBucketsReshape;

	unsigned long long curr_max_items;
	// Retrieve current max_items from device to host.
	cudaMemcpy(&curr_max_items, &dev_hash->max_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	printf("fac reshape cu1 %d \n", new_size); // Debug print.
	
	// Ensure new_size is not smaller than current max_items.
	if (new_size <= curr_max_items)
		new_size = curr_max_items;

	my_hashElem* new_table = NULL;
	// Allocate device memory for the new hash table.
	cudaMalloc((void**)&new_table, new_size * sizeof(my_hashElem));
	// Initialize the new hash table memory to zeros (empty).
	cudaMemset(new_table, 0, new_size * sizeof(my_hashElem)); // Corrected memset size.
	printf("fac reshape cu2 %d \n", new_size); // Debug print.

	// Create a host-side copy of the hash table descriptor.
	my_hashTable *hostHashtable = (my_hashTable*)calloc(1, sizeof(my_hashTable));
	cudaMemcpy(hostHashtable, dev_hash, sizeof(my_hashTable), cudaMemcpyDeviceToHost);

	// Calculate grid dimensions for kernel_rehash.
	int blocks_nr = curr_max_items / 256;
	if (curr_max_items % 256 != 0) {
		blocks_nr++;
	}

	// Launch kernel to rehash elements from the old table to the new one.
	// Assumes old table `hash->table` is still accessible on device.
	kernel_rehash > > (dev_hash, new_table, new_size);
	cudaDeviceSynchronize(); // Synchronize to ensure rehashing completes.

	// Free the old hash table elements memory on device.
	cudaFree(hostHashtable->table);
	// Update host-side descriptor with new table and new max_items.
	hostHashtable->table = new_table;
	hostHashtable->max_items = new_size; // Should be new_size, not numBucketsReshape.
	// Copy updated descriptor from host to device.
	cudaMemcpy(dev_hash, hostHashtable, sizeof(my_hashTable), cudaMemcpyHostToDevice);

	free(hostHashtable); // Free host-side temporary descriptor.
}

/**
 * @brief CUDA kernel for inserting a batch of key-value pairs into the hash table.
 *
 * Each thread processes one key-value pair from the input arrays. It computes
 * the initial hash, then uses quadratic probing and `atomicCAS` for collision
 * resolution to find an empty slot or update an existing key's value.
 *
 * @param hash A device pointer to the `my_hashTable` descriptor.
 * @param keys A device pointer to an array of integer keys to insert.
 * @param values A device pointer to an array of integer values to insert.
 * @param numKeys The number of key-value pairs in the batch.
 */
__global__ void kernel_insert(my_hashTable* hash, int* keys, int* values, int numKeys) {
	// Calculate global thread index.
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= numKeys) return; // Guard against out-of-bounds access.

	int key = keys[index];
	int value = values[index];

	int hashed = hash_function(key, hash->max_items); // Compute initial hash.
	
	my_hashElem item; // Temporary item for insertion.
	item.elem_key = key;
	item.elem_value = value;

	int i = hash->max_items; // Probing counter.
	
	// Quadratic probing loop to find an empty slot or update existing key.
	while (i != 0) {
		// Attempt to atomically compare and swap the key.
		int aux = atomicCAS(&(hash->table[hashed].elem_key), KEY_INVALID, item.elem_key);
		
		if (aux == key) { // Key already exists, update its value.
			hash->table[hashed].elem_value = value;
			return;
		}
		else if (aux == KEY_INVALID) { // Successfully inserted into an empty slot.
			atomicAdd(&hash->curr_nr_items, 1); // Atomically increment item count.
			hash->table[hashed].elem_value = value;
			return;
		}

		// If collision, compute next probe location using quadratic probing.
		hashed = (hashed + i * i) % hash->max_items;
		i--; // Decrement probe counter.
	}
	// If the loop finishes without insertion, it implies the table is full or probing failed.
}


/**
 * @brief Checks if the hash table's load factor, after adding a new batch of items,
 *        would exceed a predefined maximum load.
 *
 * This host-side function retrieves the current state of the hash table from
 * the device and performs a load factor calculation.
 *
 * @param batchSize The number of new items planned for insertion.
 * @return True if the projected load factor is within limits, False otherwise.
 */
bool GpuHashTable::check_l(unsigned long long batchSize) {
	float loadFactor;
	unsigned long long max_items, curr_nr_items;

	// Copy max_items and curr_nr_items from device to host.
	cudaMemcpy(&max_items, &dev_hash->max_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(&curr_nr_items, &dev_hash->curr_nr_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	printf("CHECK %d \n", max_items); // Debug print.
	// Calculate the projected load factor.
	loadFactor = 1.0f * (curr_nr_items + batchSize) / max_items;

	if (loadFactor * 100 > MAX_LOAD) // MAX_LOAD is in percentage.
		return false;

	return true;
}


/**
 * @brief Inserts a batch of key-value pairs into the GPU hash table.
 *
 * This host-side function checks the load factor, triggers a reshape operation
 * if necessary, copies input data to the device, launches the `kernel_insert`
 * kernel, and manages device memory.
 *
 * @param keys A pointer to an array of integer keys on the host.
 * @param values A pointer to an array of integer values on the host.
 * @param numKeys The number of key-value pairs in the batch.
 * @return True if the batch insertion process was initiated successfully.
 */
bool GpuHashTable::insertBatch(int* keys, int* values, int numKeys) {
	unsigned long long max_items;
	// Copy max_items from device to host.
	cudaMemcpy(&max_items, &(dev_hash->max_items), sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	printf("ORIGINAL MAX ITEMS %d\n", max_items); // Debug print.

	// Check if a reshape is needed before insertion.
	if (!check_l(numKeys)) {
		OK = 1; // Set flag indicating reshape happened due to load.
		// Reshape the hash table to accommodate new items.
		reshape((max_items + numKeys)); // Target capacity based on current + new items.
	}
	else OK = 0; // No reshape needed.

	int* dev_keys; // Device pointer for keys.
	cudaMalloc((void**)&dev_keys, numKeys * sizeof(int));
	int* dev_values; // Device pointer for values.
	cudaMalloc((void**)&dev_values, numKeys * sizeof(int));

	// Copy keys and values from host to device.
	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Calculate grid dimensions for kernel_insert.
	int blocks_nr = numKeys / 256;
	if (numKeys % 256 != 0)
		blocks_nr++;

	// Launch kernel to insert the batch of key-value pairs.
	kernel_insert > > (dev_hash, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize(); // Synchronize to ensure insertion completes.

	// Free temporary device memory for keys and values.
	cudaFree(dev_keys);
	cudaFree(dev_values);

	return true;
}

/**
 * @brief CUDA kernel for retrieving a batch of values corresponding to given keys from the hash table.
 *
 * Each thread processes one key. It computes the initial hash, then uses
 * quadratic probing to find the key in the hash table and retrieve its
 * associated value. If a key is not found, the corresponding value in the
 * output array remains unchanged (or default initialized if the output array
 * was cleared beforehand).
 *
 * @param hash A device pointer to the `my_hashTable` descriptor.
 * @param keys A device pointer to an array of integer keys to search for.
 * @param values A device pointer to an array where the retrieved values will be stored.
 * @param numKeys The number of keys in the batch.
 */
__global__ void kernel_get_batch(my_hashTable* hash, int* keys, int* values, int numKeys) {
	// Calculate global thread index.
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= numKeys) return; // Guard against out-of-bounds access.

	int key = keys[index];
	// Only search for valid keys. If key is KEY_INVALID (0), it's not present.
	if (key == KEY_INVALID) {
		values[index] = 0; // Or some other indicator for "not found".
		return;
	}

	int hashed = hash_function(key, hash->max_items); // Compute initial hash.
	my_hashElem item; // Temporary item to read from hash table.

	int i = hash->max_items; // Probing counter.
	// Quadratic probing loop to find the key.
	while (i != 0) {
		item = hash->table[hashed]; // Read item from current bucket.
		if (item.elem_key == key) { // Key found.
			values[index] = item.elem_value; // Store the value.
			return;
		}

		// If collision or not found, compute next probe location using quadratic probing.
		hashed = (hashed + i * i) % hash->max_items;
		i--; // Decrement probe counter.
	}
	// If loop finishes, key not found. `values[index]` would remain its initial value.
}


/**
 * @brief Retrieves a batch of values for given keys from the GPU hash table.
 *
 * This host-side function copies input keys to the device, launches the
 * `kernel_get_batch` kernel, copies results back to the host, and manages
 * device memory.
 *
 * @param keys A pointer to an array of integer keys on the host to search for.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to a newly allocated host array containing the retrieved values.
 *         The caller is responsible for freeing this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int* dev_keys; // Device pointer for input keys.
	cudaMalloc((void**)&dev_keys, numKeys * sizeof(int));
	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int* dev_values; // Device pointer for output values.
	cudaMalloc((void**)&dev_values, numKeys * sizeof(int));

	int* local_values = (int*)malloc(numKeys * sizeof(int)); // Host array for results.
	memset(local_values, 0, numKeys * sizeof(int)); // Initialize host result array to zeros.

	// Calculate grid dimensions for kernel_get_batch.
	int blocks_nr = numKeys / 256;
	if (numKeys % 256 != 0)
		blocks_nr++;

	// Launch kernel to retrieve values for the batch of keys.
	kernel_get_batch > > (dev_hash, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize(); // Synchronize to ensure retrieval completes.

	// Copy retrieved values from device to host.
	cudaMemcpy(local_values, dev_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	// Free temporary device memory.
	cudaFree(dev_keys);
	cudaFree(dev_values);

	return local_values;
}


/**
 * @brief Computes and returns the current load factor of the GPU hash table.
 *
 * This host-side function retrieves the current number of items and the maximum
 * capacity from the device and calculates their ratio.
 *
 * @return The current load factor as a float.
 */
float GpuHashTable::loadFactor() {
	float loadFactor;
	unsigned long long max_items, curr_nr_items;

	// Copy max_items and curr_nr_items from device to host.
	cudaMemcpy(&max_items, &dev_hash->max_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(&curr_nr_items, &dev_hash->curr_nr_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	loadFactor = 1.0f * curr_nr_items /max_items;

	return loadFactor; 
}


// --- Content from gpu_hashtable.hpp ---

#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

/**
 * @def KEY_INVALID
 * @brief Defines the value representing an invalid or empty key in the hash table.
 *        Used to indicate an unoccupied bucket.
 */
#define	KEY_INVALID		0

/**
 * @def DIE(assertion, call_description)
 * @brief Macro for robust error handling.
 *
 * If the `assertion` is true (indicating an error condition), this macro prints
 * the file and line number, a user-provided description, and the system error
 * message (`strerror(errno)`), then exits the program.
 * Functional Utility: Simplifies error checking and provides informative exit messages.
 *
 * @param assertion A boolean expression that, if true, indicates an error.
 * @param call_description A string describing the context of the error.
 */
#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)

/**
 * @brief Global constant array of prime numbers.
 *
 * This list of prime numbers is likely used in hash functions (`hash1`, `hash2`, `hash3`)
 * or for determining optimal hash table sizes/capacities to minimize collisions.
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




int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

typedef struct {
	int elem_value;
	int elem_key;
}my_hashElem;

typedef struct {
	my_hashElem* table;
	unsigned long long max_items;
	unsigned long long curr_nr_items;
}my_hashTable;

#define MAX_LOAD 85
#define MIN_LOAD 75




class GpuHashTable
{
public:
	GpuHashTable(int size);
	my_hashTable* dev_hash;
	int OK = 0;
	void reshape(int sizeReshape);

	bool check_l(unsigned long long batchSize);
	bool insertBatch(int* keys, int* values, int numKeys);
	int* getBatch(int* key, int numItems);

	float loadFactor();
	void occupancy();
	void print(string info);

	~GpuHashTable();
};

#endif

