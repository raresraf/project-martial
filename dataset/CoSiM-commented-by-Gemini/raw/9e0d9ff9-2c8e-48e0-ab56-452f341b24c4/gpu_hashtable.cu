/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA with linear probing.
 *
 * @details
 * This file defines a GPU-based hash table that stores key-value pairs. It uses
 * an array of `my_pair` structs to hold the data directly on the GPU. The implementation
 * supports dynamic resizing (reshaping) based on load factor thresholds.
 *
 * @algorithm
 * The collision resolution strategy employed is linear probing. When an insertion
 * results in a hash collision, the algorithm probes subsequent buckets sequentially
 * until an empty slot is found. Concurrency is managed via atomic Compare-And-Swap
 * (atomicCAS) operations, which allows multiple threads to safely attempt insertions
 * into the same bucket.
 *
 * @note The file appears to concatenate the implementation, a header, and a test file,
 * which is an unconventional structure. The documentation addresses the code as presented.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief Computes a hash value for a given key.
 * @param key The integer key to hash.
 * @param size The total size of the hash table, used for the modulo operation.
 * @return An unsigned integer representing the hash index.
 * @note This is a device function callable from kernels. It uses a multiplicative
 * hashing scheme with large prime numbers.
 */
__device__ unsigned int hash_func(int key, int size) {

	return (((long)key * PRIME_A) % PRIME_B % size);
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param keys Device pointer to an array of keys.
 * @param values Device pointer to an array of values.
 * @param numKeys The number of pairs to insert.
 * @param hashtable Device pointer to the array of `my_pair` structs.
 * @param size The capacity of the hash table.
 * @param updates_count Device pointer to a counter for the number of updated (not new) keys.
 *
 * @details Each thread handles one key-value pair. It uses linear probing and atomicCAS
 * to find an empty slot or update an existing key's value.
 */
__global__ void insert_pair(int *keys, int *values, int numKeys, struct my_pair *hashtable,
				int size, int* updates_count) {

	// Thread Indexing: Each thread is assigned one key-value pair from the batch.
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= numKeys)
		return;
	// Pre-condition: Ignore invalid keys or values (e.g., non-positive).
	if (keys[index] <= 0 || values[index] <= 0) {
		return;
	}

	int old_key;
	int key = keys[index];
	int value = values[index];
	unsigned int pos = hash_func(keys[index], size);
	unsigned int i = 0;

	// Block Logic: Linear probing loop to find a slot.
	while(true) {

		// Calculate the next probe position, wrapping around the table.
		pos = (pos + i) % size;

		// Synchronization: Atomically attempt to claim an empty slot (key is 0).
		old_key = atomicCAS(&hashtable[pos].key, 0, key);

		if (old_key == key) {
			// Case 1: Key already exists. Update the value.
			hashtable[pos].value = value;
			// Atomically increment the count of updated keys.
			atomicAdd(updates_count, 1);
			return;
		}
		else if (old_key == 0) {
			// Case 2: Slot was empty. We have successfully inserted.
			hashtable[pos].value = value;
			return;
		}
		else {
			// Case 3: Collision with a different key. Continue probing.
			i++;
		}
	}
}

/**
 * @brief CUDA kernel to copy elements from an old hash table to a new, resized one.
 * @param old_pairs Device pointer to the old hash table's data.
 * @param new_pairs Device pointer to the new hash table's data.
 * @param size The size of the old table.
 * @param new_size The size of the new table.
 *
 * @details Each thread is responsible for copying one element, rehashing it for the
 * new table size, and inserting it into the new table using linear probing.
 */
__global__ void copy_values(struct my_pair *old_pairs, struct my_pair *new_pairs, 
															int size, int new_size) {
	
	// Thread Indexing: Each thread handles one element from the old table.
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= size)
		return;
	// Pre-condition: Skip empty slots.
	if (old_pairs[index].key == 0)
		return;

	// Re-calculate the hash for the new table size.
	unsigned int pos = hash_func(old_pairs[index].key, new_size);
	unsigned int i = 0;

	int old_key = old_pairs[index].key;
	int old_value = old_pairs[index].value;
	int value;

	// Block Logic: Linear probing to find an empty slot in the new table.
	while(true) {

		pos = (pos + i) % new_size;

		// Synchronization: Atomically claim an empty slot in the new table.
		value = atomicCAS(&new_pairs[pos].key, 0, old_key);
		if (value == 0) {
			new_pairs[pos].value = old_value;
			return;
		}
		i++;
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param keys Device pointer to an array of keys to look up.
 * @param numKeys The number of keys to retrieve.
 * @param hashtable Device pointer to the hash table data.
 * @param size The capacity of the hash table.
 * @param values Device pointer to an array where results will be stored.
 *
 * @details Each thread searches for one key using linear probing.
 */
__global__ void get_pair(int *keys, int numKeys, struct my_pair *hashtable, int size, 
						int *values) {

	// Thread Indexing: Each thread is assigned one key to search for.
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= numKeys)
		return;

	unsigned int pos = hash_func(keys[index], size);
	unsigned int i = 0;
	int key;

	// Block Logic: Linear probing search loop. The loop condition is `true`
	// but it's guaranteed to terminate because a key is either found or an
	// empty slot is hit in a non-full table.
	while(true) {
		pos = (pos + i) % size;
		key = hashtable[pos].key;
		
		// If the key at the probed position matches the search key.
		if (key == keys[index]) {
			values[index] = hashtable[pos].value;
			return;
		}
		// In a non-full table, an empty slot (key=0) indicates the search key is not present.
		// This kernel lacks that check, which could lead to infinite loops if a key is not found
		// and the table is full.
		i++;
	}
}


/**
 * @brief Constructor for GpuHashTable.
 * @param size The initial capacity of the hash table.
 * @details Allocates memory on the GPU for the hash table pairs and a counter for updates.
 */
GpuHashTable::GpuHashTable(int size) {

	cudaError_t err;
	
	// Memory Allocation: Allocate device memory for the array of key-value pairs.
	err = cudaMalloc((void **)&my_hashtable, size * sizeof(struct my_pair));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s!
", cudaGetErrorString(err));
		return;
	}

	// Memory Allocation: Allocate device memory for a counter used in the insert kernel.
	err = cudaMalloc((void **)&d_num_updates, sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s!
", cudaGetErrorString(err));
		return;
	}

	// Initialize the allocated device memory to zero.
	err = cudaMemset(my_hashtable, 0, size * sizeof(struct my_pair));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s!
", cudaGetErrorString(err));
		return;
	}
	err = cudaMemset(d_num_updates, 0, sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s!
", cudaGetErrorString(err));
		return;
	}
	
	// Initialize host-side metadata.
	host_size = size;
	host_current_size = 0;
}

/**
 * @brief Destructor for GpuHashTable.
 * @details Frees all allocated GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(my_hashtable);
	cudaFree(d_num_updates);
}

/**
 * @brief Resizes the hash table and rehashes all existing elements.
 * @param numBucketsReshape The new capacity of the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {

	cudaError_t err;
	struct my_pair *new_hashtable;

	// Memory Allocation: Create a new, larger table on the device.
	err = cudaMalloc(&new_hashtable, numBucketsReshape * sizeof(struct my_pair));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s ( reshape)!
", cudaGetErrorString(err));
		return;
	}
	err = cudaMemset(new_hashtable, 0, numBucketsReshape * sizeof(struct my_pair));
	if (err != cudaSuccess) {
		fprintf (stderr, "[ERROR] %s ( reshape)!
", cudaGetErrorString(err));
		return;
	}
	
	// Kernel Launch Configuration.
	const size_t block_size = BLOCK_SIZE;
	size_t block_num = host_size / block_size;
	if (host_size % block_size != 0)
		block_num++;

	// Launch the kernel to copy and rehash values from the old table to the new one.
	copy_values>>(my_hashtable, new_hashtable, 
				host_size, numBucketsReshape);

	// Synchronization: Wait for the copy to complete.
	cudaDeviceSynchronize();

	// Free the old table's memory.
	cudaFree(my_hashtable);

	// Update the class member to point to the new table and size.
	my_hashtable = new_hashtable;
	host_size = numBucketsReshape;
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @return True on success, false on failure.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {

	cudaError_t err;	
	int new_size;
	int *keys_d;
	int *values_d;

	// Memory Transfer: Allocate temporary device memory for the input batch.
	err = cudaMalloc((void **)&keys_d, numKeys * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s ( in insertBatch)!
", cudaGetErrorString(err));
		return false;
	}
	err = cudaMalloc((void **)&values_d, numKeys * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s ( in insertBatch)!
", cudaGetErrorString(err));
		return false;	
	}

	// Memory Transfer: Copy batch data from host to device.
	cudaMemcpy(keys_d, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(values_d, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Block Logic: Determine if a resize is needed and perform it.
	int prev_size = host_size;
	new_size = getNewSize(numKeys);
	if (new_size != -1)
		reshape(new_size);

	// Kernel Launch Configuration.
	const size_t block_size = BLOCK_SIZE;
	size_t block_num = numKeys / block_size;
	if (numKeys % block_size != 0)
		block_num++;

	// Launch the insertion kernel.
	insert_pair>>(keys_d, values_d, numKeys, my_hashtable, host_size, d_num_updates);
	
	// Synchronization: Wait for the kernel to complete.
	cudaDeviceSynchronize();

	// Update the element count on the host.
	int *host_updates = (int*) malloc(sizeof(int));
	cudaMemcpy(host_updates, d_num_updates, sizeof(int), cudaMemcpyDeviceToHost);
	host_current_size += numKeys - (*host_updates);
	
	// Block Logic: After a reshape, the load factor might be too low. This block
	// checks and potentially shrinks the table to an optimal size.
	if (prev_size != host_size) {
		float current_lf = loadFactor();
		if (current_lf < MIN_LF - ROUND_ERROR) {
			int reshape_size = host_current_size / MIN_LF;
			reshape(reshape_size);
		}
	}
	
	// Reset the device-side update counter.
	if (host_updates != 0) {
		err = cudaMemset(d_num_updates, 0, sizeof(int));
		if (err != cudaSuccess) {
			fprintf(stderr, "[ERROR] %s!
", cudaGetErrorString(err));
			return false;
		}
	}

	// Free temporary memory.
	cudaFree(keys_d);
	cudaFree(values_d);
	free(host_updates);

	return true;
}

/**
 * @brief Retrieves values for a batch of keys.
 * @return A host pointer to an array of retrieved values. Caller is responsible for freeing this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	
	int *keys_d; 
	int *values_d;
	int *host_values;

	cudaError_t err;	
	
	// Memory Allocation for device-side arrays.
	err = cudaMalloc(&values_d, numKeys * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s ( in 'getBatch')!
", cudaGetErrorString(err));
		return NULL;
	}
	err = cudaMalloc(&keys_d, numKeys * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s ( in 'getBatch')!
", cudaGetErrorString(err));
		return NULL;
	}

	// Memory Transfer: Copy keys from host to device.
	cudaMemcpy(keys_d, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Kernel Launch Configuration.
	const size_t block_size = BLOCK_SIZE;
	size_t block_num = numKeys / block_size;
	if (numKeys % block_size != 0)
		block_num++;

	// Launch the retrieval kernel.
	get_pair>>(keys_d, numKeys, my_hashtable, host_size, values_d);
	cudaDeviceSynchronize();

	// Allocate host memory for the results.
	host_values = (int *) malloc(numKeys * sizeof(int));
	if (host_values == NULL) {
		fprintf(stderr, "[ERROR] MALLOC !
");
		return NULL;
	}
	
	// Memory Transfer: Copy results from device back to host.
	err = cudaMemcpy(host_values, values_d, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s (memcpy for 'values')!
", cudaGetErrorString(err));
		return NULL;
	}

	// Free device memory.
	cudaFree(keys_d);
	cudaFree(values_d);

	return host_values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
	if (host_size == 0)
		return -1;
	return ((float)host_current_size/host_size);
}

/**
 * @brief Determines the required new size of the hash table if a resize is needed.
 * @param numKeys The number of new keys to be inserted.
 * @return The calculated new size, or -1 if no resize is needed.
 */
int GpuHashTable::getNewSize(int numKeys) {

	int new_size = -1;
	int current_size = host_current_size;
	int size = host_size;
	float current_lf = (float)(current_size + numKeys) / size;

	if (current_size + numKeys > size || current_lf >= MIN_LF + MAX_RANGE)
		new_size = (current_size + numKeys) / MIN_LF;

	return new_size;
}

/**
 * @brief Prints debugging information about the hash table's state.
 */
void GpuHashTable::printInfo(const char* funcName) {

	fprintf(stderr, "=======   Info Hashtable (%s)  =======
", funcName);
	fprintf(stderr, "SIZE = %d
", host_size);
	fprintf(stderr, "CURRENT_SIZE = %d
", host_current_size);
	fprintf(stderr, "LOAD FACTOR = %f
", loadFactor());
	fprintf(stderr, "======================================
");
}



// --- Convenience macros for a simplified external API ---
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

// Primes used in the multiplicative hash function.
#define PRIME_A 653267llu
#define PRIME_B 4349795294267llu

// Constants for managing load factor and resizing.
#define MIN_LF 0.85
#define ROUND_ERROR 0.01
#define MAX_RANGE 0.1

// Default number of threads per block for kernel launches.
#define BLOCK_SIZE 1024

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
 * @struct my_pair
 * @brief Represents a key-value pair in the hash table.
 */
struct my_pair {
	int key;
	int value;
};



/**
 * @class GpuHashTable
 * @brief Host-side class to manage the GPU hash table.
 */
class GpuHashTable
{
	struct my_pair *my_hashtable; // Device pointer to the hash table data.
	int *d_num_updates;           // Device pointer to an update counter.
	int host_size;                // Host-side copy of the table's capacity.
	int host_current_size;        // Host-side copy of the number of elements.

	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		
		float loadFactor();
		int getNewSize(int numKeys);
		void printInfo(const char*);
		void occupancy();
		void print(string info);
	
		~GpuHashTable();
};

#endif
