/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 *
 * @details
 * This file provides a GPU-based hash table that supports batch insertion,
 * batch retrieval, and dynamic resizing (reshaping). It uses a struct-of-arrays
 * approach, where key-value pairs are stored in an array of 'elem' structs.
 *
 * @algorithm
 * The collision resolution strategy is linear probing. When a hash collision
 * occurs, the algorithm probes the next sequential bucket until an empty slot is
 * found. Concurrency during insertion and reshaping is managed using atomic
 * Compare-And-Swap (atomicCAS) operations to ensure that threads do not overwrite
 * each other's data in a race condition. The load factor is monitored, and if it
 * exceeds a threshold, the table is resized and all elements are rehashed.
 *
 * @note This file has an unusual structure, with implementation followed by
 * what appears to be embedded header content. The documentation follows this
 * structure.
 */

#include 
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief CUDA kernel to rehash elements from an old table to a new, larger one.
 * @param ht The old hash table structure.
 * @param newHt The new, resized hash table structure.
 *
 * @details Each thread takes one element from the old table and re-inserts it into
 * the new table. It uses linear probing and atomicCAS to find an empty slot.
 */
__global__ void kernel_reshape(hashtable ht, hashtable newHt) {

	// Thread Indexing: Each thread gets a unique index to process one element from the old table.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < ht.size) {

		int oldKey = ht.elements[idx].key;

		// Pre-condition: Skip empty slots in the old table.
		if (oldKey == KEY_INVALID)
			return;

		// Re-calculate the hash for the key based on the new table's size.
		int hash_value = my_hash(oldKey, newHt.size);

		int rez, valid_hash;
		// Block Logic: Linear probing loop to find an empty slot in the new table.
		do {
			// Synchronization: Atomically attempt to claim an empty slot by swapping
			// KEY_INVALID with the current key.
			rez = atomicCAS(&(newHt.elements[hash_value].key),
				KEY_INVALID, oldKey);
			
			valid_hash = hash_value;
			
			// Collision: Move to the next slot.
			hash_value += 1;
			// Wrap around if the end of the table is reached.
			if(hash_value >= newHt.size)
				hash_value = hash_value % newHt.size;

		} while (rez != KEY_INVALID); // Invariant: Loop until an empty slot is successfully claimed.
		
		// Once a slot is claimed, write the corresponding value.
		newHt.elements[valid_hash].value = ht.elements[idx].value;
	}
}

/**
 * @brief CUDA kernel for batch insertion of key-value pairs.
 * @param deviceKeys A device pointer to the array of keys to insert.
 * @param deviceValues A device pointer to the array of values to insert.
 * @param numKeys The number of pairs to insert.
 * @param ht The hash table structure.
 *
 * @details Each thread handles one key-value pair. It uses linear probing and
 * atomicCAS to find an empty slot or update an existing key.
 */
__global__ void kernel_insert(int *deviceKeys, int *deviceValues, int numKeys, hashtable ht) {

	// Thread Indexing: Each thread processes one key-value pair from the batch.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numKeys) {

		int oldKey = deviceKeys[idx];
		// Pre-condition: Do not insert invalid keys.
		if (oldKey == KEY_INVALID)
			return;

		// Calculate the initial hash position.
		int hash_value = my_hash(oldKey, ht.size);

		int rez, valid_hash;
		// Block Logic: Linear probing loop.
		do {
			// Synchronization: Atomically attempt to claim an empty slot or find the existing key.
			rez = atomicCAS(&(ht.elements[hash_value].key),
				KEY_INVALID, oldKey);
			
			valid_hash = hash_value;
			
			// Collision: Probe the next slot.
			hash_value += 1;
			if(hash_value >= ht.size)
				hash_value = hash_value % ht.size;

		// Invariant: Loop until we find an empty slot (rez == KEY_INVALID) or the same key (rez == oldKey).
		} while (rez != KEY_INVALID && rez != oldKey);
		
		// Write the value to the slot we found or claimed.
		ht.elements[valid_hash].value = deviceValues[idx];
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @details Each thread is responsible for looking up one key. It uses linear probing
 * to scan for the key.
 */
__global__ void kernel_get(int *deviceKeys, int *deviceValues, int numKeys, hashtable ht) {

	// Thread Indexing: Each thread is assigned one key to look up.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numKeys) {

		int oldKey = deviceKeys[idx];

		if (oldKey == KEY_INVALID)
			return;

		int hash_value = my_hash(oldKey, ht.size);
		
		int index = 0, valid_value, found = 0;
		// Block Logic: Linear probing search loop.
		// Invariant: Scan up to ht.size times to check every possible slot.
		while (index < ht.size){
			// Memory Access: Reading from global device memory.
			if(ht.elements[hash_value].key == oldKey) {
				// Key found.
				valid_value = ht.elements[hash_value].value;
				found = 1; 
				break;
			}
			
			// Probe the next slot.
			hash_value += 1;
			if(hash_value >= ht.size)
				hash_value = hash_value % ht.size;
			index++;
		}

		if(found == 1) {
			// Write the found value to the results array.
			deviceValues[idx] = valid_value;
		}
	}
}

/**
 * @brief Constructor: initializes the hash table on the GPU.
 */
GpuHashTable::GpuHashTable(int size) {

	cudaError_t res;

	ht.size = size;
	ht.nrElems = 0;
	
	// Memory Allocation: Allocate device memory for the elements array.
	res = cudaMalloc(&ht.elements, size * sizeof(elem));
	// Error Handling: Check for CUDA errors.
	if (res != cudaSuccess) {
		cout << "[init] Memory allocation error
";
		return;
	}
	
	// Initialize the allocated memory to zero.
	res = cudaMemset(ht.elements, 0, size * sizeof(elem));
	if (res != cudaSuccess) {
		cout << "[init] Memset error
";
		return;
	}
}

/**
 * @brief Destructor: frees allocated GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaError_t res;
	
	res = cudaFree(ht.elements);
	if (res != cudaSuccess) {
		cout << "[destroy] Free error
";
		return;
	}
}

/**
 * @brief Resizes the hash table to a new capacity and rehashes all elements.
 */
void GpuHashTable::reshape(int numBucketsReshape) {

	cudaError_t res;
	hashtable newHt;

	newHt.size = numBucketsReshape;
	newHt.nrElems = 0;
	
	// Memory Allocation: Allocate a new, larger table on the device.
	res = cudaMalloc(&newHt.elements, numBucketsReshape * sizeof(elem));
	if (res != cudaSuccess) {
		cout << "[reshape] Memory allocation error
";
		return;
	}
	
	res = cudaMemset(newHt.elements, 0, numBucketsReshape * sizeof(elem));
	if (res != cudaSuccess) {
		cout << "[reshape] Memset error
";
		return;
	}

	// Kernel Launch Configuration.
	int blocks = (ht.size % NR_THREADS != 0) ?
		ht.size / NR_THREADS + 1 : ht.size / NR_THREADS;
	kernel_reshape>>(ht, newHt);

	// Synchronization: Wait for the reshape kernel to complete.
	res = cudaDeviceSynchronize();
	if (res != cudaSuccess) {
		cout << "[reshape] Syncronize error
";
		return;
	}

	// Free the old hash table's memory.
	res = cudaFree(ht.elements);
	if (res != cudaSuccess) {
		cout << "[reshape] Free error
";
		return;
	}
	
	// Update the main hash table struct to point to the new data.
	newHt.nrElems = ht.nrElems;
	ht = newHt;
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {

	// Block Logic: Check if the load factor will exceed the maximum threshold.
	// If so, trigger a reshape.
	if (numKeys + ht.nrElems > ht.size || 
		(float)(ht.nrElems + numKeys) / ht.size >= LOAD_FACTOR_MAX) {
			reshape((int)((ht.nrElems + numKeys) / LOAD_FACTOR_MIN));
	}

	cudaError_t res;
	
	// Memory Transfer: Allocate temporary device memory for the batch.
	int *deviceKeys, *deviceValues;
	res = cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	if (res != cudaSuccess) {
		cout << "[insert] Memory allocation error
";
		return false;
	}
	res = cudaMalloc(&deviceValues, numKeys * sizeof(int));
	if (res != cudaSuccess) {
		cudaFree(deviceKeys);
		cout << "[insert] Memory allocation error
";
		return false;
	}

	// Memory Transfer: Copy keys and values from host to device.
	res = cudaMemcpy(deviceKeys, keys, numKeys *
		sizeof(int), cudaMemcpyHostToDevice);
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		cudaFree(deviceKeys);
		cout << "[insert] Memory allocation error
";
		return false;
	}
	res = cudaMemcpy(deviceValues, values,
		numKeys * sizeof(int), cudaMemcpyHostToDevice);
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		cudaFree(deviceKeys);
		cout << "[insert] Memcpy error
";
		return false;
	}

	// Kernel Launch Configuration.
	int blocks = (numKeys % NR_THREADS != 0) ?
		numKeys / NR_THREADS + 1 : numKeys / NR_THREADS;
	kernel_insert>>
		(deviceKeys, deviceValues, numKeys, ht);

	// Synchronization: Wait for the insertion kernel to finish.
	res = cudaDeviceSynchronize();
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		cout << "[insert] Syncronize error
";
		return false;
	}

	// Free temporary device memory.
	res = cudaFree(deviceKeys);
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		cout << "[insert] Free error
";
		return false;
	}
	res = cudaFree(deviceValues);
	if (res != cudaSuccess) {
		cout << "[insert] Free error
";
		return false;
	}
	
	// Update the element count.
	ht.nrElems += numKeys;

	return true;
}

/**
 * @brief Retrieves values for a batch of keys from the hash table.
 * @return A host pointer to an array containing the retrieved values. Caller must free this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	
	cudaError_t res;
	
	int *deviceKeys, *deviceValues, *values;
	// Memory Allocation: Allocate temporary device memory and host memory for results.
	res = cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	if (res != cudaSuccess) {
		cout << "[get] Memory allocation error
";
		return NULL;
	}
	res = cudaMalloc(&deviceValues, numKeys * sizeof(int));
	if (res != cudaSuccess) {
		cout << "[get] Memory allocation error
";
		return NULL;
	}
	values = (int *) malloc(numKeys * sizeof(int));
	if (!values) {
		cout << "[get] Memory allocation error
";
		return NULL;
	}

	// Memory Transfer: Copy keys from host to device.
	res = cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int),
		cudaMemcpyHostToDevice);
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		cudaFree(deviceKeys);
		free(values);
		cout << "[get] Memcpy error1
";
		return NULL;
	}
	
	// Kernel Launch Configuration.
	int blocks = (numKeys % NR_THREADS != 0) ?
		numKeys / NR_THREADS + 1 : numKeys / NR_THREADS;
	kernel_get>>
		(deviceKeys, deviceValues, numKeys, ht);

	// Synchronization: Wait for the get kernel to complete.
	res = cudaDeviceSynchronize();
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		cudaFree(deviceKeys);
		free(values);
		cout << "[get] Syncronize error
";
		return NULL;
	}

	// Memory Transfer: Copy results from device back to host.
	res = cudaMemcpy(values, deviceValues,
		numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		cudaFree(deviceKeys);
		free(values);
		cout << "[get] Memcpy error3
";
		return NULL;
	}

	// Free all allocated temporary memory.
	res = cudaFree(deviceKeys);
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		free(values);
		cout << "[get] Free error
";
		return NULL;
	}
	res = cudaFree(deviceValues);
	if (res != cudaSuccess) {
		free(values);
		cout << "[get] Free error
";
		return NULL;
	}
	
	return values;
}

/**
 * @brief Calculates and returns the current load factor.
 */
float GpuHashTable::loadFactor() {

	if(ht.size == 0)
		return 0;
	
	return (float)(ht.nrElems) / ht.size;
}


// --- Convenience macros for a simplified external API ---
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

#define	KEY_INVALID		0      // Value for an empty/invalid key slot.
#define NR_THREADS		1024   // Default number of threads per block for kernel launches.
#define LOAD_FACTOR_MAX		0.99   // Max load factor before triggering a reshape.
#define LOAD_FACTOR_MIN		0.8    // Target load factor after a reshape.

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
	
// A list of large prime numbers for use in hashing functions.
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



/**
 * @brief The primary hash function used by the kernels on the device.
 * @return The calculated hash index.
 */
__device__ int my_hash(int data, int limit) {
	return ((long)abs(data) * 52679969llu) % 28282345988300791llu % limit;
}

/**
 * @struct elem
 * @brief Represents a single key-value pair (a bucket) in the hash table.
 */
typedef struct elem {
	int key;
	int value;
} elem;

/**
 * @struct hashtable
 * @brief The main control structure for the hash table, passed to kernels.
 */
typedef struct hashtable {
	int size;      // Current capacity of the hash table.
	int nrElems;   // Current number of elements stored.
	elem *elements; // Device pointer to the array of elements.
} hashtable;



/**
 * @class GpuHashTable
 * @brief Host-side class definition for managing the GPU hash table.
 */
class GpuHashTable
{
	public:
		hashtable ht; // The host-side representation of the hash table struct.

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
