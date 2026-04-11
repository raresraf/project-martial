/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA.
 *
 * This file contains the implementation of a parallel hash table designed to run on NVIDIA GPUs.
 * The hash table uses open addressing with linear probing to resolve collisions. All operations,
 * including insertions, retrievals, and resizing, are handled by CUDA kernels, leveraging atomic
 * operations (atomicCAS) to ensure thread-safe concurrent access to the hash map.
 *
 * The implementation includes:
 * - A host-side C++ class `GpuHashTable` to manage the hash table's lifecycle,
 *   including memory allocation, data transfers, and kernel launches.
 * - CUDA kernels for `insert`, `get`, and `remake_hash` (for resizing).
 * - A dynamic resizing mechanism based on a load factor threshold.
 *
 * The code also contains remnants of a testing framework, including an inclusion of "test_map.cpp"
 * and a CPU-side hash implementation, likely for verification or benchmarking purposes.
 */

#include 
#include 
#include 
#include 
#include 
#include 
#include "gpu_hashtable.hpp"

// Defines the number of threads per block for CUDA kernels.
#define THREADS 1024
// A prime number used as a multiplier in the hash function.
#define A 129607llu
// A large prime number used as a modulus in the hash function to distribute keys.
#define B 5351951779llu

/**
 * @brief Computes a hash value for a given integer key.
 * @param data The integer key to hash.
 * @param limit The size of the hash table, used to constrain the hash value.
 * @return The computed hash index within the range [0, limit - 1].
 *
 * This is a device function for use within CUDA kernels. It employs a simple
 * multiplicative hashing scheme.
 */
__device__ int my_hash(int data, int limit) {
	return ((long)abs(data) * A) % B % limit;
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param hashtable The device-side hash table structure.
 * @param keys A device pointer to an array of keys to insert.
 * @param values A device pointer to an array of corresponding values to insert.
 * @param entries The total number of key-value pairs in the batch.
 *
 * Each thread in the grid is responsible for inserting one key-value pair. The kernel
 * uses linear probing to find an available slot. Atomic Compare-And-Swap (atomicCAS)
 * is used to handle race conditions, ensuring that each slot is claimed by only one thread.
 * If a key already exists, its value is updated (upsert).
 */
__global__ void insert(hashtable_t hashtable, int *keys, int *values, int entries) {
	int index;
	int current_key;
	int hash;
	int i;

	// Determine the unique index for this thread to identify which key-value pair to process.
	index = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Ensure the thread's index is within the bounds of the input arrays.
	if (index < entries) {
		
		current_key = keys[index];

		// Calculate the initial hash position for the current key.
		hash = my_hash(current_key, hashtable.size);

		// Start linear probing from the initial hash position to the end of the table.
		for(i = hash; i < hashtable.size; i++) {

			// Atomically check if the key already exists. If so, update its value.
			// This implements an "upsert" functionality.
			if (atomicCAS(&hashtable.map[i].key, current_key, current_key) == current_key) {
				hashtable.map[i].value = values[index];
				return;
			}
			// Atomically try to claim an empty slot (marked by KEY_INVALID).
			if (atomicCAS(&hashtable.map[i].key, KEY_INVALID, current_key) == KEY_INVALID) {
				// If successful, write the value and exit.
				hashtable.map[i].value = values[index];
				return;
			}
		}

		// If the end of the table is reached, wrap around and continue probing from the beginning.
		for(i = 0; i < hash; i++) {
			// Atomically check if the key already exists. If so, update its value.
			if (atomicCAS(&hashtable.map[i].key, current_key, current_key) == current_key) {
				hashtable.map[i].value = values[index];
				return;
			}
			// Atomically try to claim an empty slot.
			if (atomicCAS(&hashtable.map[i].key, KEY_INVALID, current_key) == KEY_INVALID) {
				hashtable.map[i].value = values[index];
				return;
			}
		}
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param hashtable The device-side hash table structure.
 * @param keys A device pointer to an array of keys to look up.
 * @param values A device pointer to an array where the found values will be stored.
 * @param numKeys The number of keys to look up.
 *
 * Each thread searches for one key using the same linear probing logic as the insert kernel.
 * If a key is found, its corresponding value is written to the output `values` array.
 * If not found, the value remains at its initial state (typically 0).
 */
__global__ void get (hashtable_t hashtable, int *keys, int *values, int numKeys) {
	int index;
	int current_key;
	int hash;
	int i;
	
	// Each thread gets a unique index to identify which key it is responsible for.
	index = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Check if the index is within the bounds of the number of keys to process.
	if (index < numKeys){
		
		current_key = keys[index];

		// Calculate the starting hash position for the key.
		hash = my_hash(keys[index], hashtable.size);

		// Perform linear probing, starting from the hash position to the end of the table.
		for (i = hash; i<hashtable.size; i++) {
			
			// Use atomicCAS in a non-mutating way to atomically read and compare the key.
			// If the key at the current slot matches the key we're looking for, we've found it.
			if (atomicCAS(&hashtable.map[i].key, current_key, current_key) == current_key) {
					values[index] = hashtable.map[i].value;
					return;
			}
		}

		// If the end of the table is reached, wrap around and continue probing from the beginning.
		for (i = 0; i<hash; i++) {
			
			// Continue the search for the matching key.
			if (atomicCAS(&hashtable.map[i].key, current_key, current_key) == current_key) {
					values[index] = hashtable.map[i].value;
					return;
			}
		}
	}
}

/**
 * @brief CUDA kernel to rehash the contents of an old hash table into a new, larger one.
 * @param old_hashtable The source hash table.
 * @param new_hashtable The destination hash table.
 *
 * Each thread is responsible for one slot in the old hash table. If the slot is not empty,
 * the thread takes its key-value pair and inserts it into the new hash table. This is a
 * critical part of the dynamic resizing process.
 */
__global__ void remake_hash(hashtable_t old_hashtable, hashtable_t new_hashtable) {
	int index;
	int key;
	int new_hash;
	int old_key;
	bool added; // Note: This local variable is not initialized, which could lead to unpredictable behavior.
	int i;

	// Each thread processes one entry from the old hash table.
	index = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Ensure the index is within the bounds of the old hash table size.
	if (index < old_hashtable.size) {
		
		// Retrieve the key from the old table's slot.
		key = old_hashtable.map[index].key;

		// If the slot is empty, there is nothing to do.
		if (key == KEY_INVALID)
			return;

		// Re-hash the key for the new table's size.
		new_hash = my_hash(key, new_hashtable.size);
				
		// Perform linear probing in the new table to find a slot for the old entry.
		// Pre-condition: `added` is uninitialized. The logic assumes it's false.
		for(i = new_hash; i < new_hashtable.size; i++) {
			if (added == false) {
				// Atomically attempt to insert the key into an empty slot in the new table.
				old_key = atomicCAS(&new_hashtable.map[i].key, KEY_INVALID, key);
				if (old_key == KEY_INVALID) {
					new_hashtable.map[i].value = old_hashtable.map[index].value;
					added = true;
					return;
				}
			}
		}

		// Wrap around and continue probing from the start of the new table if needed.
		for(i = 0; i < new_hash; i++) {
			if (added == false) {
				old_key = atomicCAS(&new_hashtable.map[i].key, KEY_INVALID, key);
				if (old_key == KEY_INVALID) {
					new_hashtable.map[i].value = old_hashtable.map[index].value;
					added = true;
					break;
				}
			}
		}
	}
}


/**
 * @brief Constructs a GpuHashTable object.
 * @param size The initial number of buckets for the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	hashtable.size = size; 
	hashtable.map = NULL;
	
	// Allocate memory on the GPU for the hash map entries.
	if (cudaMalloc((void**)(&hashtable.map), size * sizeof(entry_t)) != cudaSuccess) {
		printf("[INIT] Cuda malloc failed
");
		return;
	}

	// Initialize all keys to KEY_INVALID (0) to mark them as empty.
	cudaMemset(hashtable.map, 0, size * sizeof(entry_t));
}


/**
 * @brief Destroys the GpuHashTable object, freeing GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashtable.map);
}


/**
 * @brief Resizes the hash table to a new capacity and rehashes all existing elements.
 * @param numBucketsReshape The new size (number of buckets) for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hashtable_t new_hashtable;
	int nr_blocks;

	new_hashtable.size = numBucketsReshape;

	// Allocate a new, larger memory region on the GPU.
	if (cudaMalloc(&new_hashtable.map, numBucketsReshape * sizeof(entry_t)) != cudaSuccess) {
		printf("[RESHAPE] Cuda malloc failed
");
		return;
	}

	// Initialize the new hash table memory.
	cudaMemset(new_hashtable.map, 0, numBucketsReshape * sizeof(entry_t));
	
	// Calculate the number of blocks needed to launch the remake_hash kernel.
	int ct;
	nr_blocks = hashtable.size / THREADS;
	(hashtable.size % THREADS != 0) ? ct = 1 : ct = 0;  
	nr_blocks += ct;
	
	// Launch the kernel to rehash elements from the old table to the new one.
	remake_hash>>(hashtable, new_hashtable);

	cudaDeviceSynchronize();
	
	// Free the old hash table's GPU memory.
	cudaFree(hashtable.map);

	// Replace the old hash table with the new one.
	hashtable = new_hashtable;
 }


/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys A host pointer to an array of keys.
 * @param values A host pointer to an array of values.
 * @param numKeys The number of pairs to insert.
 * @return True on success, false on failure.
 *
 * This method manages the batch insertion process, including checking the load factor
 * and triggering a reshape if necessary, transferring data to the GPU, and launching
 * the insert kernel.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *d_keys; 
	int *d_values; 
	float current_size;
	float currLoadFactor;
	int nr_blocks;

	// Allocate temporary memory on the GPU for the keys and values.
	if (cudaMalloc(&d_keys, numKeys*sizeof(int)) != cudaSuccess) {
		printf("[INSERT] Cuda malloc failed
");
		return false;
	}
	if (cudaMalloc(&d_values, numKeys*sizeof(int)) != cudaSuccess) {
		printf("[INSERT] Cuda malloc failed
");
		return false;
	}

	// Update the total number of elements in the table.
	actualSize += numKeys;

	current_size = float(actualSize);

	currLoadFactor = loadFactor() + (float)numKeys / (float)hashtable.size;

	// Check if the load factor exceeds the threshold (0.8). If so, resize the table.
	if (currLoadFactor >= 0.8)
		reshape(int(current_size / 0.8));

	// Copy keys and values from host to device.
	if (cudaMemcpy(d_keys, keys, numKeys*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("[INSERT] Cuda memcpy failed
");
		return false;
	}
	if (cudaMemcpy(d_values, values, numKeys*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("[INSERT] Cuda memcpy failed
");
		return false;
	}

	// Calculate grid dimensions for the kernel launch.
	int ct;
	nr_blocks = hashtable.size / THREADS;
	(hashtable.size % THREADS != 0) ? ct = 1 : ct = 0;  
	nr_blocks += ct;

	// Launch the insert kernel.
	insert>>(hashtable, d_keys, d_values, numKeys);

	// Wait for the kernel to complete.
	cudaDeviceSynchronize();
	
	// Free the temporary device memory.
	cudaFree(d_keys);
	cudaFree(d_values);

	return true;
}


/**
 * @brief Retrieves values for a batch of keys.
 * @param keys A host pointer to an array of keys to look up.
 * @param numKeys The number of keys in the batch.
 * @return A host pointer to an array containing the retrieved values. The caller is responsible for freeing this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *d_keys;
	int *d_values;
	// Allocate host memory for the results.
	int *result = (int*)malloc(numKeys * sizeof(int));
	int nr_blocks;

	// Allocate temporary device memory for keys and the result values.
	if (cudaMalloc(&d_keys, numKeys*sizeof(int)) != cudaSuccess) {
		printf("[GET] Cuda malloc failed
");
		return NULL;
	}

	if (cudaMalloc(&d_values, numKeys*sizeof(int)) != cudaSuccess) {
		printf("[GET] Cuda malloc failed
");
		return NULL;
	}

	// Copy keys from host to device.
	if (cudaMemcpy(d_keys, keys, numKeys*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("[GET] Cuda memcpy failed
");
		return NULL;
	}

	// Initialize the device-side values array to 0.
	cudaMemset(d_values, 0, numKeys * sizeof(int));

	// Calculate grid dimensions for kernel launch.
	int ct;
	nr_blocks = hashtable.size / THREADS;
	(hashtable.size % THREADS != 0) ? ct = 1 : ct = 0;  
	nr_blocks += ct;
	
	// Launch the get kernel to perform the lookup.
	get>>(hashtable, d_keys, d_values, numKeys);

	// Wait for the kernel to finish.
	cudaDeviceSynchronize();

	// Copy the results from device back to host.
	cudaMemcpy(result, d_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	
	// Free temporary device memory.
	cudaFree(d_keys);
	cudaFree(d_values);

	return result;
}


/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor, defined as the ratio of actual elements to the total capacity.
 */
float GpuHashTable::loadFactor() {
	if (hashtable.size == 0) {
		return 0.f;
	} else {
		// Invariant: `actualSize` is the number of elements, `hashtable.size` is the capacity.
		return (float)((float)actualSize)/((float)hashtable.size); 
	}
}


// --- Test Interface Macros ---
// These macros appear to be part of a simplified API for a testing harness.

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

// Including a .cpp file is highly unconventional and suggests this is part of a single-file-style benchmark or test.
#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

// Defines the value used to represent an invalid or empty key slot in the hash table.
#define	KEY_INVALID		0

// A macro for robust error checking.
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
 * @brief A list of pre-calculated prime numbers.
 *
 * This list is likely used to provide good modulus values or multipliers for various
 * hashing schemes. The use of large primes helps in reducing collisions and distributing
 * keys more uniformly across the hash space.
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



// A series of simple hash functions for host-side use, likely for testing or comparison.
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

// Defines a single entry (key-value pair) in the hash table.
struct entry_t {
	int key;
	int value;
};

// Defines the structure of the hash table as seen by the device.
struct hashtable_t {
	int size;
	entry_t *map;
};



// The host-side interface for the GPU hash table.
class GpuHashTable
{
	int actualSize; // Tracks the number of elements currently in the hash table.
	hashtable_t hashtable; 

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
