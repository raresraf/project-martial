
/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using open addressing with linear probing.
 * @details This file defines a hash table that runs on the GPU using CUDA. It is designed
 * for massive parallelism. Collision resolution is handled by linear probing, and thread
 * safety during parallel insertions is managed with atomic operations. The implementation
 * supports dynamic resizing when a specific load factor is exceeded.
 */
#include 
#include 
#include 
#include 
#include 
#include 
#include "gpu_hashtable.hpp"

// --- Constants used for Hashing ---
#define THREADS 1024
#define A 129607llu
#define B 5351951779llu

/**
 * @brief Computes a hash value for a given key.
 * @param data The key to hash.
 * @param limit The size of the hash table, used for the modulo.
 * @return An integer hash value within the range [0, limit-1].
 * @note This is a __device__ function, executed on the GPU by each thread.
 */
__device__ int my_hash(int data, int limit) {
	return ((long)abs(data) * A) % B % limit;
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param hashtable The GPU hash table data structure.
 * @param keys Device pointer to an array of keys to insert.
 * @param values Device pointer to an array of corresponding values.
 * @param entries The total number of pairs to insert.
 * @details Each thread is responsible for inserting one key-value pair. It finds the
 * initial bucket via hashing and then uses linear probing to find a slot.
 */
__global__ void insert(hashtable_t hashtable, int *keys, int *values, int entries) {
	int index;
	int current_key;
	int hash;
	int i;

	// --- Grid-Stride Loop Pattern ---
	// Each thread computes a unique global index to determine which key-value pair to process.
	index = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Guard to ensure the thread operates within the bounds of the input arrays.
	if (index < entries) {
		current_key = keys[index];
		hash = my_hash(current_key, hashtable.size);

		// --- Collision Handling: Linear Probing (Part 1) ---
		// Probe from the initial hash location to the end of the table.
		for(i = hash; i < hashtable.size; i++) {
			// Non-idiomatic Read: Using atomicCAS to check for key existence.
			// While technically safe, a direct read (`if (hashtable.map[i].key == current_key)`)
			// would be more conventional and efficient here.
			if (atomicCAS(&hashtable.map[i].key, current_key, current_key) == current_key) {
				// RACE CONDITION: This is a non-atomic write. If multiple threads
				// try to update the value for the same key, it can lead to lost updates.
				// An `atomicExch` should be used here for safety.
				hashtable.map[i].value = values[index];
				return;
			}
			// Synchronization: Atomically claim an empty slot. This is the correct and
			// safe way to handle parallel insertions into unoccupied buckets.
			if (atomicCAS(&hashtable.map[i].key, KEY_INVALID, current_key) == KEY_INVALID) {
				hashtable.map[i].value = values[index];
				return;
			}
		}

		// --- Collision Handling: Linear Probing (Part 2 - Wrap Around) ---
		// If no slot was found, wrap around and probe from the beginning of the table.
		for(i = 0; i < hash; i++) {
			if (atomicCAS(&hashtable.map[i].key, current_key, current_key) == current_key) {
				// RACE CONDITION: Same potential for lost updates as above.
				hashtable.map[i].value = values[index];
				return;
			}
			if (atomicCAS(&hashtable.map[i].key, KEY_INVALID, current_key) == KEY_INVALID) {
				hashtable.map[i].value = values[index];
				return;
			}
		}
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param hashtable The GPU hash table data structure.
 * @param keys Device pointer to an array of keys to look up.
 * @param values Device pointer to an array where found values will be stored.
 * @param numKeys The number of keys to retrieve.
 */
__global__ void get (hashtable_t hashtable, int *keys, int *values, int numKeys) {
	int index;
	int current_key;
	int hash;
	int i;
	
	// Determine the unique index for this thread.
	index = blockIdx.x * blockDim.x + threadIdx.x;
	
	
	if (index < numKeys){
		current_key = keys[index];
		hash = my_hash(keys[index], hashtable.size);

		// --- Search: Linear Probing (Part 1) ---
		// Probe from the initial hash location to the end of the table.
		for (i = hash; i<hashtable.size; i++) {
			// Unconventional Read: Uses atomicCAS to perform a read-like check. A simple
			// `if (hashtable.map[i].key == current_key)` would be clearer.
			if (atomicCAS(&hashtable.map[i].key, current_key, current_key) == current_key) {
					values[index] = hashtable.map[i].value;
					return;
			}
		}

		// --- Search: Linear Probing (Part 2 - Wrap Around) ---
		// If not found, wrap around and search from the beginning.
		for (i = 0; i<hash; i++) {
			if (atomicCAS(&hashtable.map[i].key, current_key, current_key) == current_key) {
					values[index] = hashtable.map[i].value;
					return;
			}
		}
	}
}

/**
 * @brief CUDA kernel to resize the hash table by re-hashing all elements.
 * @param old_hashtable The original hash table.
 * @param new_hashtable The new, larger hash table to populate.
 * @details Each thread is responsible for re-hashing one bucket from the old table
 * into the new table, using linear probing to resolve collisions.
 */
__global__ void remake_hash(hashtable_t old_hashtable, hashtable_t new_hashtable) {
	int index;
	int key;
	int new_hash;
	int old_key;
	bool added = false; // Bug Prone: 'added' is not initialized for all paths.
	int i;

	index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (index < old_hashtable.size) {
		key = old_hashtable.map[index].key;

		// Skip empty buckets.
		if (key == KEY_INVALID)
			return;

		// Calculate the new hash for the larger table.
		new_hash = my_hash(key, new_hashtable.size);
				
		// --- Re-insertion with Linear Probing ---
		for(i = new_hash; i < new_hashtable.size; i++) {
			if (added == false) {
				// Atomically claim an empty slot in the new table.
				old_key = atomicCAS(&new_hashtable.map[i].key, KEY_INVALID, key);
				if (old_key == KEY_INVALID) {
					new_hashtable.map[i].value = old_hashtable.map[index].value;
					added = true;
					return;
				}
			}
		}
		// Wrap around and continue probing if necessary.
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
 * @brief Host-side constructor. Allocates memory for the hash table on the GPU.
 */
GpuHashTable::GpuHashTable(int size) {
	hashtable.size = size; 
	hashtable.map = NULL;
	
	// Allocate a single contiguous block of memory on the GPU for the table.
	if (cudaMalloc((void**)(&hashtable.map), size * sizeof(entry_t)) != cudaSuccess) {
		printf("[INIT] Cuda malloc failed
");
		return;
	}

	// Initialize all keys to KEY_INVALID (0).
	cudaMemset(hashtable.map, 0, size * sizeof(entry_t));
}

/**
 * @brief Host-side destructor. Frees the GPU memory used by the hash table.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashtable.map);
}

/**
 * @brief Resizes the hash table to a new, larger capacity.
 * @param numBucketsReshape The new size (number of buckets).
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hashtable_t new_hashtable;
	int nr_blocks;

	new_hashtable.size = numBucketsReshape;

	// 1. Allocate memory for the new table on the GPU.
	if (cudaMalloc(&new_hashtable.map, numBucketsReshape * sizeof(entry_t)) != cudaSuccess) {
		printf("[RESHAPE] Cuda malloc failed
");
		return;
	}
	cudaMemset(new_hashtable.map, 0, numBucketsReshape * sizeof(entry_t));
	
	// 2. Calculate kernel launch parameters.
	int ct;
	nr_blocks = hashtable.size / THREADS;
	(hashtable.size % THREADS != 0) ? ct = 1 : ct = 0;  
	nr_blocks += ct;

	// 3. Launch the kernel to re-hash elements from the old table to the new one.
	remake_hash>>(hashtable, new_hashtable);

	// Block host execution until the resize is complete.
	cudaDeviceSynchronize();
	
	// 4. Free the memory of the old table.
	cudaFree(hashtable.map);

	// 5. Update the host-side struct to point to the new table.
	hashtable = new_hashtable;
 }

/**
 * @brief Inserts a batch of key-value pairs from the host.
 * @param keys Host pointer to an array of keys.
 * @param values Host pointer to an array of values.
 * @param numKeys The number of pairs to insert.
 * @return True on success, false on failure.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *d_keys; 
	int *d_values; 
	float current_size;
	float currLoadFactor;
	int nr_blocks;

	// 1. Allocate temporary buffers on the GPU for the keys and values.
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

	actualSize += numKeys;
	current_size = float(actualSize);
	currLoadFactor = loadFactor() + (float)numKeys / (float)hashtable.size;

	// 2. Check if the load factor exceeds the threshold and trigger a resize if so.
	if (currLoadFactor >= 0.8)
		reshape(int(current_size / 0.8));

	// 3. Copy the batch from host memory to device memory.
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

	// 4. Calculate launch parameters and launch the 'insert' kernel.
	int ct;
	nr_blocks = hashtable.size / THREADS;
	(hashtable.size % THREADS != 0) ? ct = 1 : ct = 0;  
	nr_blocks += ct;

	insert>>(hashtable, d_keys, d_values, numKeys);

	// Wait for the kernel to finish.
	cudaDeviceSynchronize();
	
	// 5. Free temporary device buffers.
	cudaFree(d_keys);
	cudaFree(d_values);

	return true;
}

/**
 * @brief Retrieves values for a batch of keys from the host.
 * @param keys Host pointer to an array of keys to look up.
 * @param numKeys The number of keys to retrieve.
 * @return A host pointer to an array containing the results. Caller must free this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *d_keys;
	int *d_values;
	int *result = (int*)malloc(numKeys * sizeof(int));
	int nr_blocks;

	// 1. Allocate temporary device buffers.
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

	// 2. Copy keys from host to device.
	if (cudaMemcpy(d_keys, keys, numKeys*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("[GET] Cuda memcpy failed
");
		return NULL;
	}

	cudaMemset(d_values, 0, numKeys * sizeof(int));

	// 3. Calculate launch parameters and launch the 'get' kernel.
	int ct;
	nr_blocks = hashtable.size / THREADS;
	(hashtable.size % THREADS != 0) ? ct = 1 : ct = 0;  
	nr_blocks += ct;
	
	get>>(hashtable, d_keys, d_values, numKeys);
	
	// Wait for kernel completion.
	cudaDeviceSynchronize();

	// 4. Copy results from device back to host.
	cudaMemcpy(result, d_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	
	// 5. Free temporary device buffers.
	cudaFree(d_keys);
	cudaFree(d_values);

	return result;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
	if (hashtable.size == 0) {
		return 0.f;
	} else {
		return (float)((float)actualSize)/((float)hashtable.size); 
	}
}


// --- Test harness and C-style definitions ---
// This section includes a test file and C-style definitions, which is an
// unconventional project structure.

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

// Represents an empty or invalid slot in the hash table.
#define	KEY_INVALID		0

// A macro for simple error checking and exiting on failure.
#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)

// A list of prime numbers, likely used for various hashing schemes.
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


// Unused hash functions.
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

// Data structure for a single hash table entry (key-value pair).
struct entry_t {
	int key;
	int value;
};

// Data structure representing the hash table on the device.
struct hashtable_t {
	int size;
	entry_t *map;
};


class GpuHashTable
{
	int actualSize;
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
