
/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 * @details This file provides a hash table data structure designed to be operated on
 * by a massive number of parallel threads on a GPU. It uses a variation of
 * cuckoo hashing with a fixed number of slots per bucket, combined with linear
 * probing for collision resolution. All primary operations (insert, get, resize)
 * are implemented as CUDA kernels.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief Computes a hash value for a given key on the GPU.
 * @param size The size of the hash table, used for the modulo operation.
 * @param key The key to hash.
 * @return An integer hash value within the range [0, size-1].
 * @note This is a __device__ function, meaning it is executed on the GPU and
 * can be called from within CUDA kernels.
 */
__device__ int hash_function(int size, int key) {
	// Large prime numbers for hashing to reduce collisions.
	const size_t prime_number_1 = 13169977llu;
	const size_t prime_number_2 = 5351951779llu;

	// Simple multiply-and-modulo hashing scheme.
	int val = ((long)abs(key) * prime_number_1) % prime_number_2 % size;

	return val;
}

/**
 * @brief CUDA kernel to resize the hash table and re-hash all existing elements.
 * @param newHash The new, larger hash table structure to populate.
 * @param hmap The old hash table from which to read elements.
 * @details Each thread is responsible for one bucket of the old hash table and
 * attempts to re-insert its valid elements into the new table.
 */
__global__ void  resize(my_hash newHash, my_hash hmap) {
	// --- Grid-Stride Loop Pattern ---
	// Each thread computes a unique global index 'i' to work on.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int idx, x, slot;
	int once = 0;

	// Guard to ensure threads do not access out of bounds memory.
	if (!(i < hmap.hsize))
		return;

	// Iterate over each potential slot in the bucket.
	for (slot = 0; slot < NO_SLOTS; slot++) {
		once = 0;

		// Skip empty slots in the old table.
		if (hmap.buckets[slot][i].key == KEY_INVALID)
				continue;

		// --- Re-hashing and Insertion into New Table ---
		idx = hash_function(newHash.hsize, hmap.buckets[slot][i].key);

		once = 0;
		int oldIdx = idx;
		// Linear probing loop to find an empty slot in the new table.
		while(idx < newHash.hsize && once < 2) {
			// Optimization: Stop if we've wrapped around and passed the original hash index.
			if (once == 1 && idx > oldIdx)
				break;

			for(x = 0; x < NO_SLOTS; x++) {

				// Synchronization: Use atomicCAS to safely claim an empty slot.
				// This prevents race conditions where multiple threads try to write
				// to the same new bucket simultaneously.
				int old = atomicCAS(&newHash.buckets[x][idx].key, KEY_INVALID, hmap.buckets[slot][i].key);
				if (idx < newHash.hsize && (old == KEY_INVALID)) {
					// If the slot was successfully claimed, write the value.
					atomicExch(&newHash.buckets[x][idx].value, hmap.buckets[slot][i].value);

					once = 3; // Break outer loops
					break;

				}
			}

			idx++;
			// Wrap around if the end of the table is reached.
			if (idx >= newHash.hsize) {
				idx = 0;
				once++;
			}
		}
	}

	return;
}


/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param keys An array of keys to insert.
 * @param values An array of values corresponding to the keys.
 * @param numKeys The total number of pairs to insert.
 * @param hmap The hash table to insert into.
 * @details Each thread handles one key-value pair. It computes the hash, then uses
 * linear probing and atomic operations to find an empty slot or update an existing key.
 */
__global__ void insert(int *keys, int *values, int numKeys, my_hash hmap) {
	// Grid-stride loop pattern for 1D data.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int x = 0;
	int idx = 0;	
	
	int once = 0;

	// Thread guard.
	if (!(i < numKeys))
		return;

	// Skip insertion for invalid keys.
	if (keys[i] <= 0)
		return;

	idx = hash_function(hmap.hsize, keys[i]);

	// --- Collision Handling: Linear Probing with Atomic Operations ---
	once = 0;
	int oldIdx = idx;
	while(idx < hmap.hsize && once < 2) {
		// Stop probing if we've wrapped and gone past the start index.
		if (once == 1 && idx > oldIdx)
			break;
		for(x = 0; x < NO_SLOTS; x++) {

			// Synchronization: Atomically attempt to write the key.
			// This handles both finding an empty slot (KEY_INVALID) and updating
			// an existing key in a thread-safe manner.
			int old = atomicCAS(&hmap.buckets[x][idx].key, KEY_INVALID, keys[i]);
			if (idx < hmap.hsize && (old == KEY_INVALID || old == keys[i])) {
				// If successful, atomically update the value and exit.
				atomicExch(&hmap.buckets[x][idx].value, values[i]);
				return;
			}
		}

		// Linear probing: move to the next bucket.
		idx++;
		// Wrap around at the end of the table.
		if (idx >= hmap.hsize) {
			idx = 0;
			once++;
		}
	}
}


/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param keys An array of keys to look up.
 * @param values An output array to store the found values.
 * @param numKeys The number of keys to look up.
 * @param hmap The hash table to search in.
 */
__global__ void get(int *keys, int *values, int numKeys, my_hash hmap) {
	// Grid-stride loop pattern.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int x = 0;
	int idx;
	int once = 0;

	if (!(i < numKeys))
		return;

	if (keys[i] == KEY_INVALID)
		return;

	idx = hash_function(hmap.hsize, keys[i]);

	// --- Search: Linear Probing ---
	once = 0;
	int oldIdx = idx;
	while(idx < hmap.hsize && once < 2) {
		if (once == 1 && idx > oldIdx)
			break;

		for(x = 0; x < NO_SLOTS; x++) {
			// Intentional or Buggy Write: This atomicCAS appears to attempt a
			// write operation during a get. This is highly unconventional for a
			// read-only operation and may introduce side effects or be a bug.
			// A simple read should be sufficient.
			int old = atomicCAS(&hmap.buckets[x][idx].key, KEY_INVALID, keys[i]);
			
			// If the key at the current slot matches the search key...
			if (idx < hmap.hsize &&  hmap.buckets[x][idx].key == keys[i]) {
				// ...retrieve the value and exit.
				values[i] = hmap.buckets[x][idx].value;
				return;
			}
		}

		idx++;
		// Wrap around.
		if (idx >= hmap.hsize) {
			idx = 0;
			once++;
		}
	}
}

/**
 * @brief Host-side constructor for the GpuHashTable.
 * @param size Initial number of buckets for the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	int i;
	cudaError_t err;
	hmap.hsize = size;

	// Allocate GPU memory for each slot's bucket array.
	for (i = 0; i < NO_SLOTS; i++) {
		err = cudaMalloc((void **)&hmap.buckets[i], size * sizeof(list));
		DIE(err != cudaSuccess || hmap.buckets[i]  == 0, "cudaMalloc");
	}

	// Initialize the allocated GPU memory to zero (marking keys as KEY_INVALID).
	for (i = 0; i < NO_SLOTS; i++) {
		err = cudaMemset(hmap.buckets[i], 0, size * sizeof(list));
		DIE(err != cudaSuccess, "cudaMemset");
	}
	no_insPairs = 0;
}

/**
 * @brief Host-side destructor, frees all GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	int i;
	no_insPairs = 0;
	cudaError_t err;

	for (i = 0; i < NO_SLOTS; i++) {
		err = cudaFree(hmap.buckets[i]);
		DIE(err != cudaSuccess, "cudaFree");
		hmap.buckets[i] = nullptr;
	}
}

/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The new number of buckets.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int i;
	cudaError_t err;
	my_hash newHash;
	int N = numBucketsReshape;

	// 1. Allocate a new, larger hash table on the GPU.
	for (i = 0; i < NO_SLOTS; i++) {
		err = cudaMalloc(&newHash.buckets[i], N * sizeof(list));
		DIE(err != cudaSuccess || newHash.buckets[i]  == 0, "cudaMalloc");
	}
	for (i = 0; i < NO_SLOTS; i++) {
		err = cudaMemset(newHash.buckets[i], 0, N * sizeof(list));
		DIE(err != cudaSuccess, "cudaMemset");
	}
	newHash.hsize = numBucketsReshape;

	// 2. Configure and launch the 'resize' kernel.
	const size_t block_size = 256;
	size_t blocks_no = hmap.hsize / block_size;
	if (hmap.hsize % block_size)
		++blocks_no;
	
	resize>>(newHash, hmap);
	// Block until the kernel completes.
	cudaDeviceSynchronize();

	// 3. Free the old hash table's GPU memory.
	for (i = 0; i < NO_SLOTS; i++) {
		err = cudaFree(hmap.buckets[i]);
		DIE(err != cudaSuccess, "[resize] cudaFree");
		hmap.buckets[i] = nullptr;
	}

	// 4. Point to the new hash table.
	hmap = newHash;
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys Host pointer to an array of keys.
 * @param values Host pointer to an array of values.
 * @param numKeys The number of pairs to insert.
 * @return True on success, false on failure.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys = 0;
	int *device_values = 0;
	cudaError_t err;
	int num_bytes = numKeys * sizeof(*keys);

	// 1. Check load factor and reshape if necessary.
	if (!(float(no_insPairs + numKeys) / (hmap.hsize) < MAX_LOAD))
		reshape(int((no_insPairs + numKeys) / MIN_LOAD));

	
	// 2. Allocate temporary memory on the GPU for the batch.
	err = cudaMalloc((void **) &device_keys, num_bytes);
	DIE(err != cudaSuccess, "[INSERT] cudaMalloc");
	err = cudaMalloc((void **) &device_values, num_bytes);
	DIE(err != cudaSuccess, "[INSERT] cudaMalloc");

	if (device_keys == 0 || device_values == 0) {
		printf("[HOST] Couldn't allocate memory
");
		return false;
	}

	// 3. Copy data from Host RAM to GPU VRAM.
	err = cudaMemcpy(device_keys, keys, num_bytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "[INSERT] cudaMemcpy");
	err =  cudaMemcpy(device_values, values, num_bytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "[INSERT] cudaMemcpy");

	// 4. Configure and launch the 'insert' kernel.
	const size_t block_size = 256;
	size_t blocks_no = numKeys / block_size;
	if (numKeys % block_size)
		++blocks_no;
	
	insert>>(device_keys, device_values, numKeys, hmap);
	// Block until the kernel completes.
	cudaDeviceSynchronize();

	// 5. Clean up temporary GPU memory.
	err = cudaFree(device_values);
	DIE(err != cudaSuccess, "[INSERT] cudaFree");
	err = cudaFree(device_keys);
	DIE(err != cudaSuccess, "[INSERT] cudaFree");

	no_insPairs +=  numKeys;
	return true;
}

/**
 * @brief Retrieves values for a batch of keys.
 * @param keys Host pointer to an array of keys to look up.
 * @param numKeys The number of keys to get.
 * @return A host pointer to an array of found values. The caller must free this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *host_values = 0;
	int *device_keys = 0;
	int *device_values = 0;
	int num_bytes = numKeys * sizeof(int);
	cudaError_t err;

	// 1. Allocate memory on the host for the results.
	host_values = (int *) calloc(1, num_bytes);
	DIE(host_values == NULL, "[GET] calloc");

	// 2. Allocate temporary memory on the GPU for the operation.
	err = cudaMalloc((void **) &device_keys, num_bytes);
	DIE(err != cudaSuccess, "[GET] cudaMalloc");
	err = cudaMalloc((void **) &device_values, num_bytes);
	DIE(err != cudaSuccess, "[GET] cudaMalloc");

	if (device_keys == 0 || device_values == 0) {
		printf("[GET HOST] Couldn't allocate memory
");
		return NULL;
	}

	// 3. Copy keys from Host RAM to GPU VRAM.
	err = cudaMemcpy(device_keys, keys, num_bytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "[GET] cudaMemcpy");

	// 4. Configure and launch the 'get' kernel.
	const size_t block_size = 256;
	size_t blocks_no = numKeys / block_size;
	if (numKeys % block_size)
		++blocks_no;
	
	get>>(device_keys, device_values, numKeys, hmap);
	// Block until the kernel completes.
	cudaDeviceSynchronize();

	// 5. Copy results from GPU VRAM back to Host RAM.
	err = cudaMemcpy(host_values, device_values, num_bytes, cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, "[GET] cudaMemcpy");

	// 6. Free temporary GPU memory.
	err = cudaFree(device_values);
	DIE(err != cudaSuccess, "[GET] cudaFree");

	return host_values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float.
 */
float GpuHashTable::loadFactor() {
	if (hmap.hsize == 0)
		return 0;

	return float(no_insPairs) / (hmap.hsize);
}


// --- Test harness and C-style definitions ---
// This section includes a test file and C-style definitions, which is an
// unconventional project structure. The definitions below are likely shared
// with a corresponding CPU implementation.

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

// --- Constants ---
#define	KEY_INVALID		0     // Represents an empty slot in the hash table.
#define NO_SLOTS		2     // Number of slots per bucket (cuckoo-style).
#define MIN_LOAD		0.8   // Load factor to aim for after resizing.
#define MAX_LOAD		0.9   // Load factor that triggers a resize.

// A macro for simple error checking and exiting.
#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)
	
// A list of prime numbers, likely used for hashing schemes.
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

// Data structure for a single key-value pair.
struct list {
	int key;
	int value;
};

// Host-side representation of the hash table's core data.
struct my_hash {
	int hsize;
	list *buckets[NO_SLOTS];
};



// Alternative hash functions (unused in the primary kernels).
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}



class GpuHashTable
{
		my_hash hmap;
		int no_insPairs;
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
