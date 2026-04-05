/**
 * @file gpu_hashtable.cu
 * @brief A GPU-based hash table implementation using open addressing with linear probing.
 *
 * This implementation is notable for its resizing strategy: before inserting a
 * batch, it first performs a lookup (`getBatch`) for all keys in the batch to
 * determine exactly how many are new. This count is then used to decide if a
 * resize is needed. While precise, this approach is inefficient as it requires
 * a full round-trip to the GPU before the actual insertion.
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
 * @brief CUDA kernel to rehash all elements from an old table to a new, larger one.
 *
 * Each thread is assigned to an entry in the old table. If the entry is valid, 
 * the thread re-calculates the hash for the key based on the new table size and
 * inserts the element into the new table using an atomic linear probing strategy.
 */
__global__ void resize(GpuHashTable::entry *hashtable, GpuHashTable::entry *new_hash,
						int hash_size, int numBucketsReshape) {
	
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < hash_size) {
		if (hashtable[tid].key == KEY_INVALID)
			return;
		
		uint32_t key = hash_func(hashtable[tid].key, numBucketsReshape);
		while (true) {
			// Atomically attempt to claim an empty slot in the new table.
			uint32_t prev = atomicCAS(&new_hash[key].key, KEY_INVALID, hashtable[tid].key);
			if (prev == hashtable[tid].key || prev == KEY_INVALID) {
				new_hash[key].value = hashtable[tid].value;
				break; // Success
			}
			key++;
			key %= numBucketsReshape;
		}
	}
}

/**
 * @brief CUDA kernel for inserting a batch of key-value pairs.
 *
 * Each thread handles one key-value pair. It computes an initial hash and then
 * uses linear probing to find an empty slot. `atomicCAS` ensures that each
 * empty slot is claimed by only one thread, preventing race conditions.
 */
__global__ void insert(GpuHashTable::entry *hashtable, int hash_size,
						int *keys, int* values, int numKeys) {
	
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < numKeys) {
		
		uint32_t key = hash_func(keys[tid], hash_size);
		while (true) {
			// Atomically attempt to write the new key if the slot is empty (KEY_INVALID)
			// or if the slot already contains the same key (for updates).
			uint32_t prev = atomicCAS(&hashtable[key].key, KEY_INVALID, keys[tid]);
			if (prev == keys[tid] || prev == KEY_INVALID) {
				hashtable[key].value = values[tid];
				return;
			}
			key++;
			key %= hash_size;
		}
	}
}

/**
 * @brief CUDA kernel for retrieving values for a batch of keys.
 *
 * Each thread searches for one key. It uses linear probing, checking each bucket
 * until the key is found or an empty bucket (`KEY_INVALID`) is encountered, which
 * indicates the key is not in the table.
 */
__global__ void get(GpuHashTable::entry *hashtable, int hash_size,
						int *keys, int *values, int numKeys) {
	
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < numKeys) {
		

		uint32_t key = hash_func(keys[tid], hash_size);
		while (true) {
			// If the key is found, store the value and terminate.
			if (hashtable[key].key == keys[tid]) {
				values[tid] = hashtable[key].value;
				break;
			}
			// If an empty slot is found, the key does not exist.
			if (hashtable[key].key == KEY_INVALID) {
				values[tid] = KEY_INVALID;
				break;
			}
			// Probe the next slot.
			key++;
			key %= hash_size;
		}
	}
}

/**
 * @brief Host-side constructor. Allocates and initializes the hash table on the GPU.
 */
GpuHashTable::GpuHashTable(int size) {
	hash_size = size; 
	num_entries = 0;  
	cudaMalloc((void **) &hashtable, size * sizeof(entry));
	cudaMemset(hashtable, KEY_INVALID, size * sizeof(entry));
}

/**
 * @brief Host-side destructor. Frees the GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashtable);
}

/**
 * @brief Resizes the hash table and rehashes all elements on the GPU.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	uint32_t block_size = 100; // This block size seems arbitrary.
	uint32_t blocks_no = hash_size / block_size;
	if (hash_size % block_size)
		++blocks_no;
	struct entry *new_hash;
	
	// 1. Allocate a new, larger table on the device.
	cudaMalloc((void **) &new_hash, numBucketsReshape * sizeof(entry));
	cudaMemset(new_hash, KEY_INVALID, numBucketsReshape * sizeof(entry));

	// 2. Launch the kernel to rehash elements from the old table to the new one.
	resize>>(hashtable, new_hash, hash_size, numBucketsReshape);
	cudaDeviceSynchronize();

	// 3. Free the old table and update the class to point to the new one.
	cudaFree(hashtable);
	hashtable = new_hash;
	hash_size = numBucketsReshape;
}

/**
 * @brief Host-side function to insert a batch of key-value pairs.
 *
 * This function has an unusual and inefficient resizing strategy. It first
 * performs a `getBatch` operation to find which keys are new, then calculates
 * the load factor, and only then does it proceed with the actual insertion.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *new_values;
	
	// Inefficient resizing strategy: Perform a full lookup of all keys to be
	// inserted just to count how many are new.
	new_values = getBatch(keys, numKeys);
	for (int i = 0; i < numKeys; i++)
		if (new_values[i] == KEY_INVALID)
			num_entries++;

	// Decide whether to resize based on the new element count.
	if ((float)(num_entries) / hash_size >= 0.9)
		reshape(num_entries + (int)(0.1 * num_entries));

	uint32_t block_size = 100;
	uint32_t blocks_no = numKeys / block_size;
	if (numKeys % block_size)
		++blocks_no;
	
	// Copy data to the device for the actual insertion.
	int *dev_keys = 0;
	int *dev_values = 0;
	cudaMalloc((void **) &dev_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &dev_values, numKeys * sizeof(int));
	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Launch the insertion kernel.
	insert>>(hashtable, hash_size, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize();

	// Clean up.
	cudaFree(dev_keys);
	cudaFree(dev_values);
	free(new_values);
	return true;
}

/**
 * @brief Host-side function to retrieve values for a batch of keys.
 * @return A pointer to a host array with the results. The caller must free this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *results = (int *)malloc(numKeys * sizeof(int));
	uint32_t block_size = 100;
	uint32_t blocks_no = numKeys / block_size;
	if (numKeys % block_size)
		++blocks_no;
	int *dev_keys = 0;
	int *dev_values = 0;
	
	// Allocate temporary device memory.
	cudaMalloc((void **) &dev_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &dev_values, numKeys * sizeof(int));
	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemset(dev_values, KEY_INVALID, numKeys * sizeof(int));
	
	// Launch the get kernel.
	get>>(hashtable, hash_size, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize();
	
	// Copy results from device back to host.
	cudaMemcpy(results, dev_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_keys);
	cudaFree(dev_values);
	return results;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
	return (float)num_entries / hash_size; 
}


// --- Convenience macros for the test file ---
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()
#define HASH_DESTROY GpuHashTable.~GpuHashTable();

#include "test_map.cpp"
// The file includes a header guard and content that should be in gpu_hashtable.hpp
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0

// C-style error handling macro.
#define DIE(assertion, call_description) \
	do { \
		if (assertion) { \
		fprintf(stderr, "(%s, %d): ", \
		__FILE__, __LINE__); \
		perror(call_description); \
		exit(errno); \
	} \
} while (0)
	
// A large list of prime numbers, likely for hashing.
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


// Other hash functions defined in the header but not used by the main logic.
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

// A Jenkins-style hash function.
__device__ uint32_t hash_func(int key, int size) {
	key = ~key + (key << 15);
	key = key ^ (key >> 12);
	key = key + (key << 2);
	key = key ^ (key >> 4);
	key = key * 2057;
	key = key ^ (key >> 16);
	return key % size;
}

/**
 * @brief A simple key-value pair struct for the hash table entries.
 */
class GpuHashTable
{
	public:
		struct entry {
			uint32_t key;
			uint32_t value;
		};
		int hash_size;
		int num_entries;
		struct entry *hashtable;

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