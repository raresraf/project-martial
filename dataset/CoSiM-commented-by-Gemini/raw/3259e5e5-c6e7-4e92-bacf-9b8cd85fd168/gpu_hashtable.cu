/**
 * @file gpu_hashtable.cu
 * @brief A GPU-based hash table implementation using open addressing with linear probing.
 *
 * This file provides a hash table that resides entirely in GPU memory. It uses a
 * single contiguous array for storage and resolves hash collisions by linearly
 * probing for the next available bucket. Operations are performed in parallel
 * batches, leveraging CUDA kernels for high throughput.
 */
#include 
#include 
#include 
#include 
#include 
#include 
#include "gpu_hashtable.hpp"

// Prime numbers used for the multiplicative hash function.
#define PRIME_A	823117
#define PRIME_B 3452434812973

/**
 * @brief A device function to compute a hash for a given key.
 * @param Key The integer key to hash.
 * @param Size The total number of buckets in the hash table.
 * @return An integer hash value within the bounds of the table size.
 *
 * This function uses a simple multiplicative hashing scheme.
 */
__device__ int ComputeHash(int Key, int Size)
{
	return ((long long)abs(Key) * PRIME_A) % PRIME_B % Size;
}

/**
 * @brief CUDA kernel for inserting a batch of key-value pairs into the hash table.
 *
 * Each thread in the grid is responsible for inserting one key-value pair.
 * It computes an initial hash and then uses linear probing to find an empty slot.
 * The `atomicCAS` (Compare-and-Swap) operation ensures that each slot is claimed
 * by only one thread, preventing race conditions.
 */
__global__ void kernel_insert(int *keys, int *values, int NumElements, int Size, Pair *Container)
{
	int key, idx, hash, ret;

	idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= NumElements)
		return;

	key = keys[idx];

	hash = ComputeHash(key, Size);

	// Loop until the key is successfully inserted or an existing entry is updated.
	while (true) {
		// Attempt to atomically claim an empty slot (where key is KEY_INVALID).
		// `atomicCAS` returns the value that was in the slot *before* the operation.
		ret = atomicCAS(&Container[hash].key, KEY_INVALID, key);
		
		// If the slot was empty (ret == KEY_INVALID) or already contained our key,
		// the operation is successful. We can safely write the value and terminate.
		if (ret == KEY_INVALID || ret == key) {
			Container[hash].value = values[idx];
			return;
		}
		
		// If the slot was occupied by a different key, probe the next slot.
		hash = (hash + 1) % Size;
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 *
 * Each thread searches for one key. It uses linear probing, checking each bucket
 * until the key is found.
 *
 * @note This kernel has a potential bug: if a key is not present in a completely
 * full hash table, the `while(true)` loop will never terminate, as there is no
 * empty bucket or cycle detection to stop the search.
 */
__global__ void kernel_get(int *keys, int *values, int NumElements, int Size, Pair *Container)
{
	int idx, key, hash, ret;

	idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx >= NumElements)
		return;

	key = keys[idx];
	hash = ComputeHash(key, Size);

	// Linearly probe the table to find the key.
	while (true) {
		// A non-atomic read is sufficient here, as keys are not modified after insertion.
		if (Container[hash].key == key) {
			values[idx] = Container[hash].value;
			return; // Key found.
		}
		// If not found, probe the next slot.
		hash = (hash + 1) % Size;
	}
}

/**
 * @brief CUDA kernel to copy all elements from an old table to a new, larger table.
 *
 * This kernel is used during the `reshape` operation. Each thread takes one element
 * from the old table and re-inserts it into the new table using the same atomic
 * linear probing logic as `kernel_insert`.
 */
__global__ void kernel_copy(Pair *OldContainer, int OldSize, Pair *NewContainer, int NewSize)
{
	int idx, key, hash, ret;

	idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= OldSize)
		return;

	key = OldContainer[idx].key;
    if (key == KEY_INVALID) { // Skip empty slots from the old table.
        return;
    }
	hash = ComputeHash(key, NewSize); // Re-hash for the new table size.

	// Use atomic linear probing to insert into the new table.
	while(true) {
		ret = atomicCAS(&NewContainer[hash].key, KEY_INVALID, key); 
		if (ret == KEY_INVALID) {
			NewContainer[hash].value = OldContainer[idx].value;
			return;
		}
		hash = (hash + 1) % NewSize;
	}
}

/**
 * @brief Host-side constructor. Allocates and initializes the hash table on the GPU.
 */
GpuHashTable::GpuHashTable(int size) {
	cudaError_t error;

	NumElements = 0;
	Size = size;

	error = cudaMalloc((void**)&Container, Size * sizeof(struct Pair));
	if (error != cudaSuccess) {
		printf("%s\n",  cudaGetErrorString(error));
		return;
	}
	// Initialize all keys to KEY_INVALID (0).
	cudaMemset(Container, 0, Size * sizeof(Pair));
}

/**
 * @brief Host-side destructor. Frees GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(Container);
}

/**
 * @brief Resizes the hash table and rehashes all elements on the GPU.
 *
 * This function performs an efficient, GPU-only resize. It allocates a new,
 * larger table and launches a kernel to copy and re-hash elements in parallel,
 * avoiding any costly data transfers to the host.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	Pair *NewContainer;
	int NewSize = numBucketsReshape;
	unsigned int NumBlocks;

	cudaMalloc(&NewContainer, NewSize * sizeof(Pair));
	if (NewContainer == NULL)
		return;

	cudaMemset(NewContainer, 0, NewSize * sizeof(Pair));

	// Configure and launch the copy kernel to rehash elements.
	NumBlocks = Size / THREADS_PER_BLOCK;
	if (Size % THREADS_PER_BLOCK)
		++NumBlocks;

	kernel_copy>>(Container, Size, NewContainer, NewSize);
	cudaDeviceSynchronize();

	// Free the old container and update the class members.
	cudaFree(Container);
	Size = NewSize;
	Container = NewContainer;
}

/**
 * @brief Host-side function to insert a batch of key-value pairs.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *DeviceKeys, *DeviceValues;
	unsigned int NumBlocks;
	size_t size = numKeys * sizeof(int);

	cudaMalloc(&DeviceKeys, size);
	cudaMalloc(&DeviceValues, size);

	if (DeviceKeys == NULL || DeviceValues == NULL)
		return false;

	// Check the load factor and resize the table if it would be exceeded.
	if ((float)(NumElements + numKeys) / Size >= MAX_LOAD_FACTOR) {
		reshape((int)((NumElements + numKeys) / MIN_LOAD_FACTOR) +
			(int)primeList[0]);
	}

	// Copy the input batch from host to device.
	cudaMemcpy(DeviceKeys, keys, size, cudaMemcpyHostToDevice);
	cudaMemcpy(DeviceValues, values, size, cudaMemcpyHostToDevice);
	
	// Configure kernel launch parameters.
	NumBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK)
		++NumBlocks;

	// Launch the insertion kernel.
	kernel_insert>>(DeviceKeys, DeviceValues, numKeys, Size, Container);
	cudaDeviceSynchronize();

	NumElements += numKeys;

	// Free temporary device memory.
	cudaFree(DeviceKeys);
	cudaFree(DeviceValues);

	return true;
}

/**
 * @brief Host-side function to retrieve values for a batch of keys.
 * @return A pointer to a host array containing the results. The caller must free this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *DeviceValues, *DeviceKeys, *Values;
	unsigned int NumBlocks;
	size_t KeysSize = numKeys * sizeof(int);

	// Allocate memory on GPU for keys to find and for the results.
	cudaMalloc(&DeviceValues, KeysSize);
	if (DeviceValues == NULL)
		return NULL;
	cudaMalloc(&DeviceKeys, KeysSize);
	if (DeviceKeys == NULL)
		return NULL;

	// Allocate memory on host for the final returned values.
	Values = (int *)malloc(KeysSize);
	if (Values == NULL)
		return NULL;

	// Copy keys from host to device.
	cudaMemcpy(DeviceKeys, keys, KeysSize, cudaMemcpyHostToDevice);
	
    // Configure and launch the get kernel.
    NumBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK)
		++NumBlocks;

	kernel_get>>(DeviceKeys, DeviceValues, numKeys, Size, Container);
	cudaDeviceSynchronize();

	// Copy the results from device memory back to host memory.
	cudaMemcpy(Values, DeviceValues, KeysSize, cudaMemcpyDeviceToHost);

	// Clean up temporary device memory.
	cudaFree(DeviceValues);
	cudaFree(DeviceKeys);

	return Values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
	return (Size == 0) ? 0 : ((float)NumElements / Size);
}

// --- Convenience macros for the test file ---
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"

// The file includes a header guard and content that should be in gpu_hashtable.hpp
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

// --- Hash Table Constants ---
#define	KEY_INVALID			0 // Value representing an empty bucket.
#define THREADS_PER_BLOCK	1024 // Default number of threads per block for kernels.
#define MIN_LOAD_FACTOR		0.80 // The target load factor after a resize.
#define MAX_LOAD_FACTOR		0.97 // The load factor that triggers a resize.

// C-style error handling macro.
#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)
	
// A large list of prime numbers, likely used for hashing.
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


// These hash functions are defined but not used in the main logic, which
// only uses ComputeHash. This suggests they might be remnants of a different hashing scheme.
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

// A simple key-value pair struct for the hash table entries.
struct Pair {
	int key, value;
};


/**
 * @brief The host-side class providing an interface to the GPU hash table.
 */
class GpuHashTable
{
	Pair *Container; // Pointer to the hash table array in GPU memory.
	int Size; // The current capacity (number of buckets) of the hash table.
	int NumElements; // The number of elements currently stored.

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

