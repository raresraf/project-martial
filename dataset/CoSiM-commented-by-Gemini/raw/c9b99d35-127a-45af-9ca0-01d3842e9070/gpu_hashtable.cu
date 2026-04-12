
/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table using open addressing with linear probing
 *        and an interleaved data layout.
 * @details This file implements a hash table on the GPU using an open addressing
 * strategy. The key-value pairs are stored in an interleaved format `[k0, v0, k1, v1, ...]`
 * within a single buffer.
 *
 * It correctly uses atomic operations for inserting new elements and for performing
 * a parallel reshape. However, it contains two notable flaws:
 * 1. A race condition on non-atomic value updates for existing keys.
 * 2. A potential infinite loop in the `get` kernel if a key is not found in a full table.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

// Default block size for kernel launches.
#define BLOCK_SIZE 1000
// The load factor at which a resize is triggered.
#define RESIZE_TRESHOLD 0.9f
// The target load factor to achieve after a resize.
#define OPTIMAL_LOAD 0.85f

/**
 * @brief A macro for CUDA error checking.
 * @details Checks the last CUDA error and prints a detailed message before exiting
 * if an error has occurred. This is a robust practice for debugging CUDA API calls.
 */
#define cudaCheckErrors(msg) 
    { 
        cudaError_t err = cudaGetLastError(); 
        if (err != cudaSuccess) { 
            fprintf(stderr, "Cuda error: %s (%s at %s:%d)
", 
                msg, cudaGetErrorString(err), 
                __FILE__, __LINE__); 
            exit(1); 
        } 
    }

/**
 * @brief Computes a hash value for a given key.
 * @param data The key to hash.
 * @param limit The size of the hash table (number of key-value pair slots).
 * @return An integer hash value within the range [0, limit-1].
 */
__device__ int hash_func(int data, int limit)
{
	// A simple multiply-and-modulo hashing scheme using large prime numbers.
	return (data * 1087448823553llu) % 28282345988300791llu % limit;
}

/**
 * @brief CUDA kernel to resize the hash table by re-hashing all elements.
 * @param src The source hash table buffer (old).
 * @param src_size The capacity of the source hash table.
 * @param dst The destination hash table buffer (new).
 * @param dst_size The capacity of the destination hash table.
 */
__global__ void kernel_reshape(
	int *src, int src_size,
	int *dst, int dst_size
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int key, value, old_key, hash_value;

	if (idx >= src_size)
		return;

	// Data Layout: Key and value are read from interleaved positions.
	key = src[2 * idx];

	// Skip empty slots from the old table.
	if (key == KEY_INVALID)
		return;

	value = src[2 * idx + 1];

	// Re-hash the key for the new table size.
	hash_value = hash_func(key, dst_size);

	// --- Re-insertion with Linear Probing ---
	while (true) {
		// Atomically claim an empty slot in the new table.
		old_key = atomicCAS(dst + 2 * hash_value, KEY_INVALID, key);
		if (old_key == KEY_INVALID) {
			// Since each thread is migrating a unique key from the old table,
			// a non-atomic write for the value is safe here.
			dst[2 * hash_value + 1] = value;
			break;
		}
		// Probe to the next slot, wrapping around if necessary.
		hash_value = (hash_value + 1) % dst_size;
	}
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs.
 * @param buf The hash table buffer.
 * @param size The capacity of the hash table.
 * @param keys Device pointer to keys to insert.
 * @param values Device pointer to values to insert.
 * @param count The number of pairs to insert.
 * @param new_elem A device pointer to an integer used as an atomic counter for new elements.
 *
 * @warning RACE CONDITION: The value update (`buf[2 * hash_value + 1] = value`) is
 * not atomic. If multiple threads try to update an existing key, data may be lost.
 */
__global__ void kernel_insert(
	int *buf, int size,
	int *keys, int *values, int count,
	int *new_elem
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int key, value, old_key, hash_value;

	if (idx >= count)
		return;

	key = keys[idx];
	if (key == KEY_INVALID)
		return;

	value = values[idx];
	hash_value = hash_func(key, size);

	// --- Insertion with Linear Probing ---
	while (true) {
		// Atomically attempt to claim a slot. This is safe for finding an empty
		// slot or finding an slot that already contains the same key.
		old_key = atomicCAS(buf + 2 * hash_value, KEY_INVALID, key);
		if (old_key == KEY_INVALID || old_key == key) {
			// BUG: Non-atomic write. Creates a race condition for updates to existing keys.
			buf[2 * hash_value + 1] = value;
			// If a new element was inserted, atomically increment the counter.
			if (old_key == KEY_INVALID)
				atomicAdd((unsigned int *)new_elem, 1);
			break;
		}
		// Probe to the next slot, wrapping around the buffer.
		hash_value = (hash_value + 1) % size;
	}
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param buf The hash table buffer.
 * @param size The capacity of the hash table.
 * @param keys Device pointer to keys to look up.
 * @param values Device pointer to an array to store results.
 * @param count The number of keys to retrieve.
 *
 * @warning BUG: This kernel can enter an infinite loop. If the table is full and
 * a key being searched for is not present, the `while(1)` loop will never terminate.
 */
__global__ void kernel_get(
	int *buf, int size,
	int *keys, int *values, int count
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int key, hash_value;

	if (idx >= count)
		return;

	key = keys[idx];
	if (key == KEY_INVALID)
		return;

	hash_value = hash_func(key, size);
	// --- Search with Linear Probing ---
	while (1) { // Potential infinite loop.
		// A non-atomic read is sufficient and correct here.
		if (buf[2 * hash_value] == key) {
			values[idx] = buf[2 * hash_value + 1];
			break;
		}
		// Probe to the next slot.
		hash_value = (hash_value + 1) % size;
	}
}

/**
 * @brief Host-side constructor. Allocates an interleaved key-value buffer on the GPU.
 */
GpuHashTable::GpuHashTable(int size)
{
	// The buffer size is doubled to accommodate the interleaved [key, value] pairs.
	int buf_size = 2 * size * sizeof(int);

	cudaMalloc(&this->buf, buf_size);
	cudaCheckErrors("cudaMalloc");
	cudaMemset(this->buf, 0, buf_size);
	cudaCheckErrors("cudaMemset");

	this->size = size;
	this->count = 0;
}

/**
 * @brief Host-side destructor. Frees GPU memory.
 */
GpuHashTable::~GpuHashTable()
{
	cudaFree(this->buf);
	cudaCheckErrors("cudaFree");
}

/**
 * @brief Resizes the hash table to a new capacity by re-hashing all elements.
 */
void GpuHashTable::reshape(int new_size)
{
	int *new_buf, new_buf_size = 2 * new_size * sizeof(int);

	// 1. Allocate a new, larger buffer on the GPU.
	cudaMalloc(&new_buf, new_buf_size);
	cudaCheckErrors("cudaMalloc");
	cudaMemset(new_buf, 0, new_buf_size);
	cudaCheckErrors("cudaMemset");

	// 2. Launch the kernel to perform a parallel re-hash.
	kernel_reshape>>(
		this->buf, this->size,
		new_buf, new_size
	);
	cudaDeviceSynchronize();
	cudaCheckErrors("cudaDeviceSynchronize");

	// 3. Free the old buffer and update the hash table's state.
	cudaFree(this->buf);
	cudaCheckErrors("cudaFree");

	this->buf = new_buf;
	this->size = new_size;
}

/**
 * @brief Inserts a batch of key-value pairs from the host.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int num_keys)
{
	// 1. Check load factor and reshape if it exceeds the threshold.
	if (this->count + num_keys >= RESIZE_TRESHOLD * this->size) {
		this->reshape((this->count + num_keys) / OPTIMAL_LOAD);
	}

	int host_new_elem, *device_keys, *device_values, *device_new_elem;
	int buf_size = num_keys * sizeof(int);

	// 2. Allocate temporary device buffers for the batch and a counter.
	cudaMalloc(&device_keys, buf_size);
	cudaCheckErrors("cudaMalloc");
	cudaMalloc(&device_values, buf_size);
	cudaCheckErrors("cudaMalloc");
	cudaMalloc(&device_new_elem, sizeof(int));
	cudaCheckErrors("cudaMalloc");
	cudaMemcpy(device_keys, keys, buf_size, cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy");
	cudaMemcpy(device_values, values, buf_size, cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy");
	cudaMemset(device_new_elem, 0, sizeof(int));
	cudaCheckErrors("cudaMemset");

	// 3. Launch the insertion kernel.
	kernel_insert>>(
		this->buf, this->size,
		device_keys, device_values, num_keys,
		device_new_elem
	);
	cudaDeviceSynchronize();
	cudaCheckErrors("cudaDeviceSynchronize");

	// 4. Copy the count of new elements back to the host.
	cudaMemcpy(&host_new_elem, device_new_elem, sizeof(int),
		cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy");

	// 5. Free temporary buffers.
	cudaFree(device_keys);
	cudaCheckErrors("cudaFree");
	cudaFree(device_values);
	cudaCheckErrors("cudaFree");
	cudaFree(device_new_elem);
	cudaCheckErrors("cudaFree");

	// 6. Update the host-side element count.
	this->count += host_new_elem;

	return true;
}

/**
 * @brief Retrieves values for a batch of keys from the host.
 */
int* GpuHashTable::getBatch(int* keys, int num_keys)
{
	int *host_values, *device_keys, *device_values;
	int buf_size = num_keys * sizeof(int);

	// 1. Allocate temporary device buffers and host buffer for results.
	cudaMalloc(&device_keys, buf_size);
	cudaCheckErrors("cudaMalloc");
	cudaMalloc(&device_values, buf_size);
	cudaCheckErrors("cudaMalloc");
	cudaMemcpy(device_keys, keys, buf_size, cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy");
	host_values = (int *)malloc(buf_size);
	DIE(!host_values, "malloc");

	// 2. Launch the get kernel.
	kernel_get>>(
		this->buf, this->size,
		device_keys, device_values, num_keys
	);
	cudaDeviceSynchronize();
	cudaCheckErrors("cudaDeviceSynchronize");

	// 3. Copy results from device to host.
	cudaMemcpy(host_values, device_values, buf_size, cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy");

	// 4. Free temporary device buffers.
	cudaFree(device_keys);
	cudaCheckErrors("cudaFree");
	cudaFree(device_values);
	cudaCheckErrors("cudaFree");

	return host_values;
}

/**
 * @brief Calculates the current load factor.
 */
float GpuHashTable::loadFactor()
{
	return this->count / (float)this->size;
}


// --- Test harness and boilerplate ---
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);
#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)
#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

#include 

using namespace std;

#define	KEY_INVALID		0 // Represents an empty slot.

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


class GpuHashTable
{
	private:
		int *buf;
		int size;
		int count;

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
