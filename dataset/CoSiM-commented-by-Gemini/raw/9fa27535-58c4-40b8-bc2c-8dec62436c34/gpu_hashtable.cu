/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 *
 * @details
 * This file implements a hash table on the GPU using open addressing with a
 * linear probing collision resolution strategy. The implementation supports
 * batch insertion, batch retrieval, and dynamic resizing (reshaping) based
 * on the table's load factor.
 *
 * @algorithm
 * - Hashing: A multiplicative hash function (`hash_func`) is used to map keys to indices.
 * - Collision Resolution: Linear probing is employed. If a hash collision occurs,
 *   the algorithm probes the next bucket sequentially (with wrap-around) until
 *   an empty slot is found.
 * - Concurrency: The `atomicCAS` (Compare-And-Swap) operation is used during
 *   insertions and reshaping to handle race conditions, ensuring that multiple
 *   threads can safely write to the hash table. An atomic counter (`atomicAdd`)
 *   is used to track the number of new elements inserted.
 *
 * @note The file follows an unconventional structure, appearing to concatenate the
 * implementation with its own header definitions and a test file include.
 */
#include 
#include 
#include 
#include 
#include 
#include 
#include "gpu_hashtable.hpp"

// Global block size for kernel launches.
int block_size = 1024;

/**
 * @brief Computes a hash for a given key on the device.
 * @param key The key to hash.
 * @param hashSize The capacity of the hash table, used for the modulo operation.
 * @return The calculated hash index.
 */
__device__ uint32_t hash_func(uint32_t key, int hashSize) {
  return ((long)(key) * 34798362354533llu) % 11798352904533llu % hashSize;
}


/**
 * @brief CUDA kernel to copy and rehash all elements from an old table to a new one.
 * @param hashTable Device pointer to the old hash table data.
 * @param device2 Device pointer to the new hash table data.
 * @param hashSize The capacity of the new table.
 * @param old_hashSize The capacity of the old table.
 * @details Each thread is responsible for one element from the old table, rehashing and
 * inserting it into the new table using linear probing.
 */
__global__ void kernel_reshape(my_elem *hashTable, my_elem *device2, int hashSize, int old_hashSize) {
  // Thread Indexing: Each thread is assigned one slot from the old hash table.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= old_hashSize)
    return;
  int key = hashTable[idx].key;
  // Pre-condition: Only process valid (non-empty) slots.
  if (key == 0) return;
  int val = hashTable[idx].value;
  int pos = hash_func(key, hashSize);

  // Block Logic: Linear probing loop to find an empty slot in the new table.
  while (1) {
    // Synchronization: Atomically attempt to claim an empty slot.
    uint32_t old_key = atomicCAS(&device2[pos].key, 0, key);
    if (old_key == 0) {
      device2[pos].value = val;
      break; // Exit loop on successful insertion.
    }
    // Collision: Probe the next slot with wrap-around.
    pos = (pos + 1) % hashSize;
  }
  return;
}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @details Each thread searches for one key using linear probing.
 */
__global__ void kernel_get_batches(my_elem *hashTable, int *device_values, const int *keys, int numKeys, int hashSize) {
  // Thread Indexing: Each thread is assigned one key to look up.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numKeys)
    return;
  int key = keys[idx];
  int pos = hash_func(key, hashSize);

  // Block Logic: Linear probing search.
  while (1) {
    if (hashTable[pos].key == key) {
      // Key found, store the value in the output array.
      device_values[idx] = hashTable[pos].value;
      return;
    }
    // If an empty slot is found, the key is not in the table.
    if (hashTable[pos].key == 0) {
      return; // The output value will remain 0 as initialized.
    }
    // Collision: Probe the next slot.
    pos = (pos + 1) % hashSize;
  }
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param e A device pointer to a counter for the number of newly inserted elements.
 * @details Each thread handles one key-value pair, using linear probing and
 * atomicCAS to find an empty slot or update an existing key.
 */
__global__ void kernel_insert_batches(my_elem *hashTable, const int *keys, const int *values, int numKeys, int hashSize, int *e) {
  unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t key, val;
  int pos;
  if (idx >= numKeys)
    return;
  key = (uint32_t) keys[idx];
  val = (uint32_t) values[idx];
  pos = hash_func(key, hashSize);
  // Pre-condition: Ignore invalid keys.
  if (key <= 0) {
    return;
  }

  // Block Logic: Linear probing loop.
  while (1) {
    // Synchronization: Atomically attempt to claim an empty slot.
    uint32_t old_key = atomicCAS(&hashTable[pos].key, 0, key);
    if (old_key == 0) {
      // Case 1: Slot was empty. We claimed it.
      // Atomically increment the counter for new elements.
      atomicAdd(&e[0], 1);
      hashTable[pos].value = val;
      return;
    }
    if (old_key == key) {
      // Case 2: Key already existed. Update the value.
      hashTable[pos].value = val;
      return;
    }
    // Case 3: Collision. Probe the next slot.
    pos = (pos + 1) % hashSize;
  }
}

/**
 * @brief Constructor for GpuHashTable.
 */
GpuHashTable::GpuHashTable(int size) {
  hashSize = size;
  hashElems = 0;
  uint32_t hash_size = size * sizeof(my_elem);
  // Memory Allocation: Allocate the main array for key-value pairs on the device.
  cudaMalloc((void **)&hashTable, hash_size);
  // Initialize all keys to 0 (KEY_INVALID).
  cudaMemset(hashTable, 0, hash_size);
}

/**
 * @brief Destructor for GpuHashTable.
 */
GpuHashTable::~GpuHashTable() {
  cudaFree(hashTable);
}

/**
 * @brief Resizes the hash table and rehashes all elements.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
  int new_size = numBucketsReshape * sizeof(my_elem);
  const int num_elements = hashSize;
  size_t blocks_no = num_elements / block_size;
  if (num_elements % block_size)
    ++blocks_no;
  my_elem *device2;
  
  // Memory Allocation: Allocate a new, larger table on the device.
  cudaMalloc((void **)&device2, new_size);
  cudaMemset(device2, 0, new_size);
  
  // Launch the kernel to rehash elements into the new table.
  kernel_reshape >> (hashTable, device2, numBucketsReshape, hashSize);
  cudaDeviceSynchronize();

  // Free the old table memory.
  cudaFree(hashTable);
  
  // Update the class members to point to the new table.
  hashTable = device2;
  hashSize = numBucketsReshape;
}

/**
 * @brief Inserts a batch of keys and values into the hash table.
 * @return True on success.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
  const int num_elements = numKeys;
  size_t blocks_no = num_elements / block_size;
  if (num_elements % block_size)
    ++blocks_no;
  
  int neededSize = hashElems + numKeys;
  
  // Block Logic: Check if the current capacity is sufficient. If not, reshape.
  if (hashSize < neededSize) {
    // Calculate a new size with a 10% buffer.
    int mem = neededSize * 110 / 100;
    if (mem < 1)
      mem = 1;
    reshape(mem);
  }

  int *keys_device = 0;
  int *values_device = 0;
  int *insert_values_device = 0;
  // Memory Allocation: Allocate temporary device buffers for the batch.
  cudaMalloc((void **)&keys_device, numKeys * sizeof(int));
  cudaMalloc((void **)&values_device, numKeys * sizeof(int));
  cudaMalloc((void **)&insert_values_device, 1 * sizeof(int));

  // Memory Transfer: Copy batch from host to device.
  cudaMemcpy(keys_device, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(values_device, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(insert_values_device, 0, 1 * sizeof(int));
  
  int *new_elems = (int *) calloc(1, sizeof(int));

  // Launch the insertion kernel.
  kernel_insert_batches >> (hashTable, keys_device, values_device, numKeys, hashSize, insert_values_device);
  cudaDeviceSynchronize();

  // Memory Transfer: Copy the number of new elements back to the host.
  cudaMemcpy(new_elems, insert_values_device, sizeof(int), cudaMemcpyDeviceToHost);
  // Update the host-side element count.
  hashElems += new_elems[0];

  // Free temporary memory.
  cudaFree(insert_values_device);
  cudaFree(keys_device);
  cudaFree(values_device);
  free(new_elems);
  return true;
}

/**
 * @brief Retrieves values for a batch of keys.
 * @return A host pointer to an array of results. Caller must free this memory.
 */
int *GpuHashTable::getBatch(int * keys, int numKeys) {
  const int num_elements = numKeys;
  size_t blocks_no = num_elements / block_size;
  if (num_elements % block_size)
    ++blocks_no;
  int *keys_device = 0;
  int *values_array;
  int *host_values = (int *)calloc(numKeys, sizeof(int));

  // Memory Allocation and Transfer for keys.
  cudaMalloc((void **)&keys_device, numKeys * sizeof(int));
  cudaMemcpy(keys_device, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

  // Memory Allocation for the output values on the device.
  cudaMalloc((void **)&values_array, numKeys * sizeof(int));
  cudaMemset(values_array, 0, numKeys * sizeof(int));

  // Launch the get kernel.
  kernel_get_batches >> (hashTable, values_array, keys_device, numKeys, hashSize);
  cudaDeviceSynchronize();

  // Memory Transfer: Copy results from device to host.
  cudaMemcpy(host_values, values_array, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

  // Free temporary device memory.
  cudaFree(keys_device);
  cudaFree(values_array);
  return host_values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 */
float GpuHashTable::loadFactor() {
  return (float) hashElems / hashSize;
}


// --- Convenience macros for external API ---
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

// Value representing an empty slot in the hash table.
#define	KEY_INVALID		0

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
 * @struct my_elem
 * @brief Represents a key-value pair in the hash table.
 */
struct my_elem {
	uint32_t key;
	uint32_t value;
};

/**
 * @class GpuHashTable
 * @brief Host-side class for managing the GPU hash table.
 */
class GpuHashTable
{
	public:
		my_elem* hashTable; // Device pointer to the hash table data.
		int hashSize = 0;   // The capacity of the hash table.
		int hashElems = 0;  // The current number of elements.
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
