/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 *
 * This implementation uses an open addressing scheme with linear probing for
 * collision resolution. It relies on explicit host-device memory management
 * (`cudaMalloc`, `cudaMemcpy`) and uses atomic operations to handle concurrent
 * insertions.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define PRIME_A 653267llu
#define PRIME_B 107888587883llu
#define MIN_LFACTOR 0.8f
#define MAX_LFACTOR 0.9f

/**
 * @brief Computes a hash value for a given key on the GPU.
 * @param data The input key to hash.
 * @param limit The size of the hash table for the modulo operation.
 * @return The computed hash index.
 *
 * This function uses a MurmurHash-style finalizer with bitwise operations and
 * multiplications to provide good key distribution.
 */
__device__
int hash_function(int data, int limit) {
    return ((long)abs(data) * PRIME_A) % PRIME_B % limit;
}


/**
 * @brief CUDA kernel to rehash all elements from an old table to a new one.
 * @param oldHashTable The source (old) hash table.
 * @param newHashTable The destination (new) hash table.
 *
 * Note: This kernel appears to contain a bug. It uses the thread index `index`
 * (from the old table) to write into the new table, instead of using the newly
 * computed hash `hashValue`. This will lead to incorrect element placement and
 * data corruption if the table sizes differ.
 */
__global__
void kernel_reshape(hashTable oldHashTable, hashTable newHashTable) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int oldKey, newKey;

    if (index >= oldHashTable.size) {
        return;
    }
    newKey = oldHashTable.map[index].key;

    bool found = false;
    int hashValue = hash_function(newKey, newHashTable.size);

    for (int i = hashValue; i < newHashTable.size; i++) {
        oldKey = atomicCAS(&newHashTable.map[i].key, 0, newKey);
        if (oldKey == 0) {
            newHashTable.map[i].value = oldHashTable.map[index].value;
            found = true;
            break;
        }
    }

    for (int i = 0; i < hashValue && !found; i++) {
        oldKey = atomicCAS(&newHashTable.map[i].key, 0, newKey);
        if (oldKey == 0) {
            newHashTable.map[i].value = oldHashTable.map[index].value;
            break;
        }
    }
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs into the hash table.
 * @param hashMap The hash table to insert into.
 * @param device_keys Pointer to the batch of keys on the device.
 * @param device_values Pointer to the batch of values on the device.
 * @param numKeys The number of pairs to insert.
 */
__global__
void kernel_insert(hashTable hashMap, int *device_keys, int *device_values, int numKeys) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int oldKey, newKey;

    if (index >= numKeys) {
        return;
    }
    newKey = device_keys[index];

    bool found = false;
    // Step 1: Calculate the initial hash position.
    int hashValue = hash_function(newKey, hashMap.size);

    // Step 2: Linearly probe forward from the hash position.
    for (int i = hashValue; i < hashMap.size; i++) {
        // Attempt to claim an empty slot (key == 0) or find the existing key.
        oldKey = atomicCAS(&hashMap.map[i].key, 0, newKey);
        if (newKey == oldKey || oldKey == 0) {
            hashMap.map[i].value = device_values[index];
            found = true;
            return;
        }
    }

    // Step 3: Handle wrap-around if no slot was found.
    for (int i = 0; i < hashValue && !found; i++) {
        oldKey = atomicCAS(&hashMap.map[i].key, 0, newKey);
        if (newKey == oldKey || oldKey == 0) {
            hashMap.map[i].value = device_values[index];
            return;
        }
    }
}

/**
 * @brief CUDA kernel to retrieve a batch of values for a given set of keys.
 * @param hashMap The hash table to search.
 * @param device_keys The keys to look up.
 * @param values The output array for retrieved values.
 * @param numKeys The number of keys to process.
 */
__global__
void kernel_get(hashTable hashMap, int *device_keys, int *values, int numKeys) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int keyToGet;
    
    if (index >= numKeys) {
        return;
    }

    keyToGet = device_keys[index];

    bool found = false;
    int hashValue = hash_function(device_keys[index], hashMap.size);

    // Linearly probe to find the key.
    for (int i = hashValue; i < hashMap.size; i++) {
        if (hashMap.map[i].key == keyToGet) {
            values[index] = hashMap.map[i].value;
            found = true;
            return;
        }
    }
    // Handle wrap-around if not found.
    for (int i = 0; i < hashValue && !found; i++) {
        if (hashMap.map[i].key == keyToGet) {
            values[index] = hashMap.map[i].value;
            return;
        }
    }
}


/**
 * @brief Host-side constructor for the GpuHashTable class.
 * @param size The initial capacity of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
    hashMap.size = size;
    nrEntries = 0;

    // Allocate memory on the device for the hash table array.
    cudaMalloc((void **) &hashMap.map, size * sizeof(hashItem));
    if (hashMap.map == 0) {
        printf("[HOST] Couldn't allocate memory [INIT]
");
        return;
    }

    // Initialize the device memory to zero.
    cudaMemset(hashMap.map, 0, size * sizeof (hashItem));
}

/**
 * @brief Host-side destructor for the GpuHashTable class.
 */
GpuHashTable::~GpuHashTable() {
    cudaFree(hashMap.map);
}

/**
 * @brief Resizes the hash table, preserving existing data.
 * @param numBucketsReshape The new capacity for the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
    int block_size = 512;
    hashTable newHashMap;
    newHashMap.size = numBucketsReshape;

    // Allocate a new, larger table on the device.
    cudaMalloc((void **) &newHashMap.map, numBucketsReshape * sizeof (hashItem));
    if (newHashMap.map == 0) {
        printf("[HOST] Couldn't allocate memory [RESHAPE]
");
        return;
    }
    cudaMemset(newHashMap.map, 0, numBucketsReshape * sizeof (hashItem));

    unsigned int blocks_no = hashMap.size / block_size;
    if (hashMap.size % block_size) {
        ++blocks_no;
    }

    // Launch the kernel to migrate data. Note: this kernel has a potential bug.
    kernel_reshape>> (hashMap, newHashMap);
    cudaDeviceSynchronize();

    // Free the old table memory.
    cudaFree(hashMap.map);
    // Replace the old table with the new one.
    hashMap = newHashMap;
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys Host pointer to an array of keys.
 * @param values Host pointer to an array of values.
 * @param numKeys The number of pairs to insert.
 * @return True on success, false on failure.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys_array, *device_values_array;
	int numBytes = numKeys * sizeof (int);
	int block_size = 512;

	// Allocate temporary device buffers for the batch.
	cudaMalloc((void **) &device_keys_array, numBytes);
	cudaMalloc((void **) &device_values_array, numBytes);
	if (device_keys_array == 0 || device_values_array == 0) {
	    printf("[HOST] Couldn't allocate memory [INSERT_BATCH]
");
	    return false;
	}

	// Check the load factor and reshape if necessary.
	float currLFACTOR = (float) ((nrEntries + numKeys) / hashMap.size);
	if (currLFACTOR >= MAX_LFACTOR) {
	    reshape((int) ((nrEntries + numKeys) / MIN_LFACTOR));
	}

	// Copy data from host to device.
	cudaMemcpy(device_keys_array, keys, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_values_array, values, numBytes, cudaMemcpyHostToDevice);

	unsigned int blocks_no = numKeys / block_size;
	if (numKeys % block_size) {
	    ++blocks_no;
	}

	// Launch the insertion kernel.
	kernel_insert>> (hashMap, device_keys_array, device_values_array, numKeys);
	cudaDeviceSynchronize();

	// Increment the entry count. Note: this is inaccurate as it does not
    // account for updates to existing keys, only new insertions.
    nrEntries += numKeys;

    // Free temporary buffers.
    cudaFree(device_keys_array);
    cudaFree(device_values_array);
    return true;
}

/**
 * @brief Retrieves a batch of values for a given set of keys.
 * @param keys Host pointer to an array of keys.
 * @param numKeys The number of keys to retrieve.
 * @return A host pointer to an array with the results. Caller is responsible for freeing this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_keys_array, *device_values_array;
    size_t numBytes = numKeys * sizeof (int);
    int block_size = 512;

    // Allocate temporary device buffers.
	cudaMalloc ((void **) &device_keys_array, numBytes);
	cudaMalloc((void **) &device_values_array, numBytes);

	// Allocate host memory for the final results.
	int *valuesToReturn = (int *) malloc (numBytes);
	if (device_keys_array == 0 || device_values_array == 0 || valuesToReturn == 0) {
	    printf("[HOST]  Couldn't allocate memory [GET_BATCH]
");
	    return nullptr;
	}

	// Copy input keys from host to device.
	cudaMemcpy(device_keys_array, keys, numBytes, cudaMemcpyHostToDevice);

	unsigned int blocks_no = numKeys / block_size;
	if (numKeys % block_size) {
	    ++blocks_no;
	}

	// Launch the retrieval kernel.
	kernel_get>>(hashMap, device_keys_array, device_values_array, numKeys);
	cudaDeviceSynchronize();

	// Copy results from device back to the host buffer.
	cudaMemcpy(valuesToReturn, device_values_array, numBytes, cudaMemcpyDeviceToHost);

	// Free temporary device buffers.
	cudaFree(device_keys_array);
	cudaFree(device_values_array);
	
	return valuesToReturn;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float.
 */
float GpuHashTable::loadFactor() {
	return (hashMap.size == 0)? 0.f : (float (nrEntries) / hashMap.size);
}


// Macros for a simplified testing interface.
#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
// Header guard and definitions.
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0

// Macro for basic error handling.
#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)
	
// Pre-computed list of prime numbers.
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



// Alternative hash functions.
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}


// C-style struct for a hash table entry.
struct hashItem {
    int key, value;
};

// C-style struct to manage the hash table's state.
struct hashTable {
    int size;
    hashItem *map;
};

// The C++ class that provides a high-level API for the GPU hash table.
class GpuHashTable
{
	public:
        hashTable hashMap;
        int nrEntries;

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
