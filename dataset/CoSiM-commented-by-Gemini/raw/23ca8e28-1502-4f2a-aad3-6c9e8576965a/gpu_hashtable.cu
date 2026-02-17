/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA.
 * @details This file provides the 'GpuHashTable' class, a host-side interface for managing
 * a hash table that resides in GPU memory. The hash table uses open addressing with
 * linear probing for collision resolution. All major operations (insert, get, reshape)
 * are implemented as CUDA kernels to be executed in parallel on the GPU.
 *
 * The file has an unusual structure, including 'kernels.cu' and 'test_map.cpp' directly,
 * and re-defining parts of the interface. This suggests it's a component of a larger
 * project and is not intended to be a standalone, reusable library.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#define BLOCK_SIZE	1000
#define FACTOR  1.05

#include "gpu_hashtable.hpp"
#include "kernels.cu"


/**
 * @brief GpuHashTable constructor.
 * @param size Initial capacity of the hash table.
 * Allocates memory on the GPU for the key and value arrays and initializes them
 * to an invalid state.
 */
GpuHashTable::GpuHashTable(int size) {
	size_t sz;
	cudaError_t ret;

	this->maxSize = (size_t)size;
	this->currentSize = size_t(0);
	
	sz = this->maxSize * sizeof(int);

	// Allocate and initialize key array on the GPU.
	ret = cudaMalloc((void **)&this->hKeys, sz);
	DIE(ret != cudaSuccess, "cudaMallocManaged failed");
	ret = cudaMemset(this->hKeys, KEY_INVALID, sz);
	DIE(ret != cudaSuccess, "cudaMemset failed");

	// Allocate and initialize value array on the GPU.
	ret = cudaMalloc((void **)&this->hValues, sz);
	DIE(ret != cudaSuccess, "cudaMallocManaged failed");
	ret = cudaMemset(this->hValues, VALUE_INVALID, sz);
	DIE(ret != cudaSuccess, "cudaMemset failed");
}

/**
 * @brief GpuHashTable destructor.
 * Frees all allocated GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(this->hKeys);
	this->hKeys = NULL;

	cudaFree(this->hValues);
	this->hValues = NULL;
}

/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The desired minimum new capacity. The actual new size
 * will be this value multiplied by FACTOR.
 * This function allocates new, larger GPU memory arrays and launches a kernel
 * to re-hash and copy all existing elements into the new arrays.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	size_t sz;
	cudaError_t ret;
	size_t newSize;
	int *helper_hKeys;
	int *helper_hValues;
	int nrBlocks;

	newSize = FACTOR * size_t(numBucketsReshape);
	sz = newSize * sizeof(int);

	// Allocate new, larger helper arrays on the GPU.
	ret = cudaMalloc((void **)&helper_hKeys, sz);
	DIE(ret != cudaSuccess, "cudaMallocManaged failed");
	ret = cudaMemset(helper_hKeys, KEY_INVALID, sz);
	DIE(ret != cudaSuccess, "cudaMemset failed");

	ret = cudaMalloc((void **)&helper_hValues, sz);
	DIE(ret != cudaSuccess, "cudaMallocManaged failed");
	ret = cudaMemset(helper_hValues, VALUE_INVALID, sz);
	DIE(ret != cudaSuccess, "cudaMemset failed");

	// If the table is not empty, re-hash and copy elements to the new arrays.
	if (this->currentSize != 0) {
		nrBlocks = this->maxSize / BLOCK_SIZE;
		if (this->maxSize % BLOCK_SIZE) {
			++nrBlocks;
		}

		// Launch the copy kernel.
		kernel_copy>>(this->hKeys, this->hValues, helper_hKeys,
												helper_hValues, this->maxSize, newSize);
		cudaDeviceSynchronize();
	}

	// Free old arrays and replace them with the new ones.
	cudaFree(this->hKeys);
	cudaFree(this.hValues);
	this->hKeys = helper_hKeys;


	this->hValues = helper_hValues;
	this->maxSize = newSize;
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys Pointer to the host array of keys.
 * @param values Pointer to the host array of values.
 * @param numKeys The number of key-value pairs to insert.
 * @return True on success, false on failure.
 * This function handles resizing, copies data to the GPU, and launches the
 * insertion kernel.
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *ks, *vs;
	size_t var;
	int *nr, nrBlocks;
	size_t sz = size_t(numKeys) * sizeof(int);
	cudaError_t ret1, ret2, ret3;

	// Pre-condition: Check if there is enough space and reshape if necessary.
	var = this->maxSize - this->currentSize;
	if (var < size_t(numKeys)) {
		this->reshape((int)this->maxSize + numKeys);
	}

	// Allocate temporary device memory for the batch.
	ret1 = cudaMalloc((void **)&ks, sz);
	ret2 = cudaMalloc((void **)&vs, sz);
	// 'nr' will count successful insertions on the GPU.
	ret3 = cudaMallocManaged((void **)&nr, sizeof(int));
	if (ret1 != cudaSuccess || ret2 != cudaSuccess
		|| ret3 != cudaSuccess) {
		return false;
	}

	// Copy the batch from host to device.
	ret1 = cudaMemcpy(ks, keys, sz, cudaMemcpyHostToDevice);
	ret2 = cudaMemcpy(vs, values, sz, cudaMemcpyHostToDevice);
	if (ret1 != cudaSuccess || ret2 != cudaSuccess) {
		return false;
	}

	*nr = 0;
	// Calculate grid configuration for the kernel launch.
	nrBlocks = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE) {
		++nrBlocks;
	}

	// Launch the insertion kernel.
	kernel_insert>>(this->hKeys, this->hValues, ks, vs,
									numKeys, this->maxSize, nr);
	cudaDeviceSynchronize();

	this->currentSize += size_t(*nr);
	// Clean up temporary device memory.
	cudaFree(ks);
	cudaFree(vs);
	cudaFree(nr);
	
	return true;
}

/**
 * @brief Retrieves values for a batch of keys from the hash table.
 * @param keys Pointer to the host array of keys to look up.
 * @param numKeys The number of keys to look up.
 * @return A pointer to a host array containing the corresponding values. The caller
 * is responsible for freeing this memory. Returns NULL on failure.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *values_h, *values_d, *ks;
	cudaError_t ret;
	int nrBlocks;
	size_t sz = size_t(numKeys) * sizeof(int);

	// Allocate host memory for the results.
	values_h = (int *)malloc(sz);
	if (!values_h) {
		return NULL;
	}

	// Allocate device memory for results and input keys.
	ret = cudaMalloc((void **)&values_d, sz);
	if (ret != cudaSuccess) {
		free(values_h);
		return NULL;
	}
	ret = cudaMalloc((void **)&ks, sz);
	if (ret != cudaSuccess) {
		free(values_h);
		cudaFree(values_d);
		return NULL;
	}

	nrBlocks = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE) {
		++nrBlocks;
	}
	
	// Copy keys to device, launch kernel, and copy results back to host.
	cudaMemcpy(ks, keys, sz, cudaMemcpyHostToDevice);
	kernel_getValues>>(this->hKeys, this->hValues, ks,
												values_d, numKeys, this->maxSize);
	cudaDeviceSynchronize();
	cudaMemcpy(values_h, values_d, sz, cudaMemcpyDeviceToHost);

	cudaFree(ks);
	cudaFree(values_d);

	return values_h;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float.
 */
float GpuHashTable::loadFactor() {
	return this->currentSize * 1.0f / this->maxSize; 
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

#define	KEY_INVALID		0
#define VALUE_INVALID	0

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)

// A list of prime numbers used for hashing.
__device__ const size_t primeList[] =
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
 * @brief The hash function used to map keys to indices in the hash table.
 * @param data The key to be hashed.
 * @param limit The size of the hash table, used for the modulo operation.
 * @return The calculated hash index.
 */
__device__ int MY_SUPER_HASH(int data, int limit) {
    return (int)(((long)abs(data) * primeList[130]) % primeList[150] % limit);
}




class GpuHashTable
{
	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		
		float loadFactor();
	
		~GpuHashTable();

	private:
        
		size_t maxSize;
        
		size_t currentSize;
        
		int *hKeys;
        
		int *hValues;
};

#endif

/**
 * @brief CUDA kernel for rehashing and copying elements to a new, larger table.
 * @details Each thread is responsible for one element from the old table. It re-hashes
 * the key for the new table size and uses linear probing with atomicCAS to find an
 * empty slot in the new table, resolving collisions.
 */
__global__ void kernel_copy(int *hKeys, int *hValues, int *helper_hKeys,
							int *helper_hValues, size_t size, size_t newSize) {
	// Map global thread ID to an element in the old table.
	unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
	int index;
	int key;

	if (id < size) {
		// Pre-condition: Only process valid keys from the old table.
		if (hKeys[id] != 0) {
			key = hKeys[id];

			// Calculate initial hash position in the new table.
			index = MY_SUPER_HASH(key, newSize);

			// Logic: Linear probing to find an empty slot.
			while (true) {
				if (helper_hKeys[index] == 0) {
					// Use atomicCAS to atomically claim an empty slot. This prevents
					// multiple threads from writing to the same slot simultaneously.
		            key = atomicCAS(&helper_hKeys[index], 0, key);
		            if (key == 0) {
						// If the slot was successfully claimed, write the value and exit.
		            	helper_hValues[index] = hValues[id];
						return;
					}
		        }
				// If the slot was already taken, continue probing.
				key = hKeys[id];
				index = (index + 1) % newSize;
			}
		}
	}
}

/**
 * @brief CUDA kernel for inserting a batch of key-value pairs.
 * @details Each thread handles one key-value pair from the input batch. It uses
 * linear probing and atomicCAS to find an empty slot or update an existing key.
 * An atomicAdd operation is used to count the number of new insertions.
 */
__global__ void kernel_insert(int *hKeys, int *hValues, int *keys, int *values,
								int nrKeys, size_t size, int *nr) {
	// Map global thread ID to an input key-value pair.
	unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
	int index;
	
	int key;

	if (id < nrKeys) {
		key = keys[id];
		index = MY_SUPER_HASH(key, size);

		// Logic: Linear probing loop.
		while (true) {
			// If key already exists, update its value and return.
			if (hKeys[index] == key) {
				hValues[index] = values[id];
				return;
			}
			// If an empty slot is found, try to claim it.
			if (hKeys[index] == 0) {
	            key = atomicCAS(&hKeys[index], 0, key);
	            if (key == 0) {
					// If claimed successfully, write the value and increment the counter.
					hValues[index] = values[id];
					atomicAdd(nr, 1);
					return;
				}
	        }

			// If the slot was taken by another thread in the meantime, continue probing.
			key = keys[id];
			index = (index + 1) % size;
		}
	}
}

/**
 * @brief CUDA kernel for retrieving values for a batch of keys.
 * @details Each thread is responsible for looking up one key. It performs linear
 * probing starting from the key's hash index until the key is found, then writes
 * the corresponding value to the output array.
 */
__global__ void kernel_getValues(int *hKeys, int *hValues, int *keys, int *values,
								int nrKeys, size_t size) {
	// Map global thread ID to an input key.
	unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int index;

	if (id < nrKeys) {
		index = MY_SUPER_HASH(keys[id], size);

		// Logic: Linear probing to find the key.
		while (true) {
			if (hKeys[index] == keys[id]) {
				values[id] = hValues[index];
				return;
			}
			// This assumes keys are always present. A robust implementation would
			// add a check for cycling through the whole table to handle missing keys.
			index = (index + 1) % size;
		}
	}
}