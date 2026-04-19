/**
 * @file gpu_hashtable.cu
 * @brief GPGPU Hash Table with linear probing and duplicate tracking.
 * 
 * This implementation features a single-hash linear probing algorithm with manual 
 * circularity handling. It manages global memory entries and attempts to track 
 * redundant key insertions via a dedicated device-side counter.
 * 
 * Algorithm: Open addressing with two-loop linear probing.
 * Memory Model: Global memory (cudaMalloc) for entry buffers.
 * Domain: HPC, Parallel Data Structures.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief Device-side hash function using a multiplicative constant.
 */
__device__ int myHash(int data, int limit) {
	return ((long)abs(data) * 2654435761llu) % 4294967296llu % limit;
}

/**
 * @brief CUDA kernel for parallel entry insertion.
 * 
 * Functional Utility: Claims empty slots using atomicCAS. Employs a manual 
 * circular probe strategy (two distinct loops) to resolve collisions.
 * 
 * @param duplicate Pointer to an integer for tracking redundant insertions.
 */
__global__ void kernel_insert(int *keys, int *values, int limitBound, hashtable hashmap, int *duplicate) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= limitBound) {
		return;
	}

	int extractKey = keys[i], size = hashmap.size, oldK, check = 0;
	int hash = myHash(extractKey, size);

	/**
	 * Block Logic: Forward probe sequence.
	 * Logic: Searches from the initial hash index to the end of the memory buffer.
	 */
	for (int k = hash; k < size; ++k) {
		oldK = atomicCAS(&hashmap.list[k].key, KEY_INVALID, extractKey);
		if (oldK == KEY_INVALID || oldK == extractKey) {
			hashmap.list[k].value  = values[i];
			check = 1;
			if (oldK == extractKey)
				*duplicate++; // Note: Incrementing the pointer itself (potential logic bug in original).
			break;
		}
	}

	/**
	 * Block Logic: Wrap-around probe sequence.
	 * Logic: Searches from the start of the buffer to the original hash index.
	 */
	if (check == 0) {
		for (int k = 0; k < hash; ++k) {
			oldK = atomicCAS(&hashmap.list[k].key, KEY_INVALID, extractKey);
			if (oldK == KEY_INVALID || oldK == extractKey) {
				hashmap.list[k].value  = values[i];
				if (oldK == extractKey)
					*duplicate++;
				return;
			}
		}
	}
 
	return;
}

/**
 * @brief CUDA kernel for parallel value retrieval.
 * 
 * Functional Utility: Implements a two-pass linear search to find values 
 * mapped to specific keys in the concurrent hash table.
 */
__global__ void kernel_get(int *keys, int *values, int limitBound, hashtable hashmap) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= limitBound) {
		return;
	}

	int extractKey = keys[i], size = hashmap.size, check = 0;
	int hash = myHash(extractKey, size);

	/**
	 * Block Logic: Forward search pass.
	 */
	for (int k = hash; k < size; ++k) {


		if (hashmap.list[k].key == extractKey) {
			values[i] = hashmap.list[k].value;
			check = 1;
			return;
		}
	}

	/**
	 * Block Logic: Backward wrap-around search pass.
	 */
	if (check == 0) {
		for (int k = 0; k < hash; ++k) {


			if (hashmap.list[k].key == extractKey) {
				values[i] = hashmap.list[k].value;
				return;
			}
		}
	}

}

/**
 * @brief CUDA kernel for data migration during table expansion.
 */
__global__ void kernel_reshape(hashtable hashmap, hashtable newHashmap) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= hashmap.size) {
		return;
	}

	// Logic: Re-inserts valid entries into the new structure.
	if (hashmap.list[i].key != KEY_INVALID) {
		int extractKey = hashmap.list[i].key, oldK, isInserted = 0, size = newHashmap.size;
		int hash = myHash(extractKey, size);

		for (int k = hash; k < size && !isInserted; ++k) {
			oldK = atomicCAS(&newHashmap.list[k].key, KEY_INVALID, extractKey);
			if (oldK == KEY_INVALID) {
				newHashmap.list[k].value  = hashmap.list[i].value;
				isInserted = 1;
				break;
			}
		}

		if (isInserted == 0) {
			for (int k = 0; k < hash && !isInserted; ++k) {
				oldK = atomicCAS(&newHashmap.list[k].key, KEY_INVALID, extractKey);
				if (oldK == KEY_INVALID) {
					newHashmap.list[k].value  = hashmap.list[i].value;
					isInserted = 1;
					break;
				}
			}
		}
	}

}


/**
 * @brief Constructor: Initializes the hash map on the GPU.
 */
GpuHashTable::GpuHashTable(int size) {
	length = 0;
	hashmap.size = size;
	hashmap.list = nullptr;
	if (cudaMalloc(&hashmap.list, size * sizeof(entry)) != cudaSuccess) {
		std::cerr << "Memory allocation error\n";


		return;
	}
	cudaMemset(hashmap.list, 0, size * sizeof(entry));  
}


/**
 * @brief Destructor: Frees GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashmap.list);
}


/**
 * @brief Rebuilds the hash table with a new size.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hashtable newHashmap;
	newHashmap.size = numBucketsReshape;
	if (cudaMalloc(&newHashmap.list, numBucketsReshape * sizeof(entry)) != cudaSuccess) {
		std::cerr << "Memory allocation error\n";
		return;
	}
	cudaMemset(newHashmap.list, 0, numBucketsReshape * sizeof(entry));

	int numBlocks = ((hashmap.size % 1024 == 0) ? (hashmap.size / 1024) : (hashmap.size / 1024 + 1));
	kernel_reshape>>(hashmap, newHashmap);

	cudaDeviceSynchronize();


	cudaFree(hashmap.list);
	hashmap = newHashmap;
}


/**
 * @brief Batch insertion with automatic resizing.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *deviceKeys, *deviceValues, *duplicate;
	int dupl;
	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	cudaMalloc(&deviceValues, numKeys * sizeof(int));
	cudaMalloc(&duplicate, sizeof(int));
	if (!deviceKeys || !deviceValues) {
		std::cerr << "Memory allocation error\n";
		return false;
	}
	
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemset(duplicate, 0, sizeof(int));

	 // Adaptive Scaling: Resizes if the incoming batch might overflow the table.
	 if (float(length + numKeys) >= hashmap.size) {                                             
                reshape(int(((length + numKeys) / 0.83)));                                                                                                                                                                  }
	int numBlocks = ((numKeys % 1024 == 0) ? (numKeys / 1024) : (numKeys / 1024 + 1));
	kernel_insert>>(deviceKeys, deviceValues, numKeys, hashmap, duplicate);
	cudaMemcpy(&dupl, duplicate, sizeof(int), cudaMemcpyDeviceToHost);
	int dup = dupl;

	cudaDeviceSynchronize();
	// Logic: Updates host-side element count, compensating for duplicates.
	length += numKeys;
	length -= dup;

	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	return true;
}


/**
 * @brief Batch retrieval using Managed Memory for output.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *deviceKeys, *values;
	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	cudaMallocManaged(&values, numKeys * sizeof(int));

	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int numBlocks = ((numKeys % 1024 == 0) ? (numKeys / 1024) : (numKeys / 1024 + 1));
	kernel_get>>(deviceKeys, values, numKeys, hashmap);

	cudaDeviceSynchronize();

	cudaFree(deviceKeys);
	return values;
}


/**
 * @brief Current load factor.
 */
float GpuHashTable::loadFactor() {
	
	if (hashmap.size == 0) {
		return 0.f;
	}
	return (float(length) / hashmap.size);
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

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
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
 * @struct entry
 * @brief Representation of a key-value mapping on the device.
 */
typedef struct entry {
	int key;
	int value;
} entry;


/**
 * @struct hashtable
 * @brief Device-resident hash table container.
 */
typedef struct hashtable {
	int size;
	entry *list;
} hashtable;




/**
 * @class GpuHashTable
 * @brief Host controller for GPU-based hash mapping.
 */
class GpuHashTable
{	
	int length;         // Host-side tracking of valid elements.
	hashtable hashmap;  // Metadata for the device structure.
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

