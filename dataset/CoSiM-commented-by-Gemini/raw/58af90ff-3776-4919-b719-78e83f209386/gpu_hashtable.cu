/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 * @details This file defines a hash table that resides in GPU memory and is manipulated
 * through CUDA kernels. It supports batch insertion and retrieval operations. The collision
 * resolution strategy is linear probing, and thread safety is ensured using atomic operations.
 * The hash table can also be dynamically resized.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#include "gpu_hashtable.hpp"


#if 0
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}                                                                   
#else
/**
 * @def cudaCheckError
 * @brief A macro for CUDA error checking. Disabled in the current build configuration.
 * @details When enabled, this macro checks the last CUDA error and prints it to stderr
 * if an error has occurred, then exits the program.
 */
#define cudaCheckError() 
#endif


/**
 * @brief Constructs a GpuHashTable object.
 * @param size The initial size of the hash table.
 * @details Allocates memory on the GPU for the hash table and initializes it to zero.
 */
GpuHashTable::GpuHashTable(int size) {
	
	crtLimit = size;

	hshEntry *ntab;
	// Allocate memory for the hash table on the GPU.
	cudaMalloc((void **) &ntab, size * sizeof(hshEntry));
	if(ntab == 0) {
		fprintf(stderr, "[HOST] Couldn't allocate memory\n");
		fflush(stderr);
		return;
	}
	table = ntab;
	cudaCheckError();
	// Initialize the allocated GPU memory to zero.
	cudaMemset(table, 0, size * sizeof(hshEntry));
	cudaCheckError();
	cudaDeviceSynchronize();
	cudaCheckError();
	occupied = 0;
}


/**
 * @brief Destroys the GpuHashTable object.
 * @details Frees the GPU memory allocated for the hash table.
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(table);
	cudaCheckError();
	cudaDeviceSynchronize();
	cudaCheckError();
	crtLimit = 0;
}

/**
 * @var count
 * @brief A device-side global variable used as an atomic counter.
 * @details This counter is used in kernels to assign unique indices to threads, for example, 
 * when compacting data.
 */
__device__ unsigned int count;



/**
 * @brief CUDA kernel to extract key-value pairs from the hash table.
 * @param table Pointer to the hash table in GPU memory.
 * @param keys Pointer to an array in GPU memory to store the extracted keys.
 * @param values Pointer to an array in GPU memory to store the extracted values.
 * @param limit The size of the hash table.
 * @details Each thread checks a slot in the hash table. If the slot is occupied,
 * the thread atomically increments a counter to get a unique index and writes
 * the key-value pair to the output arrays at that index.
 */
__global__ void extractKeyVals(hshEntry *table, int *keys, int *values, int limit) {
	int tnr = threadIdx.x + blockDim.x * blockIdx.x;
	// The first thread initializes the atomic counter.
	if(tnr == 0) {
		count = 0;
	}
	__syncthreads();
	if(tnr < limit) {
		// Check if the hash table entry is occupied.
		if(table[tnr].key != 0) {
			// Atomically increment the counter to get a unique index for writing.
			int nxt = atomicInc(&count, limit);
			keys[nxt] = table[tnr].key;
			values[nxt] = table[tnr].value;
		}
	}
}


/**
 * @brief Resizes the hash table.
 * @param numBucketsReshape The new size of the hash table.
 * @details This function extracts all key-value pairs from the current hash table,
 * allocates a new, larger table on the GPU, and re-inserts all the pairs into the new table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int blockSize = (int) min(1024.0f, sqrtf(crtLimit));
	int *dkeys, *dvalues, *keys, *values;
	// Allocate temporary storage on the GPU for keys and values.
	cudaMalloc((void **) &dkeys, crtLimit * sizeof(int));
	cudaMalloc((void **) &dvalues, crtLimit * sizeof(int));
	
	// Launch the kernel to extract key-value pairs.
	extractKeyVals>>(table, dkeys, dvalues, crtLimit);
	cudaDeviceSynchronize();
	cudaDeviceSynchronize();
	cudaCheckError();
	
	unsigned int *rcount;
	rcount = (unsigned int *) malloc(sizeof(unsigned int));
	
	// Copy the number of occupied slots from the device to the host.
	cudaError_t err = cudaMemcpyFromSymbol(rcount, count, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
	
	// Allocate host memory for the keys and values.
	keys = (int *) malloc(sizeof(int) * *rcount);
	values = (int *) malloc(sizeof(int) * *rcount);
	cudaCheckError();
	
	// Copy the keys and values from the GPU to the host.
	cudaMemcpy(keys, dkeys, sizeof(int) * *rcount, cudaMemcpyDeviceToHost);
	cudaMemcpy(values, dvalues, sizeof(int) * *rcount, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	// Free the temporary GPU memory.
	cudaFree(dkeys);
	cudaCheckError();
	cudaFree(dvalues);
	cudaCheckError();
	
	// Free the old hash table.
	cudaFree(table);
	cudaCheckError();
	cudaDeviceSynchronize();
	cudaCheckError();
	crtLimit = numBucketsReshape;
	hshEntry *ntab;
	// Allocate a new, larger hash table on the GPU.
	cudaMalloc((void **) &ntab, numBucketsReshape * sizeof(hshEntry));
	if(ntab == 0) {
		return;
	}
	table = ntab;
	cudaCheckError();
	// Initialize the new hash table to zero.
	cudaMemset(table, 0, numBucketsReshape * sizeof(hshEntry));
	cudaCheckError();
	cudaDeviceSynchronize();
	cudaCheckError();
	
	// Re-insert the key-value pairs into the new table.
	insertBatch(keys, values, *rcount);
	free(keys);
	free(values);
}





/**
 * @brief CUDA kernel to insert a batch of key-value pairs.
 * @param table Pointer to the hash table in GPU memory.
 * @param keys Pointer to an array of keys to insert.
 * @param values Pointer to an array of values to insert.
 * @param limit The size of the hash table.
 * @details Each thread handles one key-value pair. It computes the hash of the key and uses
 * atomic compare-and-swap (atomicCAS) to find an empty slot, implementing linear probing for
 * collision resolution.
 */
__global__ void insertSingle(hshEntry *table, int *keys, int *values, int limit) {
	int tnr = threadIdx.x + blockDim.x * blockIdx.x;
	if(tnr < limit) {
		int key = keys[tnr], value = values[tnr];
		int hsh = hash3(key, limit);
		int inihsh = hsh;
		// Linear probing to find an empty slot or a slot with the same key.
		while(atomicCAS(&(table[hsh].key), 0, key) && table[hsh].key != key) {
			hsh = (hsh + 1) % limit;
			if(hsh == inihsh) break;
		};
		
		table[hsh].value = value;
	}
}



/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys An array of keys to insert.
 * @param values An array of values to insert.
 * @param numKeys The number of keys and values in the batch.
 * @return True if the insertion was successful, false otherwise.
 * @details If the load factor exceeds a threshold, this function first resizes the table.
 * It then copies the keys and values to the GPU and launches a kernel to perform the insertion.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	if(numKeys == 0)
		return false;
	// Check if the hash table needs to be resized.
	if(numKeys + occupied >= 0.9 * crtLimit) {
		reshape((numKeys + occupied) * 1.245);
		
	}
	int blockSize = (int) min(1024.0f, sqrtf(numKeys));
	while(numKeys % blockSize) {
		blockSize--;
	}
	int *dkeys, *dvalues;
	// Allocate GPU memory for the keys and values and copy them from the host.
	cudaMalloc((void **) &dkeys, sizeof(int) * numKeys);
	cudaMemcpy(dkeys, keys, sizeof(int) * numKeys, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &dvalues, sizeof(int) * numKeys);
	cudaMemcpy(dvalues, values, sizeof(int) * numKeys, cudaMemcpyHostToDevice);
	// Launch the insertion kernel.
	insertSingle>>(table, dkeys, dvalues, crtLimit);
	cudaCheckError();
	occupied += numKeys;
	cudaError_t err = cudaDeviceSynchronize();
	cudaCheckError();
	cudaFree(dkeys);
	cudaFree(dvalues);
	return (err == cudaSuccess);
}


/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param table Pointer to the hash table in GPU memory.
 * @param keys Pointer to an array of keys to look up.
 * @param limit The size of the hash table.
 * @param result Pointer to an array in GPU memory to store the retrieved values.
 * @details Each thread handles one key. It computes the hash and uses linear probing
 * to find the key in the table. If found, the corresponding value is written to the result array.
 */
__global__ void getSingle(hshEntry *table, int *keys, int limit, int *result) {


	int tnr = threadIdx.x + blockDim.x * blockIdx.x;
	if(tnr < limit) {
		int key = keys[tnr];
		int hsh = hash3(key, limit);
		int inihsh = hsh;
		// Linear probing to find the key.
		while(table[hsh].key != key) {
			hsh = (hsh + 1) % limit;
			if(hsh == inihsh) break;
		};
		result[tnr] = table[hsh].value;
	}
}


/**
 * @brief Retrieves values for a batch of keys.
 * @param keys An array of keys to look up.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to an array containing the retrieved values. The caller is responsible for freeing this memory.
 * @details This function copies the keys to the GPU, launches a kernel to retrieve the values, 
 * and then copies the results back to the host.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int blockSize = (int) min(1024.0f, sqrtf(numKeys));
	
	int *devres, *dkeys;
	// Allocate GPU memory for keys and results.
	cudaMalloc((void **) &dkeys, sizeof(int) * numKeys);
	cudaMemcpy(dkeys, keys, sizeof(int) * numKeys, cudaMemcpyHostToDevice);
	cudaMalloc(&devres, numKeys * sizeof(int));
	
	cudaDeviceSynchronize();
	cudaCheckError();
	// Launch the retrieval kernel.
	getSingle>>(table, dkeys, crtLimit, devres);
	
	int *res = (int*) malloc(numKeys * sizeof(int));
	cudaDeviceSynchronize();
	cudaCheckError();
	// Copy the results from the GPU to the host.
	cudaMemcpy(res, devres, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	cudaCheckError();
	cudaFree(devres);
	cudaFree(dkeys);
	cudaCheckError();
	return res;
}


/**
 * @brief Calculates the load factor of the hash table.
 * @return The load factor (ratio of occupied slots to total slots).
 */
float GpuHashTable::loadFactor() {
	return ((float) occupied / (float) crtLimit); 
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
	do { \
		if (assertion) { \
		fprintf(stderr, "(%s, %d): ", \
		__FILE__, __LINE__); \
		perror(call_description); \
		exit(errno); \
	} \
} while (0)
	
/**
 * @var primeList
 * @brief A list of prime numbers used for hashing.
 * @details This list provides a set of large prime numbers that can be used in hash functions
 * to help distribute keys more uniformly.
 */
__device__ const size_t primeList[] =
{
	2llu, 3llu, 5llu, 7llu, 11llu, 13llu, 17llu, 23llu, 29llu, 37llu, 47llu,
	59llu, 73llu, 97llu, 127llu, 151llu, 197llu, 251llu, 313llu, 397llu, 499llu,
	631llu, 797llu, 1009llu, 1259llu, 1597llu, 2011llu, 2539llu,
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
 * @brief Computes a hash for a given integer key.
 * @param data The integer key to hash.
 * @param limit The size of the hash table, used to scale the hash value.
 * @return The computed hash value.
 * @details This is a device function that can be called from CUDA kernels.
 */
__device__ int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

typedef struct __hshEntry {
	int key, value;
} hshEntry;




class GpuHashTable
{
	int crtLimit, occupied;
	hshEntry *table;
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
