/**
 * @file gpu_hashtable.cu
 * @brief CUDA implementation of a hash table with linear probing.
 *
 * This file provides a GPU-accelerated hash table that uses linear probing for
 * collision resolution. It is designed for batch insertion and retrieval of
 * key-value pairs. The implementation includes CUDA kernels for insertion,
 * retrieval, and rehashing, along with a host-side class to manage the hash
 * table's lifecycle and operations.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief Computes the hash for a given key.
 * @param key The key to hash.
 * @param size The size of the hash table.
 * @return The computed hash value.
 */
__device__ unsigned int hash_func(int key, int size) {

	return (((long)key * PRIME_A) % PRIME_B % size);
}


/**
 * @brief CUDA kernel to insert a batch of key-value pairs.
 * @param keys A device pointer to the array of keys.
 * @param values A device pointer to the array of values.
 * @param numKeys The number of pairs to insert.
 * @param hashtable A device pointer to the hash table.
 * @param size The size of the hash table.
 * @param updates_count A device pointer to a counter for the number of updates.
 *
 * This kernel uses linear probing and atomic compare-and-swap (atomicCAS) to
 * handle collisions and ensure thread-safe insertion.
 */
__global__ void insert_pair(int *keys, int *values, int numKeys, struct my_pair *hashtable,
				int size, int* updates_count) {

	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= numKeys)
		return;
	if (keys[index] <= 0 || values[index] <= 0) {
		
		return;
	}

	int old_key;
	int key = keys[index];
	int value = values[index];
	unsigned int pos = hash_func(keys[index], size);
	unsigned int i = 0;

	// Invariant: The loop continues until the key is inserted or updated.
	while(true) {

		
		
		pos = (pos + i) % size;

		
		
		old_key = atomicCAS(&hashtable[pos].key, 0, key);

		if (old_key == key) {
			
			hashtable[pos].value = value;
			
			atomicAdd(updates_count, 1);
			return;
		}
		else if (old_key == 0) {
			
			
			hashtable[pos].value = value;
			return;
		}
		else {
			
			i++;
		}
	}
}


/**
 * @brief CUDA kernel to copy entries from an old to a new hash table during reshape.
 * @param old_pairs The old hash table.
 * @param new_pairs The new hash table.
 * @param size The size of the old table.
 * @param new_size The size of the new table.
 */
__global__ void copy_values(struct my_pair *old_pairs, struct my_pair *new_pairs, 
															int size, int new_size) {
	
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= size)
		return;

	if (old_pairs[index].key == 0)
		return;

	unsigned int pos = hash_func(old_pairs[index].key, new_size);
	unsigned int i = 0;

	int old_key = old_pairs[index].key;
	int old_value = old_pairs[index].value;
	int value;

	// Invariant: This loop re-inserts an entry into the new, larger hash table,
	// using linear probing to resolve collisions.
	while(true) {

		pos = (pos + i) % new_size;

		
		
		value = atomicCAS(&new_pairs[pos].key, 0, old_key);
		if (value == 0) {
			new_pairs[pos].value = old_value;
			return;
		}
		i++;
	}

}

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param keys A device pointer to the array of keys.
 * @param numKeys The number of keys to look up.
 * @param hashtable A device pointer to the hash table.
 * @param size The size of the hash table.
 * @param values A device pointer to the array where values will be stored.
 *
 * This kernel performs a parallel lookup using linear probing.
 */
__global__ void get_pair(int *keys, int numKeys, struct my_pair *hashtable, int size, 
						int *values) {

	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= numKeys)
		return;

	unsigned int pos = hash_func(keys[index], size);
	unsigned int i = 0;
	int key;

	// Invariant: The loop continues until the key is found or an empty slot is
	// encountered.
	while(true) {

		pos = (pos + i) % size;
		key = hashtable[pos].key;
		
		
		if (key == keys[index]) {
			values[index] = hashtable[pos].value;
			return;
		}
		i++;
	}
}


/**
 * @brief Constructor for the GpuHashTable class.
 * @param size The initial size of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {


	cudaError_t err;
	
	err = cudaMalloc((void **)&my_hashtable, size * sizeof(struct my_pair));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s!\n", cudaGetErrorString(err));
		return;
	}

	err = cudaMalloc((void **)&d_num_updates, sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s!\n", cudaGetErrorString(err));
		return;
	}

	

	err = cudaMemset(my_hashtable, 0, size * sizeof(struct my_pair));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s!\n", cudaGetErrorString(err));
		return;
	}

	err = cudaMemset(d_num_updates, 0, sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s!\n", cudaGetErrorString(err));
		return;
	}

	

	host_size = size;
	host_current_size = 0;

}


GpuHashTable::~GpuHashTable() {

	cudaFree(my_hashtable);
	cudaFree(d_num_updates);
}


void GpuHashTable::reshape(int numBucketsReshape) {

	cudaError_t err;
	struct my_pair *new_hashtable;

	err = cudaMalloc(&new_hashtable, numBucketsReshape * sizeof(struct my_pair));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s ( reshape)!\n", cudaGetErrorString(err));
		return;
	}
	err = cudaMemset(new_hashtable, 0, numBucketsReshape * sizeof(struct my_pair));
	if (err != cudaSuccess) {
		fprintf (stderr, "[ERROR] %s ( reshape)!\n", cudaGetErrorString(err));
		return;
	}

	
	const size_t block_size = BLOCK_SIZE;
	size_t block_num = host_size / block_size;

	if (host_size % block_size != 0)
		block_num++;

	copy_values>>(my_hashtable, new_hashtable, 
				host_size, numBucketsReshape);

	
	cudaDeviceSynchronize();

	
	
	cudaFree(my_hashtable);

	my_hashtable = new_hashtable;
	host_size = numBucketsReshape;

}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {

	cudaError_t err;	
	int new_size;
	int *keys_d;
	int *values_d;

	

	err = cudaMalloc((void **)&keys_d, numKeys * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s ( in insertBatch)!\n", cudaGetErrorString(err));
		return false;
	}
	err = cudaMalloc((void **)&values_d, numKeys * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s ( in insertBatch)!\n", cudaGetErrorString(err));
		return false;	
	}

	

	cudaMemcpy(keys_d, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(values_d, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	
	
	

	int prev_size = host_size;
	new_size = getNewSize(numKeys);

	// Pre-condition: If a resize is needed, reshape the hash table.
	if (new_size != -1)
		reshape(new_size);

	
	const size_t block_size = BLOCK_SIZE;
	size_t block_num = numKeys / block_size;

	if (numKeys % block_size != 0)
		block_num++;


	insert_pair>>(keys_d, values_d, numKeys, my_hashtable, host_size, d_num_updates);
	
	cudaDeviceSynchronize();

	

	int *host_updates = (int*) malloc(sizeof(int));
	cudaMemcpy(host_updates, d_num_updates, sizeof(int), cudaMemcpyDeviceToHost);

	host_current_size += numKeys - (*host_updates);
	

	if (prev_size != host_size) {

		float current_lf = loadFactor();
		if (current_lf < MIN_LF - ROUND_ERROR) {
			
			
			int reshape_size = host_current_size / MIN_LF;
			reshape(reshape_size);
		}
	}

	
	
	if (host_updates != 0) {
		err = cudaMemset(d_num_updates, 0, sizeof(int));
		if (err != cudaSuccess) {
			fprintf(stderr, "[ERROR] %s!\n", cudaGetErrorString(err));
			return false;
		}
	}

	
	cudaFree(keys_d);
	cudaFree(values_d);
	free(host_updates);

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	
	int *keys_d; 
	int *values_d;
	int *host_values;

	cudaError_t err;	

	
	

	err = cudaMalloc(&values_d, numKeys * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s ( in 'getBatch')!\n", cudaGetErrorString(err));
		return NULL;
	}

	err = cudaMalloc(&keys_d, numKeys * sizeof(int));
	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s ( in 'getBatch')!\n", cudaGetErrorString(err));
		return NULL;
	}

	

	cudaMemcpy(keys_d, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	const size_t block_size = BLOCK_SIZE;
	size_t block_num = numKeys / block_size;

	if (numKeys % block_size != 0)
		block_num++;

	
	get_pair>>(keys_d, numKeys, my_hashtable, host_size, values_d);

	cudaDeviceSynchronize();

	

	host_values = (int *) malloc(numKeys * sizeof(int));
	if (host_values == NULL) {
		fprintf(stderr, "[ERROR] MALLOC !\n");
		return NULL;
	}

	err = cudaMemcpy(host_values, values_d, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		fprintf(stderr, "[ERROR] %s (memcpy for 'values')!\n", cudaGetErrorString(err));
		return NULL;
	}
	cudaFree(keys_d);
	cudaFree(values_d);

	return host_values;

}


float GpuHashTable::loadFactor() {
	if (host_size == 0)
		return -1;
	return ((float)host_current_size/host_size);
}


/**
 * @brief Determines if a resize is needed and calculates the new size.
 * @param numKeys The number of new keys being inserted.
 * @return The new size of the hash table, or -1 if no resize is needed.
 */
int GpuHashTable::getNewSize(int numKeys) {

	int new_size = -1;
	int current_size = host_current_size;
	int size = host_size;
	float current_lf = (float)(current_size + numKeys) / size;

	if (current_size + numKeys > size || current_lf >= MIN_LF + MAX_RANGE)
		new_size = (current_size + numKeys) / MIN_LF;

	
	return new_size;

}


void GpuHashTable::printInfo(const char* funcName) {

	fprintf(stderr, "=======   Info Hashtable (%s)  =======\n", funcName);
	fprintf(stderr, "SIZE = %d\n", host_size);
	fprintf(stderr, "CURRENT_SIZE = %d\n", host_current_size);
	fprintf(stderr, "LOAD FACTOR = %f\n", loadFactor());
	fprintf(stderr, "======================================\n");
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

#define PRIME_A 653267llu
#define PRIME_B 4349795294267llu
#define MIN_LF 0.85
#define ROUND_ERROR 0.01
#define MAX_RANGE 0.1
#define BLOCK_SIZE 1024

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)
/**
 * @brief A list of prime numbers used for hashing.
 *
 * This list is used to provide good distribution in the hash functions.
 */
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

struct my_pair {

	int key;
	int value;
};




class GpuHashTable
{
	struct my_pair *my_hashtable;
	int *d_num_updates;
	int host_size;
	int host_current_size;

	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		
		float loadFactor();
		int getNewSize(int numKeys);
		void printInfo(const char*);
		void occupancy();
		void print(string info);
	
		~GpuHashTable();
};

#endif