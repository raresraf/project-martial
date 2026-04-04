/**
 * @file gpu_hashtable.cu
 * @brief A self-contained implementation of a GPU-accelerated hash table.
 * @details This file provides a complete hash table using CUDA, featuring an
 * open-addressing, linear-probing collision strategy. The implementation uses
 * robust `atomicCAS` operations for thread-safe insertions and includes an
 * efficient, fully device-side reshape kernel.
 *
 * @warning The logic for tracking the number of elements (`currentSize`) is
 * incorrect, as it increments on every insert call rather than only for new
 * keys. This leads to an overestimated load factor and premature resizing.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


__device__ __const size_t prime_hash[] =
{


	786433llu, 1610612741llu
};


/**
 * @brief Computes a hash value for a given key on the device.
 */
__device__ int getHash(int data, int limit) {
	return ((long long) abs(data) * prime_hash[0]) % prime_hash[1] % limit;
}


GpuHashTable::GpuHashTable(int size) {
	currentSize = 0;
	ht_size = size;
	DIE((cudaMalloc(&ht, size * sizeof(ht_pair)) != cudaSuccess), "cudaMalloc error
");
	cudaMemset(ht, 0, size * sizeof(ht_pair));
}


GpuHashTable::~GpuHashTable() {
	currentSize = 0;
	cudaFree(ht);
}


/**
 * @brief CUDA kernel to rehash all elements from an old table to a new one.
 */
__global__ void reshape_function(ht_pair *old_ht, ht_pair *new_ht, int old_size, int new_size) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	
	if (i < old_size) {
		int hash = -1;
		int key = old_ht[i].key;

		if (key != KEY_INVALID)
			hash = getHash(key, new_size);

		bool key_ok = false; 

		if (hash == -1)
			key_ok = true;

		
		// Block Logic: Linear probe using two ranges to handle wrap-around.
		for (int count = 0; count < 2; count++) {
			if (key_ok == true)
				break;
			int start = -1;
			int finish = -1;

			if (count == 0) {
				start = hash;
				finish = new_size;
			} else {
				start = 0;
				finish = hash;
			}

			for (int x = start; x < finish; x++) {
				if (key_ok == false) {
					// Atomic Operation: Attempt to claim an empty slot in the new table.
					if (atomicCAS(&new_ht[x].key, KEY_INVALID, key) == KEY_INVALID) {
						key_ok = true;
						atomicExch(&new_ht[x].value, old_ht[i].value);
						break;
					}
				}
			}
		} 
	}
}


void GpuHashTable::reshape(int numBucketsReshape) {
	ht_pair *new_ht;
	int old_size = ht_size;

	
	ht_size = numBucketsReshape;
	DIE((cudaMalloc(&new_ht, ht_size* sizeof(ht_pair)) != cudaSuccess), "cudaMalloc reshape error");
	cudaMemset(new_ht, 0, ht_size * sizeof(ht_pair));

	const size_t block_size = 256;
	size_t blocks_no = (old_size + block_size - 1) / block_size;
	
	// Launch the fully device-side reshape kernel.
	reshape_function>>(ht, new_ht, old_size, ht_size);
	cudaDeviceSynchronize();
	cudaFree(ht);
	ht = NULL;
	ht = new_ht;
}


/**
 * @brief CUDA kernel to insert or update a batch of key-value pairs.
 */
__global__ void insert_function(int *keys, int *values, int numKeys, ht_pair *ht, int ht_size) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < numKeys) {
		int new_key = keys[i];
		int new_hash = getHash(new_key, ht_size);

		
		// Block Logic: Linear probe using two ranges to handle wrap-around.
		for (int count = 0; count < 2; count++) {
			int start = -1;
			int finish = -1;

			if (count == 0) {
				start = new_hash;
				finish = ht_size;
			} else {
				start = 0;
				finish = new_hash;
			}

			for (int x = start; x < finish; x++) {
				
				int key0 = atomicCAS(&ht[x].key, KEY_INVALID, new_key);

				// Invariant: If slot was empty or already held our key, the write is safe.
				if (key0 == new_key || key0 == KEY_INVALID) {
					atomicExch(&ht[x].value, values[i]);
					return;
				}
			}
		}
	}
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {


	int *d_keys, *d_values;

	DIE(cudaMalloc(&d_keys, numKeys * sizeof(int)) != cudaSuccess, "cudaMalloc insert error");
	DIE(cudaMalloc(&d_values, numKeys * sizeof(int)) != cudaSuccess, "cudaMalloc insert error");

	cudaMemcpy(d_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Pre-condition: Resize if load factor exceeds threshold.
	if (float(currentSize + numKeys) / ht_size >= 0.95f)
		reshape(int((currentSize + numKeys) / 0.9f));

	const size_t block_size = 256;
	size_t blocks_no = (numKeys + block_size - 1) / block_size;

	insert_function>>(d_keys, d_values, numKeys, ht, ht_size);
	cudaDeviceSynchronize();

	// Bug: This incorrectly assumes every insertion is a new element.
	currentSize += numKeys;

	cudaFree(d_keys);
	cudaFree(d_values);

	return true;
}


/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 */
__global__ void get_function(int *d_keys, int *values, int numKeys, ht_pair *ht, int ht_size) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < numKeys) {
		int current_key = d_keys[i];
		int current_hash = getHash(current_key, ht_size);

		for (int count = 0; count < 2; count++) {
			int start = -1;
			int finish = -1;

			if (count == 0) {
				start = current_hash;
				finish = ht_size;
			} else {
				start = 0;
				finish = current_hash;
			}
			
			for (int x = start; x < finish; x++) {
				if (ht[x].key == current_key) {
					values[i] = ht[x].value;
					return;
				}
			}
		}
	}
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *d_keys;
	int *d_val;

	DIE(cudaMalloc(&d_keys, numKeys * sizeof(int)) != cudaSuccess, "cudaMalloc get error");
	// Memory Hierarchy: Use managed memory for the result values to simplify host access.
	DIE(cudaMallocManaged(&d_val, numKeys * sizeof(int)) != cudaSuccess, "cudaMallocManaged get error");
	
	cudaMemcpy(d_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	const size_t block_size = 256;
	size_t blocks_no = (numKeys + block_size - 1) / block_size;

	get_function>>(d_keys, d_val, numKeys, ht, ht_size);
	cudaDeviceSynchronize();
	cudaFree(d_keys);

	return d_val;
}


float GpuHashTable::loadFactor() {
	if (ht_size != 0)
		return (float(currentSize)/ht_size);

	return 0.f;
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




int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}


struct ht_pair {
	int key;
	int value;
};




class GpuHashTable
{
	int ht_size;
	int currentSize;
	ht_pair *ht;

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
