

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define FIRST_PRIME 13169977
#define SECOND_PRIME 5351951779


__device__ int device_hash(int data, int limit)
{
	return ((long long)abs(data) * FIRST_PRIME) % SECOND_PRIME % limit;
}

__global__ void kernel_insert(int *keys, int *values, int numKeys,
			      entry *hash_table, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, hash_value, old_value;

	if (idx >= numKeys)
		return;

	
	hash_value = device_hash(keys[idx], size);

	
	for (i = hash_value; i < size; i++) {
		
		old_value =
			atomicCAS(&hash_table[i].key, KEY_INVALID, keys[idx]);
		
		if (old_value == keys[idx] || old_value == KEY_INVALID) {
			hash_table[i].value = values[idx];
			return;
		}
	}
	
	for (i = 0; i < hash_value; i++) {
		old_value =
			atomicCAS(&hash_table[i].key, KEY_INVALID, keys[idx]);
		
		if (old_value == keys[idx] || old_value == KEY_INVALID) {
			hash_table[i].value = values[idx];
			return;
		}
	}
}

__global__ void kernel_get(int *keys, int *values, int numKeys,
			   entry *hash_table, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, hash_value;

	if (idx >= numKeys)
		return;

	
	hash_value = device_hash(keys[idx], size);

	
	for (i = hash_value; i < size; i++) {
		if (hash_table[i].key == keys[idx]) {
			values[idx] = hash_table[i].value;
			return;
		}
	}

	
	for (i = 0; i < hash_value; i++) {
		if (hash_table[i].key == keys[idx]) {
			values[idx] = hash_table[i].value;
			return;
		}
	}
}

__global__ void kernel_reshape(entry *hash_table, entry *new_hash_table,
			       int old_size, int new_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, new_hash, old_value;

	if (idx >= old_size || hash_table[idx].key == KEY_INVALID)
		return;

	
	new_hash = device_hash(hash_table[idx].key, new_size);

	
	for (i = new_hash; i < new_size; i++) {
		


		old_value = atomicCAS(&new_hash_table[i].key, KEY_INVALID,
				      hash_table[idx].key);
		
		if (old_value == KEY_INVALID) {
			new_hash_table[i].value = hash_table[idx].value;
			return;
		}
	}
	
	for (i = 0; i < new_hash; i++) {
		old_value = atomicCAS(&new_hash_table[i].key, KEY_INVALID,
				      hash_table[idx].key);
		
		if (old_value == KEY_INVALID) {
			new_hash_table[i].value = hash_table[idx].value;
			return;
		}
	}
}


GpuHashTable::GpuHashTable(int size)
{
	cudaError_t err;

	
	this->numberInsertedPairs = 0;
	this->size = size;

	err = cudaMalloc(&this->hash_table, size * sizeof(entry));
	DIE(err != cudaSuccess, "cudaMalloc init");

	err = cudaMemset(this->hash_table, 0, size * sizeof(entry));
	DIE(err != cudaSuccess, "cudaMemset init");
}


GpuHashTable::~GpuHashTable()
{
	cudaFree(this->hash_table);
}


void GpuHashTable::reshape(int numBucketsReshape)
{
	entry *new_hash_table;
	cudaError_t err;
	unsigned int numBlocks;
	int old_size = this->size;

	
	this->size = numBucketsReshape;

	err = cudaMalloc(&new_hash_table, numBucketsReshape * sizeof(entry));
	DIE(err != cudaSuccess, "cudaMalloc reshape");

	err = cudaMemset(new_hash_table, 0, numBucketsReshape * sizeof(entry));
	DIE(err != cudaSuccess, "cudaMemset reshape");

	
	numBlocks = numBucketsReshape / THREADS_PER_BLOCK;
	if (numBucketsReshape % THREADS_PER_BLOCK)
		numBlocks++;

	kernel_reshape >>(
		this->hash_table, new_hash_table, old_size, numBucketsReshape);

	
	cudaDeviceSynchronize();

	
	err = cudaFree(this->hash_table);
	DIE(err != cudaSuccess, "cudaFree reshape");

	this->hash_table = new_hash_table;
}


bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys)
{
	int *device_keys;
	int *device_values;
	size_t total_size = numKeys * sizeof(int);
	unsigned int numBlocks;
	cudaError_t err;

	err = cudaMalloc(&device_keys, total_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc device_keys");
		return false;
	}
	err = cudaMalloc(&device_values, total_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc device_values");
		return false;
	}

	
	if (float(numberInsertedPairs + numKeys) / this->size >=
	    MAXIMUM_LOAD_FACTOR)
		reshape(int((numberInsertedPairs + numKeys) /
			    MINIMUM_LOAD_FACTOR));



	err = cudaMemcpy(device_keys, keys, total_size, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy device_keys");
	err = cudaMemcpy(device_values, values, total_size,
			 cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy device_values");

	
	numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK)
		numBlocks++;

	kernel_insert >>(
		device_keys, device_values, numKeys, this->hash_table,
		this->size);

	
	cudaDeviceSynchronize();

	


	numberInsertedPairs += numKeys;

	err = cudaFree(device_keys);
	DIE(err != cudaSuccess, "cudaFree device_keys");

	err = cudaFree(device_values);
	DIE(err != cudaSuccess, "cudaFree device_values");

	return true;
}


int *GpuHashTable::getBatch(int *keys, int numKeys)
{
	int *device_keys;
	int *values, *device_values;
	size_t total_size = numKeys * sizeof(int);
	unsigned int numBlocks;
	cudaError_t err;

	err = cudaMalloc(&device_keys, total_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc device_keys");
		return nullptr;
	}
	err = cudaMalloc(&device_values, total_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaMalloc values");
		return nullptr;
	}

	err = cudaMemset(device_values, 0, total_size);
	DIE(err != cudaSuccess, "cudaMemset getbatch");

	
	err = cudaMemcpy(device_keys, keys, total_size, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy device_keys");

	values = (int *)malloc(total_size);
	DIE(values == NULL, "malloc values");

	
	numBlocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK)
		numBlocks++;

	kernel_get >> (device_keys,
						       device_values, numKeys,
						       this->hash_table,
						       this->size);

	
	cudaDeviceSynchronize();

	
	err = cudaMemcpy(values, device_values, total_size,
			 cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, "cudaMemcpy values");

	err = cudaFree(device_keys);
	DIE(err != cudaSuccess, "cudaFree device_keys");

	err = cudaFree(device_values);
	DIE(err != cudaSuccess, "cudaFree device_values");

	return values;
}


float GpuHashTable::loadFactor()
{
	
	if (this->size == 0)
		return 0;
	else
		return (float(numberInsertedPairs) / this->size);
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys)                               \
	GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"

#ifndef _HASHCPU_
#define _HASHCPU_

#include 
#include 

using namespace std;


#define	KEY_INVALID			0
#define THREADS_PER_BLOCK 	1024
#define MINIMUM_LOAD_FACTOR	0.8
#define MAXIMUM_LOAD_FACTOR	0.9

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





typedef struct entry {
	int key;
	int value;
} entry;

class GpuHashTable
{
	public:
		
		int numberInsertedPairs;
		int size;
		entry *hash_table;

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
