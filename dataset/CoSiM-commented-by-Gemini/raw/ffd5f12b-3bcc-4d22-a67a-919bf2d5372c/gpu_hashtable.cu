
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


__global__ void kernelInsert(Entry *entries, int size, int *keys, 
	int *values, int N, int *info) {

	int hash, oldKey, i;


	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= N)
		return;
	
	hash = hashKey(keys[index], size);
	i = hash;

	
	do {
		oldKey = atomicCAS(&entries[i].key, 0, keys[index]);
		
		if (oldKey == 0) {
			atomicAdd(info, 1);
			entries[i].value = values[index];
			return;
		}
		
		if (oldKey == keys[index]) {
			entries[i].value = values[index];
			return;
		}
		i = (i + 1) % size;
	} while (i != hash);
}


__global__ void kernelGet(Entry *entries, int size, int *keys, 
	int *values, int N) {

	int hash, i;


	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= N)
		return;
	
	hash = hashKey(keys[index], size);
	i = hash;

	
	do {
		
		if (entries[i].key == keys[index]) {
			values[index] = entries[i].value;
			return;
		}
		i = (i + 1) % size;
	} while (i != hash);
}


__global__ void kernelRehash(Entry *newEntries, int newSize, 
	Entry *oldEntries, int oldSize) {

	int hash, oldKey, i;


	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= oldSize)
		return;

	if (oldEntries[index].key == 0)
		return;
	
	hash = hashKey(oldEntries[index].key, newSize);
	i = hash;

	
	do {
		oldKey = atomicCAS(&newEntries[i].key, 0, oldEntries[index].key);
		
		if (oldKey == 0) {
			newEntries[i].value = oldEntries[index].value;
			return;
		}
		i = (i + 1) % newSize;
	} while (i != hash);
}


GpuHashTable::GpuHashTable(int size) {
	cudaError_t err;

	
	this->size = size;
	this->occupied = 0;
	this->entries = NULL;

	
	err = cudaMalloc((void **) &this->entries, size * sizeof(Entry));

	if (err != cudaSuccess) {
		cout << "[INIT] Couldn't allocate memory\n";
		return;
	}

	
	cudaMemset((void *) this->entries, 0, size * sizeof(Entry));
}


GpuHashTable::~GpuHashTable() {
	
	cudaFree(this->entries);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	int totalSize;
	cudaError_t err;
	Entry *newEntries;
	size_t numBlocks = this->size / BLOCK_SIZE;

	numBucketsReshape *= SCALE_FACTOR;

	if (this->size % BLOCK_SIZE) {
		numBlocks++;
	}

	totalSize = numBucketsReshape * sizeof(Entry);
	
	
	err = cudaMalloc((void **) &newEntries, totalSize);

	if (err != cudaSuccess) {
		cout << "[RESHAPE] Couldn't allocate memory\n";
		return;
	}

	
	cudaMemset((void *) newEntries, 0, totalSize);

	
	kernelRehash>>(newEntries, numBucketsReshape,
		this->entries, this->size);
	cudaDeviceSynchronize();

	
	cudaFree(this->entries);
	this->entries = newEntries;
	this->size = numBucketsReshape;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *deviceKeys = NULL;
	int *deviceValues = NULL;
	int *info = NULL;
	int copy;
	float load;
	size_t totalSize = numKeys * sizeof(int);
	size_t numBlocks = numKeys / BLOCK_SIZE;
	cudaError_t err1, err2, err3;

	if (numKeys % BLOCK_SIZE) {
		numBlocks++;
	}

	
	err1 = cudaMalloc((void **) &deviceKeys, totalSize);
	err2 = cudaMalloc((void **) &deviceValues, totalSize);
	err3 = cudaMalloc((void **) &info, sizeof(int));

	if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
		cout << "[INSERT] Couldn't allocate memory\n";
		return false;
	}

	cudaMemset((void *) info, 0, sizeof(int));

	
	cudaMemcpy(deviceKeys, keys, totalSize, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, totalSize, cudaMemcpyHostToDevice);
	
	
	if (this->size - this->occupied < numKeys) {
		reshape(this->size + numKeys);
	}

	
	kernelInsert>>(this->entries, this->size,
		deviceKeys, deviceValues, numKeys, info);

	cudaDeviceSynchronize();
	cudaMemcpy(©, info, sizeof(int), cudaMemcpyDeviceToHost);
	this->occupied += copy;

	
	load = loadFactor();
	if (load < MIN_LOAD_FACTOR) {
		reshape(load * this->size);
	}

	cudaFree(info);
	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *deviceKeys = NULL;
	int *deviceValues = NULL;
	int *hostValues = NULL;
	size_t totalSize = numKeys * sizeof(int);
	size_t numBlocks = numKeys / BLOCK_SIZE;
	cudaError_t err1, err2;

	if (numKeys % BLOCK_SIZE) {
		numBlocks++;
	}

	
	hostValues = (int *)malloc(totalSize);
	err1 = cudaMalloc((void **) &deviceKeys, totalSize);
	err2 = cudaMalloc((void **) &deviceValues, totalSize);

	cudaMemset((void *) deviceValues, 0, totalSize);

	if (err1 != cudaSuccess || err2 != cudaSuccess || hostValues == NULL) {
		cout << "[GET] Couldn't allocate memory\n";
		return NULL;
	}

	
	cudaMemcpy(deviceKeys, keys, totalSize, cudaMemcpyHostToDevice);
	
	
	kernelGet>>(this->entries, this->size,
		deviceKeys, deviceValues, numKeys);

	cudaDeviceSynchronize();
	
	cudaMemcpy(hostValues, deviceValues, totalSize, cudaMemcpyDeviceToHost);

	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	return hostValues;
}


float GpuHashTable::loadFactor() {
	return (float)this->occupied / this->size; 
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
#define	SCALE_FACTOR	1.05
#define MIN_LOAD_FACTOR	0.8
#define BLOCK_SIZE		1024
#define PRIME_1			653267llu
#define PRIME_2			10703903591llu

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


__device__ int hashKey(int data, int limit) {
	return ((long)abs(data) * PRIME_1) % PRIME_2 % limit;
}



typedef struct entry {
	int key;
	int value;
} Entry;




class GpuHashTable
{
	Entry *entries;
	int size;
	int occupied;

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

