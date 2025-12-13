
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

__global__ void insert_entries(int* keys, int* values, int numKeys, KeyValue* hashtable, unsigned int capacity) {
	unsigned int idx;
	unsigned int key;
	unsigned int value;
	unsigned int hashedKey;
	unsigned int oldVal;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numKeys)
		return;

	key = keys[idx];
	value = values[idx];
	hashedKey = hashFunc(key, capacity);

	while (true) {
		oldVal = atomicCAS(&hashtable[hashedKey].key, KEY_INVALID, key);

		if (oldVal == KEY_INVALID || oldVal == key) {
			hashtable[hashedKey].value = value;
			break;
		}

		++hashedKey;
		hashedKey %= capacity;
	}
}

__global__ void get_values(int* keys, int numKeys, KeyValue* hashtable, unsigned int capacity, int* deviceResult) {
	unsigned int idx;
	unsigned int key;
	unsigned int hashedKey;
	unsigned int count;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numKeys)
		return;


	key = keys[idx];
	count = capacity + 1;
	hashedKey = hashFunc(key, capacity);

	while (count) {
		if (hashtable[hashedKey].key == key) {
			deviceResult[idx] = hashtable[hashedKey].value;
			break;
		}

		count--;
		hashedKey++;
		hashedKey %= capacity;
	}
}

__global__ void copy_and_rehash(KeyValue *dst, KeyValue *src,
			unsigned int oldSize, unsigned int newSize) {
	unsigned int idx;
	unsigned int newSlot;
	unsigned int oldVal;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= oldSize || src[idx].key == KEY_INVALID)
		return;

	newSlot = hashFunc(src[idx].key, newSize);

	while (true) {


		oldVal = atomicCAS(&dst[newSlot].key, KEY_INVALID, src[idx].key);

		if (oldVal == KEY_INVALID) {
			dst[newSlot].value = src[idx].value;
			break;
		}

		++newSlot;
		newSlot %= newSize;
	}
}


GpuHashTable::GpuHashTable(int size) {
	cudaError_t err;

	capacity = size;
	occupancy = 0;

	err = cudaMalloc((void**) &hashtable, size * sizeof(KeyValue));
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMemset((void *) hashtable, KEY_INVALID, size * sizeof(KeyValue));
	DIE(err != cudaSuccess, "cudaMemset");
}


GpuHashTable::~GpuHashTable() {
	cudaError_t err;

	err = cudaFree(hashtable);
	DIE(err != cudaSuccess, "cudaFree");
}


void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t err;
	KeyValue *newTable;
	unsigned int numBlocks = (capacity / BLOCK_SIZE) + 1;

	
	err = cudaMalloc((void **) &newTable, numBucketsReshape * sizeof(KeyValue));
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMemset((void *) newTable, KEY_INVALID, numBucketsReshape * sizeof(KeyValue));
	DIE(err != cudaSuccess, "cudaMemset");

	
	copy_and_rehash>>(newTable, hashtable, capacity, numBucketsReshape);
	cudaDeviceSynchronize();

	
	err = cudaFree(hashtable);
	DIE(err != cudaSuccess, "cudaFree");

	hashtable = newTable;
	capacity = numBucketsReshape;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	cudaError_t err;
	int *deviceKeys, *deviceValues;
	size_t numBytes = numKeys * sizeof(int);
	unsigned int numBlocks = (numKeys / BLOCK_SIZE) + 1;

	
	if (((float) occupancy + (float) numKeys) / (float) capacity >= MAX_LOAD_FACTOR)
		reshape((int) (((float) occupancy + (float) numKeys) / MIN_LOAD_FACTOR));
	
	
	err = cudaMalloc((void **) &deviceKeys, numBytes);
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMalloc((void **) &deviceValues, numBytes);
	DIE(err != cudaSuccess, "cudaMalloc");

	


	err = cudaMemcpy(deviceKeys, keys, numBytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");
	err = cudaMemcpy(deviceValues, values, numBytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	
	insert_entries>>(deviceKeys, deviceValues, numKeys, hashtable, capacity);
	cudaDeviceSynchronize();

	


	occupancy += numKeys;

	
	err = cudaFree(deviceKeys);
	DIE(err != cudaSuccess, "cudaFree");
	err = cudaFree(deviceValues);
	DIE(err != cudaSuccess, "cudaFree");
	
	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	cudaError_t err;
	int *deviceResult;
	int *deviceKeys;
	unsigned int numBlocks = (numKeys / BLOCK_SIZE) + 1;
	size_t numBytes = numKeys * sizeof(int);

	
#ifndef IBM
	err = cudaMallocManaged((void **) &deviceResult, numBytes);

#else
	int* result = new int[numKeys];
	err = cudaMalloc((void **) &deviceResult, numBytes);

#endif
	DIE(err != cudaSuccess, "cudaMallocManaged");

	
	err = cudaMalloc((void **) &deviceKeys, numBytes);
	DIE(err != cudaSuccess, "cudaMalloc");
	
	
	err = cudaMemcpy(deviceKeys, keys, numBytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	
	get_values>>(deviceKeys, numKeys, hashtable, capacity, deviceResult);
	cudaDeviceSynchronize();

	err = cudaFree(deviceKeys);
	DIE(err != cudaSuccess, "cudaFree");

#ifdef IBM
	err = cudaMemcpy(result, deviceResult, numBytes, cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, "cudaMemcpy");

	err = cudaFree(deviceResult);
	DIE(err != cudaSuccess, "cudaFree");

	return result;

#else
	return deviceResult;
#endif
}


float GpuHashTable::loadFactor() {
	
	return (float) ((float) occupancy / (float) capacity);
}



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

#define	KEY_INVALID		0
#define MIN_LOAD_FACTOR	0.8f
#define MAX_LOAD_FACTOR	0.95f
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define BLOCK_SIZE 512

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)
	
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


 __device__ int hashFunc(int data, int limit) {
	
	data = ((data >> 16) ^ data) * 0x45d9f3b;
	data = ((data >> 16) ^ data) * 0x45d9f3b;
	data = (data >> 16) ^ data;

	return data % limit;
 }

struct KeyValue {
	unsigned int key;
	unsigned int value;
};




class GpuHashTable
{
	
	
	
	
	

	private:
		unsigned int capacity;
		unsigned int occupancy;
		KeyValue *hashtable;
		

	public:
		GpuHashTable(int size);

		void reshape(int sizeReshape);
		bool insertBatch(int *keys, int* values, int numKeys);

		int* getBatch(int* key, int numItems);		
		float loadFactor();

		
		void printHash();
	
		~GpuHashTable();
};

#endif

