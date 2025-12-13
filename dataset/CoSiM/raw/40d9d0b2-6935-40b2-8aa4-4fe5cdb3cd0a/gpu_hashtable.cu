
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


#define HASH_VAL1 13
#define HASH_VAL2 56564691976601587

__global__ void kernel_reshape(elem *dest, elem *source, unsigned int source_size, unsigned int dest_size);
__global__ void insertFunc(int *keys, int *values, int numKeys, int dictSize, elem* dictElements);
__global__ void getFunc(int *keys, int *values, int numKeys, int dictSize, elem* dictElements);
__device__ int myHash(int data, int limit);


GpuHashTable::GpuHashTable(int size) {
	DIE(size < 0, "invalid size");

	cudaError_t rc;

	this->inserted = 0;
	this->size = size;

	rc = cudaMalloc(&this->elements, size * sizeof(*elements));
	DIE(rc != cudaSuccess || this->elements == NULL, "cudaMalloc failed");

	rc = cudaMemset(this->elements, 0, size * sizeof(*elements));
	DIE(rc != cudaSuccess, "cudaMemset failed");
}


GpuHashTable::~GpuHashTable() {
	cudaFree(this->elements);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t rc;
	elem *aux;

	
	

	rc = cudaMalloc(&aux, numBucketsReshape * sizeof(elem));
	DIE(rc != cudaSuccess || aux == NULL, "cudaMalloc failed");

	rc = cudaMemset(aux, 0, numBucketsReshape * sizeof(*aux));
	DIE(rc != cudaSuccess, "cudaMemset failed");

	size_t num_blocks;

	num_blocks = this->size / block_size;
	if (this->size % block_size)
		num_blocks++;

	elem *old_elements = this->elements;
	
	
	
	
	

	kernel_reshape>>(aux, this->elements,
		this->size, numBucketsReshape);
	cudaDeviceSynchronize();

	
	rc = cudaFree(old_elements);
	DIE(rc != cudaSuccess, "free failed");

	
	
	this->elements = aux;
	this->size = numBucketsReshape;

	
	
	
	
	
	
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *deviceKeys;
	int *deviceValues;
	size_t num_blocks;
	cudaError_t rc;

	
	rc = cudaMalloc(&deviceKeys, numKeys * sizeof(*deviceKeys));
	DIE(rc != cudaSuccess, "cudaMalloc failed");
	if (deviceKeys == NULL)
		return false;

	rc = cudaMalloc(&deviceValues, numKeys * sizeof(*deviceValues));
	DIE(rc != cudaSuccess, "cudaMalloc failed");
	if (deviceValues == NULL)
		return false;

	rc = cudaMemcpy(deviceKeys, keys, numKeys * sizeof(*deviceKeys), cudaMemcpyHostToDevice);
	DIE(rc != cudaSuccess, "cudaCopy failed");

	rc = cudaMemcpy(deviceValues, values, numKeys * sizeof(*deviceValues), cudaMemcpyHostToDevice);
	DIE(rc != cudaSuccess, "cudaCopy failed");

	
	

	
	
	
	

	num_blocks = numKeys / block_size;
	if (numKeys % block_size)
		num_blocks++;

	

	
	this->inserted += numKeys;
	if (loadFactor() > 0.8) {
		reshape(this->inserted / 0.8);
	}

	
	
	int dictSize = this->size;

	
	insertFunc>>(deviceKeys, deviceValues, numKeys, dictSize, this->elements);
	cudaDeviceSynchronize();

	

	
	

	
	
	
	



	rc = cudaFree(deviceKeys);
	DIE(rc != cudaSuccess, "free failed");
	
	rc = cudaFree(deviceValues);
	DIE(rc != cudaSuccess, "free failed");

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *deviceKeys;
	cudaError_t rc;
	int *deviceValues;
	int *hostValues;
	size_t num_blocks;

	
	
	
	

	rc = cudaMalloc(&deviceKeys, numKeys * sizeof(*deviceKeys));
	DIE(rc != cudaSuccess, "malloc failed");
	if (deviceKeys == NULL)
		return NULL;

	rc = cudaMalloc(&deviceValues, numKeys * sizeof(*deviceValues));
	DIE(rc != cudaSuccess, "malloc failed");
	if (deviceValues == NULL)
		return NULL;

	hostValues = (int *)malloc(numKeys * sizeof(*hostValues));
	if (hostValues == NULL)
		return NULL;

	rc = cudaMemset(deviceValues, 0, numKeys * sizeof(*deviceValues));
	DIE(rc != cudaSuccess, "cudaMemset failed");

	rc = cudaMemcpy(deviceKeys, keys, numKeys * sizeof(*deviceKeys), cudaMemcpyHostToDevice);
	DIE(rc != cudaSuccess, "copy failed");

	num_blocks = numKeys / block_size;
	if (numKeys % block_size)
		num_blocks++;

	int dictSize = this->size;

	getFunc>>(deviceKeys, deviceValues, numKeys, dictSize, this->elements);
	cudaDeviceSynchronize();



	rc = cudaMemcpy(hostValues, deviceValues, numKeys * sizeof(*hostValues), cudaMemcpyDeviceToHost);
	DIE(rc != cudaSuccess, "cudaCopy failed");

	rc = cudaFree(deviceKeys);
	DIE(rc != cudaSuccess, "free failed");

	rc = cudaFree(deviceValues);
	DIE(rc != cudaSuccess, "free failed");

	return hostValues;

}


float GpuHashTable::loadFactor() {
	
	if (size == 0)
		return 0;
	return (float)inserted / size;
}

__global__ void getFunc(int *keys, int *values, int numKeys, int dictSize, elem *dictElems) {
	unsigned int id;
	int hash_pos;

	
	

	id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numKeys)
		return;

	hash_pos = myHash(keys[id], dictSize);

	for (int i = hash_pos; i < dictSize; ++i) {
		if (dictElems[i].key == keys[id]) {
			values[id] = dictElems[i].value;
			return;
		}
	}
	for (int i = 0; i < hash_pos; ++i) {
		if (dictElems[i].key == keys[id]) {
			values[id] = dictElems[i].value;
			return;
		}
	}
	
}

__global__ void insertFunc(int *keys, int *values, int numKeys, int dictSize, elem* dictElems) {
	unsigned int id;
	unsigned int old_key;
	int hash_pos;

	
	

	id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numKeys)
		return;
	
	hash_pos = myHash(keys[id], dictSize);

	for (int i = hash_pos; i < dictSize; ++i) {


		old_key = atomicCAS(&dictElems[i].key, 0, keys[id]);
		
		if (old_key == 0 || old_key == keys[id]) {
			dictElems[i].value = values[id];
			return;
		}
	}

	for (int i = 0; i < hash_pos; ++i) {
		old_key = atomicCAS(&dictElems[i].key, 0, keys[id]);
		if (old_key == 0 || old_key == keys[id]) {
			dictElems[i].value = values[id];
			return;
		}
	}
}

__global__ void kernel_reshape(elem *dest, elem *source, unsigned int source_size, unsigned int dest_size){
	int hash_pos;
	unsigned int id;

	id = threadIdx.x + blockDim.x * blockIdx.x;

	
	
	

	if (id >= source_size)
		return;
	if (source[id].key == 0)
		return;

	

	hash_pos = myHash(source[id].key, dest_size);

	
	
	

	
	
	

	for (int i = hash_pos; i < dest_size; ++i) {
		

		if (atomicCAS(&dest[i].key, 0, source[id].key) == 0) {
			dest[i].value = source[id].value;
			return;
		}
	}

	for (int i = 0; i < hash_pos; ++i) {
		if (atomicCAS(&dest[i].key, 0, source[id].key) == 0) {
			dest[i].value = source[id].value;
			return;
		}
	}
}

__device__ int myHash(int data, int limit) {
	
	return ((long)abs(data) * HASH_VAL1) % HASH_VAL2 % limit;
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

const size_t block_size = 256;

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





typedef struct {
	unsigned int key;
	unsigned int value;
} elem;

class GpuHashTable
{

	public:
		unsigned int inserted;
		unsigned int size;
		elem *elements;

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

