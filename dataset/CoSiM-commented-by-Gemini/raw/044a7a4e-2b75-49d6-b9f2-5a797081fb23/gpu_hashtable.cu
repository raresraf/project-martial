
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

__device__ int myhashKernel(int data, int limit) {
	return ((long long)abs(data) * PRIMENO) % PRIMENO2 % limit;
}

__global__ void insert_kerFunc(int *keys, int *values, int limit, hashtable myTable) {
	int newIndex;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int old, i, j;
	int low, high;

	if (index < limit) {
		newIndex = myhashKernel(keys[index], myTable.size);
		low = newIndex;
		high = myTable.size;
		for(j = 0; j < 2; j++) {
			for (i = low; i < high; i++) {
				old = atomicCAS(&myTable.list[i].key, 0, keys[index]);
				if (old == keys[index] || old == 0) {
					myTable.list[i].value = values[index];
					return;
				}
			}
			low = 0;
			high = newIndex;
		}

	}
}

__global__ void get_kerFunc(hashtable myTable, int *keys, int *values, int limit) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j;
	int newIndex;
	int low, high;

	if (index < limit) {
		newIndex = myhashKernel(keys[index], myTable.size);
		low = newIndex;
		high = myTable.size;
		for(j = 0; j < 2; j++) {
			for (i = low; i < high; i++) { 
				if (myTable.list[i].key == keys[index]) {
					values[index] = myTable.list[i].value;
					return;
				}
			}
			low = 0;
			high = newIndex;	
		}
	}
}

__global__ void reshapeKerFunc(hashtable oldTable, hashtable newTable, int size) {
	int newIndex, i, j;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int low, high;

	if (index >= newTable.size)
		return;

	newIndex = myhashKernel(oldTable.list[index].key, size);
	low = newIndex;
	high = size;
	for(j = 0; j < 2; j++) {
		for (i = low; i < high; i++) {
			if (atomicCAS(&newTable.list[i].key, 0,
				oldTable.list[index].key) == 0) {
				newTable.list[i].value = oldTable.list[index].value;
				return;
			}
		}
		low = 0;
		high = newIndex;
	}
}


GpuHashTable::GpuHashTable(int size) {
	myTable.size = size;
	myTable.slotsTaken = 0;

	DIE(cudaMalloc(&myTable.list, size * sizeof(node)) != cudaSuccess, "BOO COULDNT ALLOC MYTABLE LIST INIT\n"); 
	cudaMemset(myTable.list, 0, size * sizeof(node));
}


GpuHashTable::~GpuHashTable() {
	cudaFree(myTable.list);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	hashtable newTable;
	int noBlocks;
	int slots = myTable.slotsTaken;
	newTable.size = numBucketsReshape;
	newTable.slotsTaken = myTable.slotsTaken;

	DIE(cudaMalloc(&newTable.list, numBucketsReshape * sizeof(node)) != cudaSuccess, "BOO COULDNT ALLOC MYTABLE LIST INIT\n"); 	
	cudaMemset(newTable.list, 0, numBucketsReshape * sizeof(node));

	noBlocks = (myTable.size % BLOCK_SIZE != 0) ?
		(myTable.size / BLOCK_SIZE + 1) : (myTable.size / BLOCK_SIZE);
	reshapeKerFunc>>(myTable, newTable, numBucketsReshape);
	
	cudaDeviceSynchronize();
	cudaFree(myTable.list);
	myTable = newTable;
}


bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *device_keys, *device_values;
	int noBlocks;
	myTable.slotsTaken += numKeys;

	size_t memSize = numKeys * sizeof(int);
	DIE(cudaMalloc(&device_keys, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE KEYS\n");
	DIE(cudaMalloc(&device_values, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE VALUES\n");

	if ((float(myTable.slotsTaken) / myTable.size) >= 0.95)
		reshape(int((myTable.slotsTaken) / 0.8));

	cudaMemcpy(device_keys, keys, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, memSize, cudaMemcpyHostToDevice);

	noBlocks = (numKeys % BLOCK_SIZE != 0) ?


		(numKeys / BLOCK_SIZE + 1) : (numKeys / BLOCK_SIZE);
	insert_kerFunc>>(device_keys, device_values, numKeys, myTable);

	cudaDeviceSynchronize();

	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}


 int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_values;
    int *device_keys;
    int noBlocks;

	DIE(cudaMallocManaged(&device_values, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE VALUES\n");
	DIE(cudaMalloc(&device_keys, numKeys * sizeof(int)) != cudaSuccess, "BOO COULDNT ALLOC DEVICE KEYS\n");

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	noBlocks = (numKeys % BLOCK_SIZE != 0) ?


		(numKeys / BLOCK_SIZE + 1) : (numKeys / BLOCK_SIZE);

	get_kerFunc >> (myTable, device_keys, device_values, numKeys);
	
	cudaDeviceSynchronize();
	
	cudaFree(device_keys);
	return device_values;
}


 float GpuHashTable::loadFactor() {
	return (myTable.size > 0) ? float(myTable.slotsTaken) / myTable.size : 0;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp">>>> file: gpu_hashtable.hpp
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
#define PRIMENO 4148279llu
#define PRIMENO2 53944293929llu
#define BLOCK_SIZE 1024



int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {


	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}



struct node {
	int key;
	int value;
};
struct hashtable {
	int size;
	int slotsTaken;
	node *list;
};
class GpuHashTable
{	
	hashtable myTable;
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

