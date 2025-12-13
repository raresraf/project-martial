
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

__device__ int hash_function(int key, unsigned long long table_capacity) {
	return (((key * 350745513859007llu) % 3452434812973llu) % table_capacity);
}


__global__ void kernel_init_table(my_hashTable* hashtable, my_hashElem* table, int size) {
	hashtable->table = table;
	hashtable->max_items = size;

	printf("SUS DE TOT MAX : %d \n", hashtable->max_items);
	for (int i = 0; i < size; i++) {
		hashtable->table[i].elem_key = 0;
		hashtable->table[i].elem_value = 0;
	}

	hashtable->curr_nr_items = 0;
}


GpuHashTable::GpuHashTable(int size) {

	my_hashElem* table = NULL;
	cudaMalloc((void**)&table, size * sizeof(my_hashElem));
	cudaMemset(table, 0, size * sizeof(my_hashElem));

	cudaMalloc((void **)&dev_hash, sizeof(my_hashTable));
	kernel_init_table > > (dev_hash, table, size);
	cudaDeviceSynchronize();


}


GpuHashTable::~GpuHashTable() {

	cudaFree(dev_hash);
}

__global__ void kernel_rehash(my_hashTable* hash, my_hashElem* new_hash, unsigned long long new_max_items) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;


	my_hashElem item = hash->table[index];
	int key = item.elem_key;

	int hashed = hash_function(key, new_max_items);
	int i = new_max_items;

	while (i != 0) {

		int aux = atomicCAS(&new_hash[index].elem_key, 0, item.elem_key);
		if (aux == 0) {
			new_hash[index].elem_value = item.elem_value;
			return;
		}
		hashed = (hashed + i * i) % new_max_items;
	}
}


void GpuHashTable::reshape(int numBucketsReshape) {

	int new_size;
	if (OK == 1)
		new_size = numBucketsReshape * 100 / MIN_LOAD;
	else
		new_size = numBucketsReshape;

	unsigned long long curr_max_items;
	cudaMemcpy(&curr_max_items, &dev_hash->max_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	printf("fac reshape cu1 %d \n", new_size);
	
	if (new_size <= curr_max_items)
		new_size = curr_max_items;

	my_hashElem* new_table = NULL;
	cudaMalloc((void**)&new_table, new_size * sizeof(my_hashElem));
	cudaMemset(new_table, 0, new_size);
	printf("fac reshape cu2 %d \n", new_size);

	my_hashTable *hostHashtable = (my_hashTable*)calloc(1, sizeof(my_hashTable));
	cudaMemcpy(hostHashtable, dev_hash, sizeof(my_hashTable), cudaMemcpyDeviceToHost);

	int blocks_nr = curr_max_items / 256;
	if (curr_max_items % 256 != 0) {
		blocks_nr++;
	}

	kernel_rehash > > (dev_hash, new_table, new_size);
	cudaDeviceSynchronize();

	cudaFree(hostHashtable->table);
	hostHashtable->table = new_table;
	hostHashtable->max_items = numBucketsReshape;
	cudaMemcpy(dev_hash, hostHashtable, sizeof(my_hashTable), cudaMemcpyHostToDevice);

	free(hostHashtable);
}



__global__ void kernel_insert(my_hashTable* hash, int* keys, int* values, int numKeys) {

	int index = threadIdx.x + blockDim.x * blockIdx.x;

	int key = keys[index];
	int value = values[index];

	int hashed = hash_function(key, hash->max_items);
	

	my_hashElem item;
	item.elem_key = key;
	item.elem_value = value;

	int i = hash->max_items;
	

	while (i != 0) {
		
		int aux = atomicCAS(&(hash->table[hashed].elem_key), 0, item.elem_key);
		
		
		if (aux == keys[index]) {
			hash->table[hashed].elem_value = value;
			
			return;
		}
		else if (aux == 0) {
			atomicAdd(&hash->curr_nr_items, 1);
			hash->table[hashed].elem_value = value;
			
			return;
		}

		hashed = (hashed + i * i) % hash->max_items;
		i--;
	}
}

bool GpuHashTable::check_l(unsigned long long batchSize) {

	float loadFactor;
	unsigned long long max_items, curr_nr_items;

	cudaMemcpy(&max_items, &dev_hash->max_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(&curr_nr_items, &dev_hash->curr_nr_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	printf("CHECK %d \n", max_items);
	loadFactor = 1.0f * (curr_nr_items + batchSize) / max_items;

	if (loadFactor * 100 > MAX_LOAD)
		return false;

	return true;
}

bool GpuHashTable::insertBatch(int* keys, int* values, int numKeys) {

	unsigned long long max_items;
	cudaMemcpy(&max_items, &(dev_hash->max_items), sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	printf("ORIGINAL MAX ITEMS %d\n", max_items);

	if (!check_l(numKeys)) {
		OK = 1;
		reshape((max_items + numKeys));
	}
	else OK = 0;
	int* dev_keys;
	cudaMalloc((void**)&dev_keys, numKeys * sizeof(int));
	int* dev_values;
	cudaMalloc((void**)&dev_values, numKeys * sizeof(int));

	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int blocks_nr = numKeys / 256;
	if (numKeys % 256 != 0)
		blocks_nr++;



	kernel_insert > > (dev_hash, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize();

	return true;
}

__global__ void kernel_get_batch(my_hashTable* hash, int* keys, int* values, int numKeys) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	int key = keys[index];
	int hashed = hash_function(key, hash->max_items);
	my_hashElem item;

	int i = hash->max_items;
	while (i != 0) {
		item = hash->table[hashed];
		if (item.elem_key == key) {
			values[index] = item.elem_value;
			
			return;
		}

		
		hashed = (hashed + i * i) % hash->max_items;
		i--;
	}
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {

	int* dev_keys;
	cudaMalloc((void**)&dev_keys, numKeys * sizeof(int));
	cudaMemcpy(dev_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int* dev_values;
	cudaMalloc((void**)&dev_values, numKeys * sizeof(int));

	int* local_values = (int*)malloc(numKeys * sizeof(int));
	memset(local_values, 0, numKeys);

	int blocks_nr = numKeys / 256;
	if (numKeys % 256 != 0)
		blocks_nr++;



	kernel_get_batch > > (dev_hash, dev_keys, dev_values, numKeys);
	cudaDeviceSynchronize();

	cudaMemcpy(local_values, dev_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	return local_values;
}


float GpuHashTable::loadFactor() {
	float loadFactor;
	unsigned long long max_items, curr_nr_items;

	cudaMemcpy(&max_items, &dev_hash->max_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaMemcpy(&curr_nr_items, &dev_hash->curr_nr_items, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	loadFactor = 1.0f * curr_nr_items /max_items;

	return loadFactor; 
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()>>>> file: gpu_hashtable.hpp
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

typedef struct {
	int elem_value;
	int elem_key;
}my_hashElem;

typedef struct {
	my_hashElem* table;
	unsigned long long max_items;
	unsigned long long curr_nr_items;
}my_hashTable;

#define MAX_LOAD 85
#define MIN_LOAD 75




class GpuHashTable
{
public:
	GpuHashTable(int size);
	my_hashTable* dev_hash;
	int OK = 0;
	void reshape(int sizeReshape);

	bool check_l(unsigned long long batchSize);
	bool insertBatch(int* keys, int* values, int numKeys);
	int* getBatch(int* key, int numItems);

	float loadFactor();
	void occupancy();
	void print(string info);

	~GpuHashTable();
};

#endif

