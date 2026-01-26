
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"



__device__ int hash_function(int data, int limit) {
	return ((long)abs(data) * 51437llu) % 543724411781llu % limit;
}


__global__ void insert(int *keys, int *values, int numKeys, hashmap hm,
			struct cell *device_hashmap) {
	int hash_value, last_key, stop_condition, start_condition, index;


	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= numKeys)
		return;

	
	hash_value = hash_function(keys[i], hm.size);
	stop_condition = hm.size;
	start_condition = hash_value;

	
	for (index = start_condition; index < stop_condition; index++) {
		last_key = atomicCAS(&device_hashmap[index].key, 0, keys[i]);
		if (last_key == 0 || last_key == keys[i]) {
			device_hashmap[index].value = values[i];
			if (last_key == keys[i])
				hm.numElem--;
			break;
		}
		
		if (index + 1 == hm.size) {
			index = -1;
			stop_condition = hash_value;
		}
	}
}


__global__ void get(int *keys, int *values, int numKeys, hashmap hm,
			struct cell *device_hashmap) {
	int hash_value, start_condition, stop_condition, index;


	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= numKeys)
		return;

	
	hash_value = hash_function(keys[i], hm.size);
	stop_condition = hm.size;
	start_condition = hash_value;

	
	for (index = start_condition; index < stop_condition; index++) {
		if (keys[i] == device_hashmap[index].key) {
			values[i] = device_hashmap[index].value;
			break;
		}
		
		if (index + 1 == hm.size) {
			index = -1;
			stop_condition = hash_value;
		}
	}	
}


__global__ void reshape_hashmap(int size, int new_size, struct cell *device_hashmap,
				struct cell *device_hashmap_reshaped) {
	int hash_value, start_condition, stop_condition, index, last_key;


	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= size)
		return;
	if (device_hashmap[i].key == 0)
		return;
	
	hash_value = hash_function(device_hashmap[i].key, new_size);
	stop_condition = new_size;
	start_condition = hash_value;

	
	for (index = start_condition; index < stop_condition; index++) {
		last_key = atomicCAS(&device_hashmap_reshaped[index].key, 0,
				device_hashmap[i].key);
		if (last_key == 0 || last_key == device_hashmap[i].key) {
			device_hashmap_reshaped[index].value =
						device_hashmap[i].value;
			break;
		}
		
		if (index + 1 == new_size) {
			index = -1;
			stop_condition = hash_value;
		}
	}	
}


GpuHashTable::GpuHashTable(int size) {
	hm.size = size;
	hm.numElem = 0;

	cudaMalloc(&device_hashmap, size * sizeof(struct cell));

	if (device_hashmap == 0) {
		printf("Couldn't allocate memory\n");
		return;
	}

	cudaMemset(device_hashmap, 0, size * sizeof(struct cell));
}


GpuHashTable::~GpuHashTable() {
	cudaFree(device_hashmap);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	int last_size = hm.size;
	const size_t block_size = 256;
	size_t blocks_no = 0;

	if (numBucketsReshape % block_size != 0)
		blocks_no = 1;
	blocks_no += numBucketsReshape / block_size;

	hm.size = numBucketsReshape;

	cudaMalloc(&device_hashmap_reshaped, numBucketsReshape *
			sizeof(struct cell));
	if (device_hashmap_reshaped == 0) {
		printf("Couldn't allocate memory\n");
		return;
	}

	cudaMemset(device_hashmap_reshaped, 0, numBucketsReshape *
			sizeof(struct cell));

	reshape_hashmap>>(last_size, hm.size,
				device_hashmap, device_hashmap_reshaped);
	cudaDeviceSynchronize();

	cudaFree(device_hashmap);
	device_hashmap = device_hashmap_reshaped;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys;
	int *device_values;
	const size_t block_size = 256;
	size_t blocks_no = 0;

	if (numKeys % block_size != 0)
		blocks_no = 1;
	blocks_no += numKeys / block_size;

	cudaMalloc(&device_keys, numKeys * sizeof(int));
	cudaMalloc(&device_values, numKeys * sizeof(int));

	if (device_keys == 0 || device_values == 0) {
		printf("Couldn't allocate memory\n");
		return NULL;
	}

	
	if (((hm.numElem + numKeys) * 1.0) / hm.size >= 0.9)
		reshape((hm.numElem + numKeys) / 0.8);

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	insert>>(device_keys, device_values, numKeys, hm,
						device_hashmap);
	cudaDeviceSynchronize();

	hm.numElem += numKeys;

	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_keys;
	int *result_values;
	int *result_values_host;
	const size_t block_size = 256;
	size_t blocks_no = 0;

	if (numKeys % block_size != 0)
		blocks_no = 1;
	blocks_no += numKeys / block_size;

	cudaMalloc(&device_keys, numKeys * sizeof(int));
	cudaMalloc(&result_values, numKeys * sizeof(int));

	if (device_keys == 0 || result_values == 0) {
		printf("Couldn't allocate memory\n");
		return NULL;
	}

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	get>>(device_keys, result_values, numKeys, hm,
					device_hashmap);
	cudaDeviceSynchronize();

	cudaFree(device_keys);

	result_values_host = (int *)malloc(numKeys * sizeof(int));

	if (!result_values_host) {
		printf("Couldn't allocate memory\n");
		cudaFree(result_values);
		return NULL;
	}	

	cudaMemcpy(result_values_host, result_values, numKeys * sizeof(int),
						cudaMemcpyDeviceToHost);
	cudaFree(result_values);
	return result_values_host;
}


float GpuHashTable::loadFactor() {
	return (1.f * hm.numElem) / hm.size;
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




struct hashmap
{
	int size;
	int numElem;
};




struct cell
{
	int key;
	int value;
};




class GpuHashTable
{
	struct hashmap hm;
	struct cell *device_hashmap;
	struct cell *device_hashmap_reshaped;
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

