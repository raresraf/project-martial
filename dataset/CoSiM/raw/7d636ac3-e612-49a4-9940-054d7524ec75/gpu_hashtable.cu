
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"
#define UNDEFINED 	0
#define HASH_A 		2074129llu
#define HASH_B 		1061961721llu


__global__ void kernel_insert(Pair* map_stored, int* keys, int *values,
		int *stored_pairs, int map_size, int total) 
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int hashed_key;
	int key;
	int value;
	
	if (i < total) {
		
		hashed_key = ((long)abs(keys[i]) * HASH_A) % HASH_B % map_size;
		key = keys[i];
		value = values[i];
		while (true) {
			
			int prev = atomicCAS(&map_stored[hashed_key].key, UNDEFINED, key);
			if (prev == UNDEFINED) {
				atomicAdd(&stored_pairs[0], 1);
				map_stored[hashed_key].value = value;
				return;
			} else if (prev == key) {
				map_stored[hashed_key].value = value;
				return;
			}


			hashed_key = (hashed_key + 1) % map_size;
		}
	}
}


__global__ void kernel_insert_reshape(Pair *map_stored, Pair* old_map,
		int map_size, int total)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int hashed_key;
	int key;
	int value;
	if (i < total) {
		if (old_map[i].key == UNDEFINED) {
			return;
		}
		hashed_key = ((long)abs(old_map[i].key) * HASH_A) % HASH_B % map_size;
		key = old_map[i].key;
		value = old_map[i].value;
		while (true) {
			
			int prev = atomicCAS(&map_stored[hashed_key].key, UNDEFINED, key);
			if (prev == UNDEFINED) {
				map_stored[hashed_key].value = value;
				return;
			}
			hashed_key = (hashed_key + 1) % map_size;
		}
	}
}




__global__ void kernel_lookup(int* keys, int* values, Pair* stored_map,
		int map_size, int total) 
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int hashed_key;
	unsigned int idx;
	
	if (i < total) {
		hashed_key = ((long)abs(keys[i]) * HASH_A) % HASH_B % map_size;
		idx = hashed_key;
		
		while (true) {
			if (stored_map[idx].key == keys[i]) {
				values[i] = stored_map[idx].value; 
				break;
			} else {
				idx = (idx + 1) % map_size;
			}
		}
	}
}


GpuHashTable::GpuHashTable(int size) {
	this->size = size;

	
	cudaMalloc((void **) &map, size * sizeof(Pair));
	DIE(map == NULL, "cudaMalloc");

	cudaMalloc((void **) &stored_pairs, sizeof(int));
	DIE(stored_pairs == NULL, "cudaMalloc");
	
	cudaMemset(stored_pairs, 0, sizeof(int));
	cudaMemset(map, 0, size * sizeof(Pair));
}


GpuHashTable::~GpuHashTable() {
	
	cudaFree(stored_pairs);
	cudaFree(map);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	int prev_size;
	Pair *new_map;
	Pair *prev_map;

	
	cudaMalloc((void **)&new_map, numBucketsReshape * sizeof(Pair));
	DIE(new_map == NULL, "cudaMalloc realloc new values");
	cudaMemset(new_map, 0, size * sizeof(Pair));
	
	
	prev_map = this->map;
	prev_size = this->size;
	
	this->map = new_map;

	kernel_insert_reshape>>(new_map, prev_map, numBucketsReshape, prev_size);
	cudaDeviceSynchronize();

	cudaFree(prev_map);
	this->size = numBucketsReshape;

}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys;
	int *device_values;
	int *host_stored_pairs;
	Pair *host_pairs;

	host_stored_pairs = (int *)malloc(sizeof(int));
	cudaMemcpy(host_stored_pairs, stored_pairs, sizeof(int), cudaMemcpyDeviceToHost);
	
	if ((float)(host_stored_pairs[0] + numKeys) > 0.9 * (float)this->size) {
		reshape((host_stored_pairs[0] + numKeys) * 1.2);
	}
	host_pairs = (Pair *) malloc(this->size * sizeof(Pair));
	DIE(host_pairs == NULL, "malloc pairs");
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	DIE(device_values == NULL, "cudaMalloc");
	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	DIE(device_keys == NULL, "cudaMalloc");

	
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	kernel_insert>>(this->map, device_keys, device_values, stored_pairs, this->size, numKeys);
	cudaDeviceSynchronize();

	free(host_pairs);
	cudaFree(host_stored_pairs);
	cudaFree(device_keys);
	cudaFree(device_values);
	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *values;
	int *device_keys;
	int *device_values;

	values = (int *) malloc(numKeys * sizeof(int));
	DIE(values == NULL, "malloc values get batch");
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	DIE(device_values == NULL, "cudaMalloc device values get batch");
	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	DIE(device_keys == NULL, "cudaMalloc device keys get batch");
	
	
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	kernel_lookup>>(device_keys, device_values, this->map, this->size, numKeys);
	cudaDeviceSynchronize();
	cudaMemcpy(values, device_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(device_values);
	cudaFree(device_keys);
	return values;
}


float GpuHashTable::loadFactor() {
	int *host_stored_pairs;
	
	host_stored_pairs = (int *)malloc(sizeof(int));
	cudaMemcpy(host_stored_pairs, stored_pairs, sizeof(int), cudaMemcpyDeviceToHost);
	return (float) host_stored_pairs[0] / (float) size;
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

typedef struct {
	int key;
	int value;
} Pair;




class GpuHashTable
{
	public:
		int size;
		int *stored_pairs;
		Pair *map;

		int *stored_keys;
		int *stored_values;

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

