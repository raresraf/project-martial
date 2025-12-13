
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define minim 0.82
#define maxim 0.93
#define num_threads 256


__global__ void do_reshape(hashtable oldHash, hashtable newHash) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < oldHash.size &&
		oldHash.table[index].key != KEY_INVALID) {
		int key = oldHash.table[index].key;
		int hash = Hash(key, newHash.size);
		int size = newHash.size, i;
		
		
		for (i = hash; i < size; i++){
			if (atomicCAS(&newHash.table[i].key, KEY_INVALID, key) == KEY_INVALID) {
				newHash.table[i].value = oldHash.table[index].value;
				return;
			}
		}
		for (i = 0; i < hash; i++){
			if (atomicCAS(&newHash.table[i].key, KEY_INVALID, key) == KEY_INVALID) {
				newHash.table[i].value = oldHash.table[index].value;


				return;
			}
		}
	}
}


__global__ void insert(int *keys, int *values, hashtable hashmap, int count) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < count) {
		int key = keys[index];
		int hash = Hash(key, hashmap.size);
		int size = hashmap.size, i;
		
		
		for(i = hash; i < size; i++){
			atomicCAS(&hashmap.table[i].key, KEY_INVALID, key);
			if (hashmap.table[i].key == key) {
				hashmap.table[i].value = values[index];
				return;
			}
		}
		for(i = 0; i < hash; i++){
			atomicCAS(&hashmap.table[i].key, KEY_INVALID, key);
			if (hashmap.table[i].key == key) {
				hashmap.table[i].value = values[index];


				return;
			}
		}
	}
	
}


__global__ void get(int *keys, int *values, hashtable hashmap, int count) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < count) {
		int key = keys[index];
		int hash = Hash(keys[index], hashmap.size);
		int size = hashmap.size, i;
		
		for (i = hash; i < size; i++){
			if (hashmap.table[i].key == key) {
				values[index] = hashmap.table[i].value;
				return;
			}
		}
		for (i = 0; i < hash; i++){
			if (hashmap.table[i].key == key) {
				values[index] = hashmap.table[i].value;
				return;
			}
		}
	}
}


GpuHashTable::GpuHashTable(int size) {
	h.real_size = 0;
	h.size = size;
	cudaMalloc(&h.table, size * sizeof(assoc));
	cudaMemset(h.table, 0, size * sizeof(assoc));
}


GpuHashTable::~GpuHashTable() {
	cudaFree(h.table);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	hashtable newh;
	newh.size = numBucketsReshape;
	newh.real_size = h.real_size;
	cudaMalloc(&newh.table, numBucketsReshape * sizeof(assoc));
	cudaMemset(newh.table, 0, numBucketsReshape * sizeof(assoc));
	if (h.size / num_threads * num_threads == h.size) {
		do_reshape>>(h, newh);
	} else {


		do_reshape>>(h, newh);
	}
	cudaDeviceSynchronize();
	cudaFree(h.table);
	h = newh;
}


bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int *deviceKeys, *deviceValues;
	cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	cudaMalloc(&deviceValues, numKeys * sizeof(int));
	h.real_size += numKeys;

	
	if (float(h.real_size) >= (float)(h.size * (float)maxim))
		reshape(int((h.real_size) / minim));

	
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	if (numKeys / num_threads * num_threads == numKeys) {
		insert>>
			(deviceKeys, deviceValues, h, numKeys);
	} else {
		insert>>
			(deviceKeys, deviceValues, h, numKeys);
	}
	
	cudaDeviceSynchronize();
	cudaFree(deviceKeys);
	cudaFree(deviceValues);
	return false;
}


int *GpuHashTable::getBatch(int *keys, int numKeys) {
	int *deviceKeys, *values;
	cudaMallocManaged(&deviceKeys, numKeys * sizeof(int));
	cudaMallocManaged(&values, numKeys * sizeof(int));

	
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	if (numKeys / num_threads * num_threads == numKeys) {
		get>>
			(deviceKeys, values, h, numKeys);
	} else {
		get>>
			(deviceKeys, values, h, numKeys);
	}
	cudaDeviceSynchronize();
	cudaFree(deviceKeys);
	return values;
}


float GpuHashTable::loadFactor() {
	if (h.size > 0)
		return (float(h.real_size) / h.size);
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

#include 
#include 

#define KEY_INVALID 0

#define A 518509
#define B 6904869625999

#define DIE(assertion, call_description) \
    do {    \
        if (assertion) {    \
        fprintf(stderr, "(%s, %d): ",    \
        __FILE__, __LINE__);    \
        perror(call_description);    \
        exit(errno);    \
    }    \
} while (0)


__device__ int Hash(int data, int limit) {
	return ((long long) abs(data) * A) % B % limit;
}

const std::size_t primeList[] =
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
	return ((long) abs(data) * primeList[64]) % primeList[90] % limit;
}

int hash2(int data, int limit) {
	return ((long) abs(data) * primeList[67]) % primeList[91] % limit;
}

int hash3(int data, int limit) {
	return ((long) abs(data) * primeList[70]) % primeList[93] % limit;
}


struct assoc {
	int key;
	int value;
};


struct hashtable {
	int size;
	int real_size;
	assoc *table;
};

class GpuHashTable {
	hashtable h;
public:
	GpuHashTable(int size);

	void reshape(int sizeReshape);

	bool insertBatch(int *keys, int *values, int numKeys);

	int *getBatch(int *key, int numItems);

	float loadFactor();

	void occupancy();

	void print(std::string info);

	~GpuHashTable();
};

#endif
