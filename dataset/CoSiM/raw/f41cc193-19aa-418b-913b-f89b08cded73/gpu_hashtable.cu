
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


GpuHashTable::GpuHashTable(int size) {
	cudaMalloc((void **)&map, size*sizeof(GpuHashNode));
	limit = size;
	curr_size = 0;
	cudaMalloc((void **)&dev_size, sizeof(int));
}


GpuHashTable::~GpuHashTable() {
	cudaFree(map);
}

__device__ int hash1(int data, int limit) {
	return ((long)abs(data) * 1061961721llu) % 14480561146010017169llu;
}


__global__ void kernel_reshape_map(GpuHashNode *hmap, GpuHashNode *new_hmap, int new_lim, int lim) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= lim) {
		return;
	}

	GpuHashNode node = hmap[idx];
	if(node.val == 0 || node.key == 0) {
		return;
	}

	int key = node.key;
	int val = node.val;
	
	unsigned int hash_code = hash1(key, new_lim);
	unsigned int tmp_code = hash_code;
	hash_code %= new_lim;
	for(int i = 1; i <= new_lim; i++) {

		if(atomicCAS(&(new_hmap[hash_code].key), 0, key) == 0) {
			atomicExch(&(new_hmap[hash_code].val), val);
			return;
		}

		if(new_hmap[hash_code].key == key && new_hmap[hash_code].val == val) {
			return;
		}

		hash_code = (tmp_code + i%new_lim) % new_lim;
	}
}


void GpuHashTable::reshape(int numBucketsReshape) {
	if(limit < numBucketsReshape) {
		GpuHashNode *new_map;
		cudaMalloc((void **)&new_map, numBucketsReshape*sizeof(GpuHashNode));

		int numBlocks = limit / 512 + 1;
		kernel_reshape_map>>(map, new_map, numBucketsReshape, limit);
		cudaDeviceSynchronize();

		GpuHashNode *tmp_map = map;
		map = new_map;
		limit = numBucketsReshape;


		cudaFree(tmp_map);
	}

}

__global__ void kernel_insertBatch(GpuHashNode *hmap, int *keys, int *values, int limit, int numKeys, int *size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= numKeys) {
		return;
	}

	int key = keys[idx];
	int val = values[idx];

	if(key == 0 || val == 0) {
		return;
	}
	
	unsigned int hash_code = hash1(key, limit);
	unsigned int tmp_code = hash_code;
	hash_code %= limit;
	for(int i = 1; i <= limit; i++) {
		if(hmap[hash_code].key == key && hmap[hash_code].val == val) {
			return;
		}

		if(atomicCAS(&(hmap[hash_code].key), 0, key) == 0) {
				atomicExch(&(hmap[hash_code].val), val);
			atomicAdd((unsigned int *)size, 1);
			return;
		}

		
		if(hmap[hash_code].key == key) {
			atomicExch(&(hmap[hash_code].val), val);
			return;
		}

		hash_code = (tmp_code + i%limit) % limit;
	}

}

bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	
	
	
	GpuHashTable::reshape((((double)curr_size + (double)numKeys))/0.85);

	int numBlocks = numKeys / 512 + 1;
	int *dev_keys, *dev_values;
	cudaMalloc((void **)&dev_keys, numKeys*sizeof(int));
	cudaMalloc((void **)&dev_values, numKeys*sizeof(int));
	cudaMemcpy(dev_keys, keys, numKeys*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_values, values, numKeys*sizeof(int), cudaMemcpyHostToDevice);

	kernel_insertBatch>>(map, dev_keys, dev_values, limit, numKeys, dev_size);
	cudaDeviceSynchronize();
	
	cudaMemcpy(&curr_size, dev_size, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_keys);
	cudaFree(dev_values);
	return true;
}



__global__ void kernel_getBatch(GpuHashNode *hmap, int *keys, int *values, int lim, int numKeys) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= numKeys) {
		return;
	}
	int key = keys[idx];

	unsigned int hash_code = hash1(key, lim);
	unsigned int tmp_code = hash_code;
	hash_code %= lim;
	
	for(int i = 1; i <= lim; i++) {
		
		if(hmap[hash_code].key == key) {
			values[idx] = hmap[hash_code].val;
			return;
		}
		hash_code = (tmp_code + i%lim) % lim;
	}
}

int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *result = (int*)malloc(numKeys * sizeof(int));
	int *dev_keys, *dev_values;
	cudaMalloc((void **)&dev_keys, numKeys*sizeof(int));
	cudaMalloc((void **)&dev_values, numKeys*sizeof(int));
	cudaMemcpy(dev_keys, keys, numKeys*sizeof(int), cudaMemcpyHostToDevice);

	int numBlocks = numKeys / 512 + 1;
	kernel_getBatch>>(map, dev_keys, dev_values, limit, numKeys);
	cudaDeviceSynchronize();

	cudaMemcpy(result, dev_values, numKeys*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_values);
	cudaFree(dev_keys);
	return result;
}


float GpuHashTable::loadFactor() {
	
	return (curr_size*1.0f)/limit; 
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










int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}




typedef struct GpuHashNode {
	int val = 0;
	int key = 0;
}GpuHashNode;

class GpuHashTable
{

	public:
		GpuHashNode *map;
		int limit;
		int curr_size;
		int *dev_size;
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		void reshape_and_rehash(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		
		float loadFactor();
		void occupancy();
		void print(string info);
	
		~GpuHashTable();
};

#endif

