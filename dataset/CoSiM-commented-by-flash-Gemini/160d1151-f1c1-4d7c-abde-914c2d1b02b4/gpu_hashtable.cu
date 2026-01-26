
#include 
#include 
#include 
#include 
#include 
#include 
#include "gpu_hashtable.hpp"


__global__ void init(hashmap_struct *hashmap, int *key, int *value, int size)
{
	hashmap->key = key;
	hashmap->value = value;
	hashmap->nr = 0;
	hashmap->max = size;
}

GpuHashTable::GpuHashTable(int size)
{
	int *key = NULL;
	int *value = NULL;

	cudaMalloc((void **) &key, size * sizeof(int));
	cudaMemset(key, 0, size);
	cudaMalloc((void **) &value, size * sizeof(int));
	cudaMemset(value, 0, size);

	cudaMalloc((void **) &hashmap, sizeof(hashmap_struct));

	init>>(hashmap, key, value, size);
	cudaDeviceSynchronize();

	cudaMalloc((void **) &aux_key, current_size * sizeof(int));
	cudaMalloc((void **) &aux_value, current_size * sizeof(int));
	get_value = (int *) calloc(current_size, sizeof(int));
}


GpuHashTable::~GpuHashTable()
{
	free(get_value);
	cudaFree(hashmap);
	cudaFree(aux_value);
	cudaFree(aux_key);
}


__device__ int my_hash(int data, int limit)
{
	return ((long)abs(data) * 41812097llu) % 3371518343llu % limit;
}



__global__ void reshape_cuda(hashmap_struct *hashmap, int *keys, int *values, int new_size)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i max && hashmap->key[i] > 0)
	{
		int hash = my_hash(hashmap->key[i], new_size);
		for(int j = 0; j < new_size; j++)
		{
			int aux = atomicCAS(keys + hash, 0, hashmap->key[i]);
			if(aux == 0)
			{
				atomicExch(values + hash, hashmap->value[i]);
				return;
			}
			else
				hash = (hash + 1) % new_size;
		}
	}
}

void GpuHashTable::reshape(int numBucketsReshape)
{
	float load = 0;
	hashmap_struct *load_map = (hashmap_struct *)calloc(1, sizeof(hashmap_struct));
	cudaMemcpy(load_map, hashmap, sizeof(hashmap_struct), cudaMemcpyDeviceToHost);
	load = ((float) load_map->nr + numBucketsReshape) / ((float) load_map->max);

	
	if(numBucketsReshape > current_size)
	{
		current_size = numBucketsReshape;
		cudaFree(aux_value);
		cudaFree(aux_key);
		free(get_value);
		cudaMalloc((void **) &aux_key, current_size * sizeof(int));
		cudaMalloc((void **) &aux_value, current_size * sizeof(int));
		get_value = (int *) calloc(current_size, sizeof(int));
	}

	if(load >= 0.98f)
	{
		int new_size = load_map->nr + numBucketsReshape;

		int *keys = NULL;
		int *values = NULL;
		cudaMalloc((void **) &keys, new_size * sizeof(int));
		cudaMemset(keys, 0, new_size);
		cudaMalloc((void **) &values, new_size * sizeof(int));
		cudaMemset(values, 0, new_size);

		int chunks = load_map->max / 256 + 1;
		reshape_cuda>>(hashmap, keys, values, new_size);
		cudaDeviceSynchronize();

		cudaFree(load_map->key);
		cudaFree(load_map->value);
		
		
		

		cudaFree(hashmap);
		cudaMalloc((void **) &hashmap, sizeof(hashmap_struct));
		init>>(hashmap, keys, values, new_size);

		
		free(load_map);
	}
}


__global__ void insert(hashmap_struct *hashmap, int *keys, int *values, int numKeys)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i < numKeys)
	{
		int key = keys[i];
		int value = values[i];
		if(key > 0 && value > 0)
		{
			int hash = my_hash(key, hashmap->max);

			for (int j = 0; j max; j++)
			{
				int aux = atomicCAS(&(hashmap->key[hash]), 0, key);
				if(aux == 0)
				{
					atomicAdd(&hashmap->nr, 1);
					atomicExch(&(hashmap->value[hash]), value);
					return;
				}
				else if(aux == key)
				{
					atomicExch(&(hashmap->value[hash]), value);
					return;
				}
				hash = (hash + 1) % hashmap->max;
			}
		}
	}
}

bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys)
{
	reshape(numKeys);
	cudaMemcpy(aux_key, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(aux_value, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	int chunks = numKeys / 256 + 1;



	insert>>(hashmap, aux_key, aux_value, numKeys);
	cudaDeviceSynchronize();

	return true;
}


__global__ void get(hashmap_struct *hashmap, int *keys, int *values, int numKeys)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i < numKeys)
	{
		int key = keys[i];
		int hash = my_hash(key, hashmap->max);

		for(int j = 0; j max; j++)
		{
			if(hashmap->key[hash] == key)
			{
				values[i] = hashmap->value[hash];
				return;
			}
			hash = (hash + 1) % hashmap->max;
		}
	}
}

int* GpuHashTable::getBatch(int* keys, int numKeys)
{
	int chunks = numKeys / 256 + 1;

	cudaMemcpy(aux_key, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);


	get>>(hashmap, aux_key, aux_value, numKeys);
	cudaDeviceSynchronize();

	cudaMemcpy(get_value, aux_value, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	return get_value;
}









float GpuHashTable::loadFactor() {
	
	
	
	float load_factor = 0;
	hashmap_struct *load_map = (hashmap_struct *)calloc(1, sizeof(hashmap_struct));
	cudaMemcpy(load_map, hashmap, sizeof(hashmap_struct), cudaMemcpyDeviceToHost);
	load_factor = ((float) load_map->nr) / ((float) load_map->max);
	free(load_map);
	
	return load_factor;
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

typedef struct
{
	int *key;
	int *value;
	int nr;
	int max;
}hashmap_struct;

#define i_size 1



int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}




class GpuHashTable
{
	hashmap_struct *hashmap = NULL;
	int *aux_key = NULL;
	int *aux_value = NULL;
	int *get_value = NULL;
	int current_size = 1;
	

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

