

__device__ int hash_func(int key, int *HMAX) {
	return ((long)abs(key) * 26339969) % 6743036717 % *HMAX;
}


__global__ void insert(int *hash_keys, int *hash_values, int *empty_slots,
                        int *gpu_keys, int *gpu_values, int *size, int *HMAX)
{
	
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

	
	int key = gpu_keys[i];
	int value = gpu_values[i];

	
	if (key <= 0 || value <= 0) {
		return;
	}
						
	
	int hash_key = hash_func(key, HMAX);
	int index = hash_key;

	
	int elem = atomicExch(&empty_slots[index], 1);
	
	if (elem == 0) {
		hash_keys[index] = key;
		hash_values[index] = value;
		atomicAdd(size, 1);
		return;
	}
	atomicExch(&empty_slots[index], elem);
	
	if (hash_keys[index] == key) {
		atomicExch(&hash_values[index], value);
		return;
	}
	
	index = (index + 1) % *HMAX;
	while (index != hash_key) {
		elem = atomicExch(&empty_slots[index], 1);
		
		if (elem == 0) {
			hash_keys[index] = key;
			hash_values[index] = value;
			atomicAdd(size, 1);
			return;
		}
		atomicExch(&empty_slots[index], elem);
		
		if (hash_keys[index] == key) {
			atomicExch(&hash_values[index], value);
			return;
		}
		index = (index + 1) % *HMAX;
	}
	
}


__global__ void get_val(int *hash_keys, int *hash_values, int *empty_slots,
                        int *keys, int* values, int *HMAX)
{
	
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int key = keys[i]; 
						
	
	int hash_key = hash_func(key, HMAX);
	int index = hash_key;

	
	while (empty_slots[index] == 1) {

		
		if (hash_keys[index] == key) {
			values[i] = hash_values[index];
			return;
		}
		index = (index + 1) % *HMAX;
		
		if (index == hash_key) {
			break;
		}
	}
}>>>> file: gpu_hashtable.cu
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"
#include "gpu_functions.cu"


GpuHashTable::GpuHashTable(int size) {
	
	this->HMAX = size;
	this->size = 0;

	
	this->hash_keys = 0;
	this->hash_values = 0;
	this->empty_slots = 0;
	cudaMalloc((void **) &this->hash_keys, sizeof(int) * this->HMAX);
	cudaMalloc((void **) &this->hash_values, sizeof(int) * this->HMAX);
	cudaMalloc((void **) &this->empty_slots, sizeof(int) * this->HMAX);
	DIE(this->hash_keys == 0 || this->hash_values == 0 ||
		this->empty_slots == 0, "[GpuHashTable] cudaMalloc error");
	cudaMemset(this->hash_keys, 0, sizeof(int) * this->HMAX);
	cudaMemset(this->hash_values, 0, sizeof(int) * this->HMAX);
	cudaMemset(this->empty_slots, 0, sizeof(int) * this->HMAX);
}


GpuHashTable::~GpuHashTable() {
	if (this->hash_keys != 0 || this->hash_keys != NULL) {
		cudaFree(this->hash_keys);
	}
	if (this->hash_values != 0 || this->hash_values != NULL) {
		cudaFree(this->hash_values);
	}
	if (this->empty_slots != 0 || this->empty_slots != NULL) {
		cudaFree(this->empty_slots);
	}

}


 void GpuHashTable::reshape(int numBucketsReshape) {
	int *gpu_keys = 0;
	int *gpu_values = 0;

	
	cudaMalloc(&gpu_keys, sizeof(int) * this->HMAX);
	cudaMalloc(&gpu_values, sizeof(int) * this->HMAX);
	DIE(gpu_keys == 0 || gpu_values == 0, "[RESHAPE] cudaMalloc error");

	
	cudaMemcpy(gpu_keys, this->hash_keys, sizeof(int) * this->HMAX, cudaMemcpyHostToHost);
	cudaFree(this->hash_keys);
	cudaMemcpy(gpu_values, this->hash_values, sizeof(int) * this->HMAX, cudaMemcpyHostToHost);
	cudaFree(this->hash_values);
	cudaFree(this->empty_slots);
	
	
	this->hash_keys = 0;
	this->hash_values = 0;
	this->empty_slots = 0;
	cudaMalloc((void **) &this->hash_keys, sizeof(int) * numBucketsReshape);
	cudaMalloc((void **) &this->hash_values, sizeof(int) * numBucketsReshape);
	cudaMalloc((void **) &this->empty_slots, sizeof(int) * numBucketsReshape);
	DIE(this->hash_keys == 0 || this->hash_values == 0 ||
		this->empty_slots == 0, "[RESHAPE] cudaMalloc error");
	cudaMemset(this->empty_slots, 0, sizeof(int) * numBucketsReshape);
	cudaMemset(this->hash_keys, 0, sizeof(int) * numBucketsReshape);
	cudaMemset(this->hash_values, 0, sizeof(int) * numBucketsReshape);
	
	if (this->size > 0) {
		int numKeys = this->HMAX;
		int *gpu_HMAX = 0;
		int *gpu_size = 0;
		
		cudaMalloc((void **) &gpu_size, sizeof(int));
		cudaMalloc((void **) &gpu_HMAX, sizeof(int));
		DIE(gpu_size == 0 || gpu_HMAX == 0, "[RESHAPE] malloc error");
		cudaMemset(&gpu_size, 0, sizeof(int));

		
		cudaMemcpy(gpu_HMAX, &numBucketsReshape, sizeof(int), cudaMemcpyHostToDevice);

		
		insert>>(this->hash_keys, this->hash_values,
				this->empty_slots, gpu_keys, gpu_values, gpu_size, gpu_HMAX);
		
		cudaDeviceSynchronize(); 

		
		cudaFree(gpu_keys);
		cudaFree(gpu_values);
		cudaFree(gpu_size);
		cudaFree(gpu_HMAX);
	}
	
	this->HMAX = numBucketsReshape;
}

bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	
	float fact = this->loadFactor();
	int old_HMAX = this->HMAX;
	if (fact + (float) numKeys / (float) this->HMAX > 0.9f) {
		this->reshape((int)(((float) this->size + (float) numKeys) / 0.8f));
	}
	
	int *gpu_keys = 0;
	int *gpu_values = 0;
	int *gpu_size = 0;
	int *gpu_HMAX = 0;
	cudaMalloc((void **) &gpu_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &gpu_values, numKeys * sizeof(int));
	cudaMalloc((void **) &gpu_size, sizeof(int));
	cudaMalloc((void **) &gpu_HMAX, sizeof(int));
	DIE(gpu_keys == 0 || gpu_values == 0 || gpu_size == 0 ||
		gpu_HMAX == 0, "[insertBatch] malloc error");

	
	cudaMemcpy(gpu_keys, keys, sizeof(int) * numKeys, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_values, values, sizeof(int) * numKeys, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_size, &this->size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_HMAX, &this->HMAX, sizeof(int), cudaMemcpyHostToDevice);

	
	insert>>(this->hash_keys, this->hash_values, this->empty_slots, gpu_keys,
						gpu_values, gpu_size, gpu_HMAX);
	
	cudaDeviceSynchronize(); 

	
	cudaMemcpy(&this->size, gpu_size, sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(gpu_keys);
	cudaFree(gpu_values);
	cudaFree(gpu_size);
	cudaFree(gpu_HMAX);
	
	
	float fact2 = this->loadFactor();
	if (this->size <= old_HMAX && fact2 < 0.8f) {
		this->reshape(old_HMAX);
	}
	
	return true;
}




int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *values = (int *)malloc(numKeys * sizeof(int));
	int *gpu_keys = 0;
	int *gpu_values = 0;
	int *gpu_HMAX = 0;
	cudaMalloc((void **) &gpu_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &gpu_values, numKeys * sizeof(int));
	cudaMalloc((void **) &gpu_HMAX, sizeof(int));
	DIE(gpu_keys == 0 || gpu_values == 0 || gpu_HMAX == 0,
				"[getBatch] malloc error");
	cudaMemset(gpu_values, 0, numKeys * sizeof(int));

	
	cudaMemcpy(gpu_keys, keys, sizeof(int) * numKeys, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_HMAX, &this->HMAX, sizeof(int), cudaMemcpyHostToDevice);

	
	get_val>>(this->hash_keys, this->hash_values, this->empty_slots,
							gpu_keys, gpu_values, gpu_HMAX);
	
	cudaDeviceSynchronize(); 

	
	cudaMemcpy(values, gpu_values, sizeof(int) * numKeys, cudaMemcpyDeviceToHost);
	
	cudaFree(gpu_keys);
	cudaFree(gpu_values);
	cudaFree(gpu_HMAX);



	return values;
}


float GpuHashTable::loadFactor() {
	return (float) this->size / (float) this->HMAX; 
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




class GpuHashTable
{
	public:
		int *hash_keys; 
		int *hash_values; 


		int *empty_slots; 
		int size; 
		int HMAX; 
	
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

