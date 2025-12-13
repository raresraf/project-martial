
#include 
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

__global__ void kernel_reshape(hashtable ht, hashtable newHt) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < ht.size) {

		
		
		int oldKey = ht.elements[idx].key;

		if (oldKey == KEY_INVALID)
			return;

		
		int hash_value = my_hash(oldKey, newHt.size);

		
		int rez, valid_hash;
		do {
			rez = atomicCAS(&(newHt.elements[hash_value].key),
				KEY_INVALID, oldKey);

			
			valid_hash = hash_value;
			
			hash_value += 1;
			if(hash_value >= newHt.size)
				hash_value = hash_value % newHt.size;

		} while (rez != KEY_INVALID);
		
		newHt.elements[valid_hash].value = ht.elements[idx].value;
	}
}

__global__ void kernel_insert(int *deviceKeys, int *deviceValues, int numKeys, hashtable ht) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numKeys) {

		
		
		int oldKey = deviceKeys[idx];
		if (oldKey == KEY_INVALID)
			return;

		
		int hash_value = my_hash(oldKey, ht.size);

		
		int rez, valid_hash;
		do {
			rez = atomicCAS(&(ht.elements[hash_value].key),
				KEY_INVALID, oldKey);

			
			valid_hash = hash_value;
			
			hash_value += 1;
			if(hash_value >= ht.size)
				hash_value = hash_value % ht.size;

		} while (rez != KEY_INVALID && rez != oldKey);
		
		
		
		ht.elements[valid_hash].value = deviceValues[idx];
	}
}

__global__ void kernel_get(int *deviceKeys, int *deviceValues, int numKeys, hashtable ht) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numKeys) {

		
		
		int oldKey = deviceKeys[idx];

		if (oldKey == KEY_INVALID)
			return;

		
		int hash_value = my_hash(oldKey, ht.size);

		
		
		int index = 0, valid_value, found = 0;
		while (index < ht.size){

			if(ht.elements[hash_value].key == oldKey) {
				
				valid_value = ht.elements[hash_value].value;
				found = 1; 
				break;
			}
			
			hash_value += 1;
			if(hash_value >= ht.size)
				hash_value = hash_value % ht.size;
			index++;
		}
		if(found == 1) {
			
			
			deviceValues[idx] = valid_value;
		}
	}
}


GpuHashTable::GpuHashTable(int size) {

	cudaError_t res;

	ht.size = size;
	ht.nrElems = 0;
	
	res = cudaMalloc(&ht.elements, size * sizeof(elem));
	if (res != cudaSuccess) {
		cout << "[init] Memory allocation error\n";
		return;
	}
	
	res = cudaMemset(ht.elements, 0, size * sizeof(elem));
	if (res != cudaSuccess) {
		cout << "[init] Memset error\n";
		return;
	}
}


GpuHashTable::~GpuHashTable() {

	cudaError_t res;

	
	res = cudaFree(ht.elements);
	if (res != cudaSuccess) {
		cout << "[destroy] Free error\n";
		return;
	}
}


void GpuHashTable::reshape(int numBucketsReshape) {

	cudaError_t res;
	hashtable newHt;

	newHt.size = numBucketsReshape;
	newHt.nrElems = 0;
	
	res = cudaMalloc(&newHt.elements, numBucketsReshape * sizeof(elem));
	if (res != cudaSuccess) {
		cout << "[reshape] Memory allocation error\n";
		return;
	}
	
	res = cudaMemset(newHt.elements, 0, numBucketsReshape * sizeof(elem));
	if (res != cudaSuccess) {
		cout << "[reshape] Memset error\n";
		return;
	}

	
	


	int blocks = (ht.size % NR_THREADS != 0) ?
		ht.size / NR_THREADS + 1 : ht.size / NR_THREADS;
	kernel_reshape>>(ht, newHt);

	res = cudaDeviceSynchronize();
	if (res != cudaSuccess) {
		cout << "[reshape] Syncronize error\n";
		return;
	}

	
	res = cudaFree(ht.elements);
	if (res != cudaSuccess) {
		cout << "[reshape] Free error\n";
		return;
	}

	
	newHt.nrElems = ht.nrElems;
	ht = newHt;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {

	
	if (numKeys + ht.nrElems > ht.size || 
		(float)(ht.nrElems + numKeys) / ht.size >= LOAD_FACTOR_MAX) {
			reshape((int)((ht.nrElems + numKeys) / LOAD_FACTOR_MIN));
	}

	cudaError_t res;

	
	
	int *deviceKeys, *deviceValues;
	res = cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	if (res != cudaSuccess) {
		cout << "[insert] Memory allocation error\n";
		return false;
	}
	res = cudaMalloc(&deviceValues, numKeys * sizeof(int));
	if (res != cudaSuccess) {
		cudaFree(deviceKeys);
		cout << "[insert] Memory allocation error\n";
		return false;
	}

	
	res = cudaMemcpy(deviceKeys, keys, numKeys *
		sizeof(int), cudaMemcpyHostToDevice);
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		cudaFree(deviceKeys);
		cout << "[insert] Memory allocation error\n";
		return false;
	}
	res = cudaMemcpy(deviceValues, values,
		numKeys * sizeof(int), cudaMemcpyHostToDevice);
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		cudaFree(deviceKeys);
		cout << "[insert] Memcpy error\n";
		return false;
	}

	


	int blocks = (numKeys % NR_THREADS != 0) ?
		numKeys / NR_THREADS + 1 : numKeys / NR_THREADS;
	kernel_insert>>
		(deviceKeys, deviceValues, numKeys, ht);

	res = cudaDeviceSynchronize();
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		cout << "[insert] Syncronize error\n";
		return false;
	}

	
	res = cudaFree(deviceKeys);
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		cout << "[insert] Free error\n";
		return false;
	}
	res = cudaFree(deviceValues);
	if (res != cudaSuccess) {
		cout << "[insert] Free error\n";
		return false;
	}

	
	ht.nrElems += numKeys;

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	
	cudaError_t res;

	
	int *deviceKeys, *deviceValues, *values;
	res = cudaMalloc(&deviceKeys, numKeys * sizeof(int));
	if (res != cudaSuccess) {
		cout << "[get] Memory allocation error\n";
		return NULL;
	}
	res = cudaMalloc(&deviceValues, numKeys * sizeof(int));
	if (res != cudaSuccess) {
		cout << "[get] Memory allocation error\n";
		return NULL;
	}
	values = (int *) malloc(numKeys * sizeof(int));
	if (!values) {
		cout << "[get] Memory allocation error\n";
		return NULL;
	}

	
	res = cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int),
		cudaMemcpyHostToDevice);
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		cudaFree(deviceKeys);
		free(values);
		cout << "[get] Memcpy error1\n";
		return NULL;
	}

	
	


	int blocks = (numKeys % NR_THREADS != 0) ?
		numKeys / NR_THREADS + 1 : numKeys / NR_THREADS;
	kernel_get>>
		(deviceKeys, deviceValues, numKeys, ht);

	res = cudaDeviceSynchronize();
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		cudaFree(deviceKeys);
		free(values);
		cout << "[get] Syncronize error\n";
		return NULL;
	}

	
	res = cudaMemcpy(values, deviceValues,
		numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		cudaFree(deviceKeys);
		free(values);
		cout << "[get] Memcpy error3\n";
		return NULL;
	}

	
	res = cudaFree(deviceKeys);
	if (res != cudaSuccess) {
		cudaFree(deviceValues);
		free(values);
		cout << "[get] Free error\n";
		return NULL;
	}
	res = cudaFree(deviceValues);
	if (res != cudaSuccess) {
		free(values);
		cout << "[get] Free error\n";
		return NULL;
	}

	
	return values;
}


float GpuHashTable::loadFactor() {

	
	if(ht.size == 0)
		return 0;

	
	
	return (float)(ht.nrElems) / ht.size;
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
#define NR_THREADS		1024
#define LOAD_FACTOR_MAX		0.99
#define LOAD_FACTOR_MIN		0.8

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




__device__ int my_hash(int data, int limit) {
	return ((long)abs(data) * 52679969llu) % 28282345988300791llu % limit;
}

typedef struct elem {
	int key;
	int value;
} elem;

typedef struct hashtable {
	int size;
	int nrElems;
	elem *elements;
} hashtable;




class GpuHashTable
{
	public:
		hashtable ht;

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

