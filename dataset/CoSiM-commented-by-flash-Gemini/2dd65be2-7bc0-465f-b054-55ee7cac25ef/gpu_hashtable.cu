
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include "gpu_hashtable.hpp"

#define MAX_LOAD 0.95

int *hash_keys;
int *hash_values;
int hashsize;
int maxsize;
int inserted = 0;

__global__ void gpu_insert(int *hash_keys, int *hash_values, int *keys,
							int *values, int numKeys, int maxsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numKeys) {
		int keyToAdd = keys[idx];
		int valueToAdd = values[idx];
		
		int hash = hash1(keyToAdd, maxsize);
		int temp = hash;
		do {
			int value = atomicCAS(&hash_keys[hash], 0, keyToAdd);
			
			
			
			
			if (value == keyToAdd || value == 0) {
				hash_values[hash] = valueToAdd;
				break;
			}
			hash++;
			if (hash == maxsize) {
				hash = 0;
			}
		} while (hash != temp);
	}
}

__global__ void gpu_get(int *hash_keys, int *hash_values, int *keysToFind,
						int *valuesToFind, int numKeys, int maxsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numKeys) {
		int keyToFind = keysToFind[idx];
		int hash = hash1(keyToFind, maxsize);
		
		while (1) {
			
			
			
			if (hash_keys[hash] == keyToFind) {
				valuesToFind[idx] = hash_values[hash];
				break;
			}
			hash++;
			if (hash == maxsize) {
				hash = 0;
			}
		}
	}
}

GpuHashTable::GpuHashTable(int size) {
	cudaMalloc((void **) &hash_keys, size * sizeof(int));
	cudaMalloc((void **) &hash_values, size * sizeof(int));
	hashsize = 0;
	maxsize = size;
	cudaMemset(hash_keys, 0, size * sizeof(int));
	cudaMemset(hash_values, 0, size * sizeof(int));
}

GpuHashTable::~GpuHashTable() {
	cudaFree(hash_keys);
	cudaFree(hash_values);
}

void GpuHashTable::reshape(int numBucketsReshape) {
	if (inserted == 0) {
		
		
		maxsize = numBucketsReshape;
		hashsize = 0;


		cudaFree(hash_keys);
		cudaFree(hash_values);
		cudaMalloc((void **) &hash_keys, numBucketsReshape * sizeof(int));
		cudaMalloc((void **) &hash_values, numBucketsReshape * sizeof(int));
		cudaMemset(hash_keys, 0, numBucketsReshape * sizeof(int));
		cudaMemset(hash_values, 0, numBucketsReshape * sizeof(int));
	} else {
		
		
		
		
		int temp_size = maxsize;
		maxsize = numBucketsReshape;
		hashsize = 0;
		int *temp_keys, *temp_values;
		temp_keys = (int *)malloc(temp_size * sizeof(int));


		temp_values = (int *)malloc(temp_size * sizeof(int));
		cudaMemcpy(temp_keys, hash_keys, temp_size * sizeof(int),
											cudaMemcpyDeviceToHost);
		cudaMemcpy(temp_values, hash_values, temp_size * sizeof(int),
												cudaMemcpyDeviceToHost);
		cudaFree(hash_keys);
		cudaFree(hash_values);
		cudaMalloc((void **) &hash_keys, numBucketsReshape * sizeof(int));
		cudaMalloc((void **) &hash_values, numBucketsReshape * sizeof(int));
		cudaMemset(hash_keys, 0, numBucketsReshape * sizeof(int));
		cudaMemset(hash_values, 0, numBucketsReshape * sizeof(int));
		insertBatch(temp_keys, temp_values, temp_size);
		free(temp_keys);
		free(temp_values);
	}
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	
	
	if (hashsize + numKeys > maxsize * MAX_LOAD) {
		reshape(((hashsize + numKeys) * 1.2));
	}
	inserted = 1;
	int *device_keys;
	int *device_values;
	
	
	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int),
									cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int),
										cudaMemcpyHostToDevice);
	gpu_insert>>(hash_keys, hash_values,
										device_keys, device_values,
										numKeys, maxsize);
	cudaDeviceSynchronize();
	cudaFree(device_keys);
	cudaFree(device_values);
	
	hashsize += numKeys;


	return true;
}

int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_keys;
	int *valuesToFind;
	int *valuesToReturn;
	
	
	
	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &valuesToFind, numKeys * sizeof(int));
	valuesToReturn = (int *)malloc(numKeys * sizeof(int));
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int),
									cudaMemcpyHostToDevice);
	gpu_get>>(hash_keys, hash_values,
										device_keys, valuesToFind,
										numKeys, maxsize);
	cudaDeviceSynchronize();
	cudaMemcpy(valuesToReturn, valuesToFind, numKeys * sizeof(int),
												cudaMemcpyDeviceToHost);
	cudaFree(device_keys);
	cudaFree(valuesToFind);
	return valuesToReturn;
}

float GpuHashTable::loadFactor() {
	return (float)hashsize / (float)maxsize;
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
	
__device__ const size_t primeList[] =
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




__device__ int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}




class GpuHashTable
{
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

