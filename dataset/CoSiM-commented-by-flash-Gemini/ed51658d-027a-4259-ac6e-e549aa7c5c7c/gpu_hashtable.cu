
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"
#define MIN_LOAD_FACTOR 85


__device__ int hashFunc(int data, int size) {
	return ((long)abs(data) * 905035071625626043llu) % 5746614499066534157llu % size;
}
__global__ void kernel_update_hash(hashtable *hash, hashelement *table, int capacity, int items) {
	hash->table = table;
	hash->capacity = capacity;
	hash->numElements = items;
}
__global__ void kernel_insert(int *keys, int *values, int numElements, hashtable *hash) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int f = 0;
	if (index > numElements)
		return;
	int key = keys[index];
	int value = values[index];
	if (key == 0 || value == 0)
		return;
	int valueHash = hashFunc(key, hash->capacity);
	int tableIndex = valueHash;
	hashelement elementToInsert;
	elementToInsert.key = key;
	elementToInsert.value = value;
	for (int i = 0; i capacity; ++i) {
		hashelement oldElement = hash->table[tableIndex];
		hashelement element;
		element.key = atomicCAS(&oldElement.key, 0, elementToInsert.key);
		printf("element.key%d\n", element.key);
		if (element.key == 0) {
			atomicAdd(&hash->numElements, 1);
			printf("number of elements%d\n", hash->numElements);
			hash->table[tableIndex].value = value;
			int a = hash->table[tableIndex].value;
			return;
		}
		else if (element.key == key) {
			hash->table[tableIndex].value = value;
			return;
		}
		tableIndex = (valueHash + i * f) % hash->capacity;
		f += 1;
	}
}


__global__ void kernel_rehash(hashtable *oldHash, hashelement *newTable, int newCapacity) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int f = 0;
	if (index >= oldHash->capacity)
		return;

	hashelement elementToInsert = oldHash->table[index];
	if (elementToInsert.key == 0) {
		return;
	}
	int valueHash = hashFunc(elementToInsert.key, newCapacity);
	int tableIndex = valueHash;

	for (int i = 0; i < newCapacity; ++i) {
		hashelement element;
		hashelement oldElement = newTable[tableIndex];
		element.key = atomicCAS(&oldElement.key, 0, elementToInsert.key);
		if (element.key == 0) {
			newTable[tableIndex].value = elementToInsert.value;
			return;
		}
		tableIndex = (valueHash + i * f) % newCapacity;
		f += 1;
	}
}
__global__ void kernel_get(int *keys, int *values, int numElements, hashtable *hash) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int f = 0;
	if (index >= numElements)
		return;
	int key = keys[index];
	int valueHash = hashFunc(key, hash->capacity);
	int tableIndex = valueHash;
	hashelement elementToGet;
	for (int i = 0; i capacity; ++i) {
		elementToGet = hash->table[tableIndex];
		if (elementToGet.key == key) {
			values[index] = elementToGet.value;
			return;
		}
		tableIndex = (valueHash + i * f) % hash->capacity;
		f += 1;
	}
}
GpuHashTable::GpuHashTable(int size) {
	hashelement *table = NULL;
	cudaMalloc((void **)&table, size * sizeof(hashelement));
	cudaMemset(table, 0, size);

	cudaMalloc((void **)&hash, sizeof(hashtable));

	kernel_update_hash > >(hash, table, size, 0);

	cudaDeviceSynchronize();
}


GpuHashTable::~GpuHashTable() {
	cudaFree(hash->table);
	cudaFree(hash);
}


void GpuHashTable::reshape(int batchSize) {
	hashtable *hostTable = (hashtable *)calloc(1, sizeof(hashtable));
	cudaMemcpy(hostTable, hash, sizeof(hashtable), cudaMemcpyDeviceToHost);
	float loadFactor;
	loadFactor = ((float)hostTable->numElements + (float)batchSize) / ((float)hostTable->capacity);
	if (loadFactor > 0.9) {
		int newCapacity = ((float)hostTable->numElements + (float)batchSize) * 100 / (float)MIN_LOAD_FACTOR;
		printf("%d\n\n", newCapacity);
		hashelement *newTable = NULL;
		cudaMalloc((void **)&newTable, newCapacity * sizeof(hashelement));
		cudaMemset(newTable, 0, newCapacity);
		int threadsPerBlock = 1024;
		int blocks = hostTable->capacity / 512 + 1;


		kernel_rehash > >(hash, newTable, newCapacity);
		cudaDeviceSynchronize();
		cudaFree(hostTable->table);
		
		hostTable->table = newTable;
		hostTable->capacity = newCapacity;
		
		
		cudaMemcpy(hash, hostTable, sizeof(hashtable), cudaMemcpyHostToDevice);
		cudaMemcpy(&hash->capacity, &newCapacity, sizeof(int), cudaMemcpyHostToDevice);
	}
	free(hostTable);
}

bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	reshape(numKeys);
	int threadsPerBlock = 1024; 
	int blocks = numKeys / threadsPerBlock + 1;
	int *deviceKeys = NULL;
	int *deviceValues = NULL;
	cudaMalloc((void **)&deviceKeys, numKeys * sizeof(int));
	cudaMalloc((void **)&deviceValues, numKeys * sizeof(int));
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);


	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	kernel_insert > >(deviceKeys, deviceValues, numKeys, hash);
	cudaDeviceSynchronize();
	cudaFree(deviceKeys);
	cudaFree(deviceValues);
	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *getValues = NULL;
	int *deviceKeys = NULL;
	cudaMalloc((void **)&getValues, numKeys * sizeof(int));
	cudaMalloc((void **)&deviceKeys, numKeys * sizeof(int));
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	int threadsPerBlock = 1024; 
	int blocks = numKeys / threadsPerBlock + 1;
	kernel_get > >(deviceKeys, getValues, numKeys, hash);
	int *hostValues = (int *)calloc(numKeys, sizeof(int));
	cudaMemcpy(hostValues, getValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(deviceKeys);
	return hostValues;
}


float GpuHashTable::loadFactor() {
	
	int numElements = 0;
	int capacity = 0;
	cudaMemcpy(&numElements, &hash->numElements, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&capacity, &hash->capacity, sizeof(int), cudaMemcpyDeviceToHost);
	float loadFactor;
	loadFactor = ((float)numElements) / ((float)capacity);
	return loadFactor;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 

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

using namespace std;

void fillRandom(vector &vecKeys, vector &vecValues, int numEntries) {
	vecKeys.reserve(numEntries);
	vecValues.reserve(numEntries);

	int interval = (numeric_limits::max() / numEntries) - 1;
	default_random_engine generator;
	uniform_int_distribution distribution(1, interval);

	for (int i = 0; i < numEntries; i++) {
		vecKeys.push_back(interval * i + distribution(generator));
		vecValues.push_back(interval * i + distribution(generator));
	}

	random_shuffle(vecKeys.begin(), vecKeys.end());
	random_shuffle(vecValues.begin(), vecValues.end());
}

int main(int argc, char **argv)
{
	clock_t begin;
	double elapsedTime;

	int numKeys = 0;
	int numChunks = 0;
	vector vecKeys;
	vector vecValues;
	int *valuesGot = NULL;

	DIE(argc != 3,
		"ERR, args num, call ./bin test_numKeys test_numChunks");

	numKeys = stoll(argv[1]);
	DIE((numKeys = numeric_limits::max()),
		"ERR, numKeys should be greater or equal to 1 and less than maxint");

	numChunks = stoll(argv[2]);
	DIE((numChunks = numKeys),
		"ERR, numChunks should be greater or equal to 1");

	fillRandom(vecKeys, vecValues, numKeys);

	HASH_INIT;

	int chunkSize = numKeys / numChunks;
	HASH_RESERVE(chunkSize);

	
	for (int chunkStart = 0; chunkStart < numKeys; chunkStart += chunkSize) {

		int* keysStart = &vecKeys[chunkStart];
		int* valuesStart = &vecValues[chunkStart];

		begin = clock();
		
		HASH_BATCH_INSERT(keysStart, valuesStart, chunkSize);
		elapsedTime = double(clock() - begin) / CLOCKS_PER_SEC;

		cout << "HASH_BATCH_INSERT, " << chunkSize
			<< ", " << chunkSize / elapsedTime / 1000000
			<< ", " << 100.f * HASH_LOAD_FACTOR << endl;
	}

	
	int chunkSizeUpdate = min(64, numKeys);
	for (int chunkStart = 0; chunkStart < chunkSizeUpdate; chunkStart++) {
		vecValues[chunkStart] += 1111111 + chunkStart;
	}
	HASH_BATCH_INSERT(&vecKeys[0], &vecValues[0], chunkSizeUpdate);

	
	for (int chunkStart = 0; chunkStart < numKeys; chunkStart += chunkSize) {

		int* keysStart = &vecKeys[chunkStart];

		begin = clock();
		
		valuesGot = HASH_BATCH_GET(keysStart, chunkSize);
		elapsedTime = double(clock() - begin) / CLOCKS_PER_SEC;

		cout << "HASH_BATCH_GET, " << chunkSize
			<< ", " << chunkSize / elapsedTime / 1000000
			<< ", " << 100.f * HASH_LOAD_FACTOR << endl;

		DIE(valuesGot == NULL, "ERR, ptr valuesCheck cannot be NULL");

		int mistmatches = 0;
		for (int i = 0; i < chunkSize; i++) {
			if (vecValues[chunkStart + i] != valuesGot[i]) {
				mistmatches++;
				if (mistmatches < 32) {
					cout << "Expected " << vecValues[chunkStart + i]
						<< ", but got " << valuesGot[i] << " for key:" << keysStart[i] << endl;
				}
			}
		}

		if (mistmatches > 0) {
			cout << "ERR, mistmatches: " << mistmatches << " / " << numKeys << endl;
			exit(1);
		}
	}

	return 0;
}


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
typedef struct {
	int key;
	int value;
} hashelement;

typedef struct {
	hashelement *table;
	int capacity;
	int numElements;
} hashtable;
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
	hashtable *hash;
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

