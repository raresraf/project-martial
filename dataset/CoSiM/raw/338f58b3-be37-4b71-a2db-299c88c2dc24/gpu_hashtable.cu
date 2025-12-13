
#include 
#include 
#include 
#include 
#include 
#include 

#include
#include "gpu_hashtable.hpp"

#define PERFORMACE_VIEW 1

#define MIN_LOAD_FACTOR 0.9
#define EMPTY_KEY 	0
#define EMPTY_VALUE 0

#define NUM_BLOCKS(n) (((n) + 255) / 256)
#define NUM_THREADS 256



__device__ int hash_func(int k, int htable_size)
{
	k = ((k >> 16) ^ k) * 0x45d9f3b;
    k = ((k >> 16) ^ k) * 0x45d9f3b;
	k = (k >> 16) ^ k;
	
	
    return k % htable_size;
}


__global__ void gpu_init_hashTable(entry_t *htable, const int size)
{
	unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (threadId < size) {
		htable[threadId].key = EMPTY_KEY;
		htable[threadId].value = EMPTY_VALUE;
	}
}


GpuHashTable::GpuHashTable(int size) {

	
	cudaMalloc(&htable, size * sizeof(entry_t));
	DIE(htable == 0, "cudaMalloc htable");

	
	cudaMallocManaged(&count, sizeof(unsigned int));
	DIE(count == 0, "cudaMallocManaged count");

	
	gpu_init_hashTable>>(htable, size);
	cudaDeviceSynchronize();

	
	htable_size = size;
	
	*count = 0;
}


GpuHashTable::~GpuHashTable() {
	
	if (htable != 0)
		cudaFree(htable);

	
	if (count != 0)
		cudaFree(count);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	int size = numBucketsReshape;

	if (htable != 0)
		cudaFree(htable);
	
	
	cudaMalloc(&htable, size * sizeof(entry_t));
	DIE(htable == 0, "cudaMalloc htable");

	if (count != 0)
		cudaFree(count);
	
	
	cudaMallocManaged(&count, sizeof(unsigned int));
	DIE(count == 0, "cudaMallocManaged count");

	
	gpu_init_hashTable>>(htable, size);
	cudaDeviceSynchronize();

	
	htable_size = size;
	
	*count = 0;
}


__global__ void gpu_hashtable_copy(entry_t *old_htable, entry_t *new_htable, const int old_htable_size, const int new_htable_size)
{
	unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (threadId >= old_htable_size)
		return;

	
	int key = old_htable[threadId].key;

	
	if (key == EMPTY_KEY)
		return;
		
	int current_key;
	int index = hash_func(key, new_htable_size);	
	
	
	while (1) {
		
		current_key = atomicCAS(&new_htable[index].key, EMPTY_KEY, key);
		
		
		if (current_key == EMPTY_KEY || current_key == key) {
			new_htable[index].value = old_htable[threadId].value;
			return;
		}

		
		index = (index + 1) % new_htable_size;
	}
}


__global__ void gpu_hashtable_insert(entry_t *htable, unsigned int *count, const int htable_size, const int *keys, const int *values, const int numKeys)
{
	unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (threadId >= numKeys)
		return;

	int key = keys[threadId];
	int value = values[threadId];
	int current_key;
	int index = hash_func(key, htable_size);

	
	while (1) {
		
		current_key = atomicCAS(&htable[index].key, EMPTY_KEY, key);

		
		if (current_key == EMPTY_KEY || current_key == key) {
			htable[index].value = value;
			
			
			if (current_key == EMPTY_KEY)
				atomicAdd(count, 1);
			return;
		}

		
		index = (index + 1) % htable_size;
	}
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	
	int *device_keys;
	int *device_values;
	int old_htable_size;
	entry_t *new_htable;

	
	if (*count + numKeys > MIN_LOAD_FACTOR * htable_size) {

		

		old_htable_size = htable_size;
		
		htable_size = (*count + numKeys) / MIN_LOAD_FACTOR;
		
		
		cudaMalloc(&new_htable, htable_size * sizeof(entry_t));
		DIE(new_htable == 0, "cudaMalloc new_htable");

		
		gpu_init_hashTable>>(new_htable, htable_size);
		cudaDeviceSynchronize();

		
		gpu_hashtable_copy>>(htable, new_htable, old_htable_size, htable_size);
		cudaDeviceSynchronize();

		
		cudaFree(htable);

		
		htable = new_htable;
	}

	


	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));	
	DIE(device_keys == 0, "cudaMalloc device_keys");

	cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	DIE(device_values == 0, "cudaMalloc device_keys");
	
	
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

#if PERFORMACE_VIEW
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
#endif

	
	gpu_hashtable_insert>>(htable, count, htable_size, device_keys, device_values, numKeys);
	cudaDeviceSynchronize();

#if PERFORMACE_VIEW
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float time = 0;
	cudaEventElapsedTime(&time, start, stop);
	float seconds = time / 1000.0f;
	printf("Inserted %d elements in %f ms (%f million keys/second)\n", numKeys, time, numKeys / (double)seconds / 1000000.0f);
#endif

	
	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}


__global__ void gpu_hashtable_lookup(entry_t *htable, const int htable_size, const int *keys, int *values, const int numKeys)
{
	unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (threadId >= numKeys)
		return;

	int key; 
	int index;
	int timeout = 0;
	key = keys[threadId];
	index = hash_func(key, htable_size);

	
	while (true) {

		
		if (timeout == htable_size) {
			values[threadId] = EMPTY_VALUE;
			return;
		}

		
		if (htable[index].key == key) {
			values[threadId] = htable[index].value;
			return;
		}

		
		index = (index + 1) % htable_size;
		timeout += 1;
	}
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *values;
	int *device_keys;
	int *device_values;

	
	values = (int *)malloc(numKeys * sizeof(int));

	


	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));	
	DIE(device_keys == 0, "cudaMalloc device_keys");

	cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	DIE(device_values == 0, "cudaMalloc device_keys");

	
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

#if PERFORMACE_VIEW
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
#endif

	
	gpu_hashtable_lookup>>(htable, htable_size, device_keys, device_values, numKeys);
	cudaDeviceSynchronize();

#if PERFORMACE_VIEW
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float time = 0;
	cudaEventElapsedTime(&time, start, stop);
	float seconds = time / 1000.0f;
	printf("Got %d elements in %f ms (%f million keys/second)\n", numKeys, time, numKeys / (double)seconds / 1000000.0f);
#endif

	
	cudaMemcpy(values, device_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	return values;
}


float GpuHashTable::loadFactor() {
	
	return (float)*count /(float)htable_size; 
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

typedef struct entry{
	int key;
	int value;
}entry_t;

class GpuHashTable
{	
	private:
		unsigned int htable_size;
		unsigned int  *count;
		entry_t *htable;

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

