
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define IN
#define OUT
#define FREE_SPOT -1
#define MAX_OK_LOAD 0.9f
#define HASH_MUL 653267llu
#define HASH_MOD 2740199326961llu
#define EFF_FACT 1.1

__device__ unsigned long long hash_test(int key, unsigned long long mod) {
	return (1ull * key * HASH_MUL) % HASH_MOD % mod;

}

__global__ void kernel_insert(int *keys, int* values, int numberOfPairs,
	Entry* data, int data_size, int *inserted) {
	
	int index =  blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long h;
	int old_key;
	int range[2][2];
	

	if (index >= numberOfPairs)
		return;
	
	h = hash_test(keys[index], data_size);

	range[0][0] = h;
	range[0][1] = data_size;
	range[1][0] = 0;
	range[1][1] = h;
	
	for (int r = 0; r < 2; r++ ) {
		for (int alt = range[r][0]; alt < range[r][1]; ++alt) {
			old_key = atomicCAS(&data[alt].key, FREE_SPOT, keys[index]);

			if (old_key == FREE_SPOT) {
				
				atomicAdd(inserted, 1);
				data[alt].value = values[index];
				return;
			} else if (old_key == keys[index]){
				
				data[alt].value = values[index];
				return;
			} else {
				
				continue;
			}
		}
	}
   
}

__global__ void kernel_get(IN int *keys, OUT int* values,
	IN int numberOfPairs, IN Entry* data, IN int data_size) {

	int index =  blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long h;
	int range[2][2];

	if (index >= numberOfPairs)
		return;

	h = hash_test(keys[index], data_size);
	range[0][0] = h;
	range[0][1] = data_size;
	range[1][0] = 0;
	range[1][1] = h;

	for (int r = 0; r < 2; r++ ) {
		for (int alt = range[r][0]; alt < range[r][1]; ++alt) {


			if (data[alt].key == keys[index]) {
				values[index] = data[alt].value;
				return;
			}
		}
	}

}

__global__ void kernel_init_data(Entry *data, int data_size) {
	int index =  blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= data_size)
	   return;

	data[index].key = -1;
	data[index].value = -1;
}

__global__ void kernel_rehash_transfer(Entry *old_data,
	 int old_size, Entry *new_data, int new_size) {

	int index =  blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long long h;
	int range[2][2];
	int old_key;

	if (index >= old_size) 
	   return;

	if (old_data[index].key == FREE_SPOT)
		return;

	h = hash_test(old_data[index].key, new_size);

	range[0][0] = h;
	range[0][1] = new_size;
	range[1][0] = 0;
	range[1][1] = h;

	for (int r = 0; r < 2; r++ ) {
		for (int alt = range[r][0]; alt < range[r][1]; ++alt) {


			old_key = atomicCAS(&new_data[alt].key, FREE_SPOT, old_data[index].key);

			if (old_key == FREE_SPOT) {
				
				new_data[alt].value = old_data[index].value;
				return;
			}
			
		}
	}
	
}


GpuHashTable::GpuHashTable(int size) {
	cudaError_t err;
	int block_no = size / this->block_size + 1;

	err = cudaMalloc((void **) &this->data, size * sizeof(Entry));
	DIE(err != cudaSuccess, "cudaMalloc");

	kernel_init_data>>(this->data, size);
	cudaDeviceSynchronize();

	this->limit = size * 1ull;
	this->filled = 0;

}


GpuHashTable::~GpuHashTable() {
	if (this->data)
		cudaFree(this->data);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	numBucketsReshape *= EFF_FACT;
	Entry *new_data;
	cudaError_t err;
	int block_no = numBucketsReshape / this->block_size + 1;

	err = cudaMalloc((void **) &new_data, numBucketsReshape * sizeof(Entry));
	DIE(err != cudaSuccess, "cudaMalloc");

	kernel_init_data>>(new_data, numBucketsReshape);
	cudaDeviceSynchronize();

	


	kernel_rehash_transfer>>(data, limit, new_data, numBucketsReshape);
	cudaDeviceSynchronize();

	err = cudaFree(this->data);
	DIE(err != cudaSuccess, "cudaMalloc");

	this->data = new_data;
	this->limit = numBucketsReshape;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int block_no;
	int *device_keys;
	int *device_values;
	cudaError_t err;

	if ((this->filled + numKeys) > MAX_OK_LOAD * this->limit) {
		reshape((this->filled + numKeys));
	}

	int tally, *dev_tally;
	cudaMalloc((void **)&dev_tally, sizeof(int));
	tally = 0;
	cudaMemcpy(dev_tally, &tally, sizeof(int), cudaMemcpyHostToDevice);

	err = cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	block_no = numKeys / this->block_size + 1;

	kernel_insert>>(
		device_keys, device_values, numKeys, this->data, this->limit, dev_tally
	);
	
	cudaDeviceSynchronize();
	cudaMemcpy(&tally, dev_tally, sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(device_keys);
	cudaFree(device_values);

	this->filled += tally;
	cudaFree(dev_tally);
	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_values = NULL;
	int *device_keys = NULL;
	int *host_values = NULL;
	int block_no;
	cudaError_t err;

	
	err = cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	DIE(err != cudaSuccess, "cudaMalloc");

	host_values = (int *)malloc(numKeys * sizeof(int));
	
	block_no = numKeys / this->block_size + 1;
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	kernel_getblock_size>>>(device_keys, device_values, numKeys,
		 this->data, this->limit);
	cudaDeviceSynchronize();

	
	cudaMemcpy(host_values, device_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(device_values);
	cudaFree(device_keys);

	return host_values;
}


float GpuHashTable::loadFactor() {
	float lf = 0.0f;
	if (limit >= 0.0f)
		lf = (float)((float)filled / (float)limit);
	return lf; 
}

unsigned long long GpuHashTable::occupancy() {
	return filled;
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

using namespace std;

#define KEY_INVALID 0

#define DIE(assertion, call_description)                                       \
	do {                                                                   \
		if (assertion) {                                               \
			fprintf(stderr, "(%s, %d): ", __FILE__, __LINE__);     \
			perror(call_description);                              \
			exit(errno);                                           \
		}                                                              \
	} while (0)

const size_t primeList[] = {2llu,
			    3llu,
			    5llu,
			    7llu,
			    11llu,
			    13llu,
			    17llu,
			    23llu,
			    29llu,
			    37llu,
			    47llu,
			    59llu,
			    73llu,
			    97llu,
			    127llu,
			    151llu,
			    197llu,
			    251llu,
			    313llu,
			    397llu,
			    499llu,
			    631llu,
			    797llu,
			    1009llu,
			    1259llu,
			    1597llu,
			    2011llu,
			    2539llu,
			    3203llu,
			    4027llu,
			    5087llu,
			    6421llu,
			    8089llu,
			    10193llu,
			    12853llu,
			    16193llu,
			    20399llu,
			    25717llu,
			    32401llu,
			    40823llu,
			    51437llu,
			    64811llu,
			    81649llu,
			    102877llu,
			    129607llu,
			    163307llu,
			    205759llu,
			    259229llu,
			    326617llu,
			    411527llu,
			    518509llu,
			    653267llu,
			    823117llu,
			    1037059llu,
			    1306601llu,
			    1646237llu,
			    2074129llu,
			    2613229llu,
			    3292489llu,
			    4148279llu,
			    5226491llu,
			    6584983llu,
			    8296553llu,
			    10453007llu,
			    13169977llu,
			    16593127llu,
			    20906033llu,
			    26339969llu,
			    33186281llu,
			    41812097llu,
			    52679969llu,
			    66372617llu,
			    83624237llu,
			    105359939llu,
			    132745199llu,
			    167248483llu,
			    210719881llu,
			    265490441llu,
			    334496971llu,
			    421439783llu,
			    530980861llu,
			    668993977llu,
			    842879579llu,
			    1061961721llu,
			    1337987929llu,
			    1685759167llu,
			    2123923447llu,
			    2675975881llu,
			    3371518343llu,
			    4247846927llu,
			    5351951779llu,
			    6743036717llu,
			    8495693897llu,
			    10703903591llu,
			    13486073473llu,
			    16991387857llu,
			    21407807219llu,
			    26972146961llu,
			    33982775741llu,
			    42815614441llu,
			    53944293929llu,
			    67965551447llu,
			    85631228929llu,
			    107888587883llu,
			    135931102921llu,
			    171262457903llu,
			    215777175787llu,
			    271862205833llu,
			    342524915839llu,
			    431554351609llu,
			    543724411781llu,
			    685049831731llu,
			    863108703229llu,
			    1087448823553llu,
			    1370099663459llu,
			    1726217406467llu,
			    2174897647073llu,
			    2740199326961llu,
			    3452434812973llu,
			    4349795294267llu,
			    5480398654009llu,
			    6904869625999llu,
			    8699590588571llu,
			    10960797308051llu,
			    13809739252051llu,
			    17399181177241llu,
			    21921594616111llu,
			    27619478504183llu,
			    34798362354533llu,
			    43843189232363llu,
			    55238957008387llu,
			    69596724709081llu,
			    87686378464759llu,
			    110477914016779llu,
			    139193449418173llu,
			    175372756929481llu,
			    220955828033581llu,
			    278386898836457llu,
			    350745513859007llu,
			    441911656067171llu,
			    556773797672909llu,
			    701491027718027llu,
			    883823312134381llu,
			    1113547595345903llu,
			    1402982055436147llu,
			    1767646624268779llu,
			    2227095190691797llu,
			    2805964110872297llu,
			    3535293248537579llu,
			    4454190381383713llu,
			    5611928221744609llu,
			    7070586497075177llu,
			    8908380762767489llu,
			    11223856443489329llu,
			    14141172994150357llu,
			    17816761525534927llu,
			    22447712886978529llu,
			    28282345988300791llu,
			    35633523051069991llu,
			    44895425773957261llu,
			    56564691976601587llu,
			    71267046102139967llu,
			    89790851547914507llu,
			    113129383953203213llu,
			    142534092204280003llu,
			    179581703095829107llu,
			    226258767906406483llu,
			    285068184408560057llu,
			    359163406191658253llu,
			    452517535812813007llu,
			    570136368817120201llu,
			    718326812383316683llu,
			    905035071625626043llu,
			    1140272737634240411llu,
			    1436653624766633509llu,
			    1810070143251252131llu,
			    2280545475268481167llu,
			    2873307249533267101llu,
			    3620140286502504283llu,
			    4561090950536962147llu,
			    5746614499066534157llu,
			    7240280573005008577llu,
			    9122181901073924329llu,
			    11493228998133068689llu,
			    14480561146010017169llu,
			    18446744073709551557llu};




int hash1(int data, int limit)
{
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit)
{
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit)
{
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}

class Entry
{
      public:
	int key;
	int value;
};




class GpuHashTable
{
      public:
	GpuHashTable(int size);
	void reshape(int sizeReshape);

	bool insertBatch(int *keys, int *values, int numKeys);
	int *getBatch(int *key, int numItems);

	float loadFactor();
	unsigned long long occupancy();
	void print(string info);

	~GpuHashTable();

	unsigned long long filled;
	unsigned long long limit;
	const int block_size = 1024;

	Entry *data;
};

#endif
