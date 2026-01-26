
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

__device__ int hash_function(int size, int key) {
	const size_t prime_number_1 = 13169977llu;
	const size_t prime_number_2 = 5351951779llu;

	int val = ((long)abs(key) * prime_number_1) % prime_number_2 % size;

	return val;
}

__global__ void  resize(my_hash newHash, my_hash hmap) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int idx, x, slot;
	int once = 0;

	if (!(i < hmap.hsize))
		return;

	for (slot = 0; slot < NO_SLOTS; slot++) {
		once = 0;

		if (hmap.buckets[slot][i].key == KEY_INVALID)
				continue;

		idx = hash_function(newHash.hsize, hmap.buckets[slot][i].key);

		once = 0;
		int oldIdx = idx;
		while(idx < newHash.hsize && once < 2) {
			if (once == 1 && idx > oldIdx)
				break;

			for(x = 0; x < NO_SLOTS; x++) {

				int old = atomicCAS(&newHash.buckets[x][idx].key, KEY_INVALID, hmap.buckets[slot][i].key);
				if (idx < newHash.hsize && (old == KEY_INVALID)) {
					atomicExch(&newHash.buckets[x][idx].value, hmap.buckets[slot][i].value);

					once = 3;
					break;

				}
			}

			idx++;
			if (idx >= newHash.hsize) {
				idx = 0;
				once++;
			}
		}
	}

	return;
}



__global__ void insert(int *keys, int *values, int numKeys, my_hash hmap) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int x = 0;
	int idx = 0;
	
	
	int once = 0;

	if (!(i < numKeys))
		return;

	if (keys[i] <= 0)
		return;

	idx = hash_function(hmap.hsize, keys[i]);

	once = 0;
	int oldIdx = idx;
	while(idx < hmap.hsize && once < 2) {
		if (once == 1 && idx > oldIdx)
			break;
		for(x = 0; x < NO_SLOTS; x++) {

			int old = atomicCAS(&hmap.buckets[x][idx].key, KEY_INVALID, keys[i]);
			if (idx < hmap.hsize && (old == KEY_INVALID || old == keys[i])) {
				atomicExch(&hmap.buckets[x][idx].value, values[i]);
				return;
			}
		}

		idx++;
		if (idx >= hmap.hsize) {
			idx = 0;
			once++;
		}
	}
}



__global__ void get(int *keys, int *values, int numKeys, my_hash hmap) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int x = 0;
	int idx;
	int once = 0;

	if (!(i < numKeys))
		return;

	if (keys[i] == KEY_INVALID)
		return;

	idx = hash_function(hmap.hsize, keys[i]);

	once = 0;
	int oldIdx = idx;
	while(idx < hmap.hsize && once < 2) {
		if (once == 1 && idx > oldIdx)
			break;

		for(x = 0; x < NO_SLOTS; x++) {

			int old = atomicCAS(&hmap.buckets[x][idx].key, KEY_INVALID, keys[i]);
			if (idx < hmap.hsize &&  hmap.buckets[x][idx].key == keys[i]) {
				values[i] = hmap.buckets[x][idx].value;
				return;
			}
		}

		idx++;
		if (idx >= hmap.hsize) {
			idx = 0;
			once++;
		}
	}
}


GpuHashTable::GpuHashTable(int size) {
	int i;
	cudaError_t err;
	hmap.hsize = size;

	for (i = 0; i < NO_SLOTS; i++) {
		err = cudaMalloc((void **)&hmap.buckets[i], size * sizeof(list));
		DIE(err != cudaSuccess || hmap.buckets[i]  == 0, "cudaMalloc");
	}

	for (i = 0; i < NO_SLOTS; i++) {
		err = cudaMemset(hmap.buckets[i], 0, size * sizeof(list));
		DIE(err != cudaSuccess, "cudaMemset");
	}
	no_insPairs = 0;
}


GpuHashTable::~GpuHashTable() {
	int i;
	no_insPairs = 0;
	cudaError_t err;

	for (i = 0; i < NO_SLOTS; i++) {
		err = cudaFree(hmap.buckets[i]);
		DIE(err != cudaSuccess, "cudaMalloc");
		hmap.buckets[i] = nullptr;
	}
}


void GpuHashTable::reshape(int numBucketsReshape) {
	int i;
	cudaError_t err;
	my_hash newHash;
	int N = numBucketsReshape;

	for (i = 0; i < NO_SLOTS; i++) {
		err = cudaMalloc(&newHash.buckets[i], N * sizeof(list));
		DIE(err != cudaSuccess || newHash.buckets[i]  == 0, "cudaMalloc");
	}

	for (i = 0; i < NO_SLOTS; i++) {
		err = cudaMemset(newHash.buckets[i], 0, N * sizeof(list));
		DIE(err != cudaSuccess, "cudaMemset");
	}

	newHash.hsize = numBucketsReshape;

	const size_t block_size = 256;


	size_t blocks_no = hmap.hsize / block_size;

	if (hmap.hsize % block_size)
		++blocks_no;

	
	resize>>(newHash, hmap);
	cudaDeviceSynchronize();

	for (i = 0; i < NO_SLOTS; i++) {
		err = cudaFree(hmap.buckets[i]);
		DIE(err != cudaSuccess, "[resize] cudaFree");
		hmap.buckets[i] = nullptr;
	}

	hmap = newHash;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys = 0;
	int *device_values = 0;
	cudaError_t err;
	int num_bytes = numKeys * sizeof(*keys);

	if (!(float(no_insPairs + numKeys) / (hmap.hsize) < MAX_LOAD))
		reshape(int((no_insPairs + numKeys) / MIN_LOAD));

	
	err = cudaMalloc((void **) &device_keys, num_bytes);
	DIE(err != cudaSuccess, "[INSERT] cudaMalloc");
	err = cudaMalloc((void **) &device_values, num_bytes);
	DIE(err != cudaSuccess, "[INSERT] cudaMalloc");

	if (device_keys == 0 || device_values == 0) {
		printf("[HOST] Couldn't allocate memory\n");
		return false;
	}

	
	err = cudaMemcpy(device_keys, keys, num_bytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "[INSERT] cudaMemcpy");

	err =  cudaMemcpy(device_values, values, num_bytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "[INSERT] cudaMemcpy");

	
	
	
	const size_t block_size = 256;
	size_t blocks_no = numKeys / block_size;

	if (numKeys % block_size)
		++blocks_no;

	
	insert>>(device_keys, device_values, numKeys, hmap);
	cudaDeviceSynchronize();

	err = cudaFree(device_values);
	DIE(err != cudaSuccess, "[INSERT] cudaFree");
	err = cudaFree(device_keys);
	DIE(err != cudaSuccess, "[INSERT] cudaFree");

	no_insPairs +=  numKeys;
	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *host_values = 0;
	int *device_keys = 0;
	int *device_values = 0;
	int num_bytes = numKeys * sizeof(int);
	cudaError_t err;

	
	host_values = (int *) calloc(1, num_bytes);
	DIE(host_values == NULL, "[GET] calloc");

	
	err = cudaMalloc((void **) &device_keys, num_bytes);
	DIE(err != cudaSuccess, "[GET] cudaMalloc");

	err = cudaMalloc((void **) &device_values, num_bytes);
	DIE(err != cudaSuccess, "[GET] cudaMalloc");


	if (device_keys == 0 || device_values == 0) {
		printf("[GET HOST] Couldn't allocate memory\n");
		return NULL;
	}

	
	err = cudaMemcpy(device_keys, keys, num_bytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "[GET] cudaMemcpy");

	
	
	
	const size_t block_size = 256;
	size_t blocks_no = numKeys / block_size;

	if (numKeys % block_size)
		++blocks_no;

	
	get>>(device_keys, device_values, numKeys, hmap);
	cudaDeviceSynchronize();

	err = cudaMemcpy(host_values, device_values, num_bytes, cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, "[GET] cudaMemcpy");

	err = cudaFree(device_values);
	DIE(err != cudaSuccess, "[GET] cudaFree");

	return host_values;
}


float GpuHashTable::loadFactor() {
	if (hmap.hsize == 0)
		return 0;

	return float(no_insPairs) / (hmap.hsize);
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
#define NO_SLOTS		2
#define MIN_LOAD		0.8
#define MAX_LOAD		0.9

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


struct list {
	int key;
	int value;
};


struct my_hash {
	int hsize;
	list *buckets[NO_SLOTS];
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
		my_hash hmap;
		int no_insPairs;
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

