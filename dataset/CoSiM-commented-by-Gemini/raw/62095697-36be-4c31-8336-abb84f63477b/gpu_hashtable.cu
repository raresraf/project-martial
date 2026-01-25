
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define PRIME1 49157
#define PRIME2 402653189

__device__ int h(int data, int limit) {


	return ((long)abs(data) * PRIME1) % PRIME2 % limit;
}

__global__ void kernel_insert(Bucket *buckets, int *keys, int* values,
 		int numKeys, int buckets_no, int *occupied) {

	int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_idx >= numKeys)
		return;

	int key = keys[thread_idx];
	int value = values[thread_idx];

	
	if (key == KEY_INVALID){
		return;
	}

	int key_idx = h(key, buckets_no);
	int old_key;

	
	for (int step = 0; step < buckets_no; step++) {
		
		int idx = (key_idx + step) % buckets_no;

		
		for (int i = 0; i < BUCKET_SIZE; i++) {
			old_key = atomicCAS(&buckets[idx].entries[i].key, key, key);
			if (old_key == key) {
				atomicExch(&buckets[idx].entries[i].value, value);
				return;
			}
		}

		
		for (int i = 0; i < BUCKET_SIZE; i++) {
			old_key = atomicCAS(&buckets[idx].entries[i].key, 0, key);
			if (old_key == 0) {
				atomicExch(&buckets[idx].entries[i].value, value);
				atomicAdd(occupied, 1);
				return;
			}
		}
	}
}



__global__ void kernel_get(Bucket *buckets, int *keys, int* values, int numKeys, int buckets_no) {
	int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_idx >= numKeys)
		return;

	int key = keys[thread_idx];
	int key_idx = h(key, buckets_no);

	
	for (int step = 0; step < buckets_no; step++) {
		int idx = (key_idx + step) % buckets_no;

		
		for (int i = 0; i < BUCKET_SIZE; i++) {
			if (buckets[idx].entries[i].key == key) {
				values[thread_idx] = buckets[idx].entries[i].value;
				return;
			}

			
			if (buckets[idx].entries[i].key == 0) {
				values[thread_idx] = 0;
				return;
			}
		}
	}

	values[thread_idx] = 0;
}



__global__ void kernel_reshape(Bucket *new_buckets, Bucket *old_buckets,
		int new_buckets_no, int old_buckets_no) {

	
	int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_idx >= old_buckets_no * BUCKET_SIZE)
		return;

	
	int bucket_idx = thread_idx / BUCKET_SIZE;
	int entry_idx = thread_idx % BUCKET_SIZE;

	
	int key = old_buckets[bucket_idx].entries[entry_idx].key;
	int value =  old_buckets[bucket_idx].entries[entry_idx].value;

	
	if (key == KEY_INVALID)
		return;

	int key_idx = h(key, new_buckets_no);

	int old_key, step = 0;
	bool place_found = false;

	
	while ( !place_found && step < new_buckets_no) {
		
		int idx = (key_idx + step) % new_buckets_no;

		
		for (int j = 0; j < BUCKET_SIZE; j++) {
			old_key = atomicCAS(&new_buckets[idx].entries[j].key, 0, key);
			if (old_key == 0) {
				atomicExch(&new_buckets[idx].entries[j].value, value);
				place_found = true;
				break;
			}
		}
		step++;
	}
}


GpuHashTable::GpuHashTable(int size) {
	int buckets_no = size / BUCKET_SIZE;
	if (size % BUCKET_SIZE)
		buckets_no++;

	
	cudaMalloc((void**)&(hashtable.buckets), buckets_no * sizeof(Bucket));
	cudaMemset(hashtable.buckets, 0, buckets_no * sizeof(Bucket));

	hashtable.size = buckets_no * BUCKET_SIZE;
	hashtable.occupied = 0;

}


GpuHashTable::~GpuHashTable() {
	cudaFree(hashtable.buckets);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	int new_buckets_no = numBucketsReshape / BUCKET_SIZE;
	if (numBucketsReshape % BUCKET_SIZE)
		new_buckets_no++;

	Bucket *new_buckets;
	cudaMalloc((void**)&new_buckets, new_buckets_no * sizeof(Bucket));
	cudaMemset(new_buckets, 0, new_buckets_no * sizeof(Bucket));

	
	if (just_created) {
		cudaFree(hashtable.buckets);
		hashtable.buckets = new_buckets;
		hashtable.size = new_buckets_no * BUCKET_SIZE;
		return;
	}

	int old_buckets_no = hashtable.size / BUCKET_SIZE;

	
	size_t blocks_no = hashtable.size / BLOCK_SIZE;
	if (hashtable.size % BLOCK_SIZE)
		blocks_no++;

	kernel_reshape>>(new_buckets, hashtable.buckets,
		new_buckets_no, old_buckets_no);
	cudaDeviceSynchronize();


	
	cudaFree(hashtable.buckets);
	hashtable.buckets = new_buckets;
	hashtable.size = new_buckets_no * BUCKET_SIZE;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	
	if (just_created) {
		int size = numKeys * (1 / LOAD_FACTOR);
		GpuHashTable::reshape(size);
		just_created = false;
	} else {
		int size = (hashtable.occupied + numKeys) * (1 / LOAD_FACTOR);
		GpuHashTable::reshape(size);
	}

	int *device_keys, *device_values, *device_occupied;
	int occupied = 0;

	
	cudaMalloc(&device_keys, numKeys * sizeof(int));
	cudaMalloc(&device_values, numKeys * sizeof(int));
	cudaMalloc(&device_occupied, sizeof(int));

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_occupied, &occupied, sizeof(int), cudaMemcpyHostToDevice);

	int buckets_no = hashtable.size / BUCKET_SIZE;

	size_t blocks_no = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE)
		blocks_no++;

	
	kernel_insert>>(hashtable.buckets, device_keys,
		device_values, numKeys, buckets_no, device_occupied);
	cudaDeviceSynchronize();

	
	cudaMemcpy(&occupied, device_occupied, sizeof(int), cudaMemcpyDeviceToHost);
	hashtable.occupied += occupied;

	cudaFree(device_keys);
	cudaFree(device_values);
	cudaFree(device_occupied);

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *values;
	int *device_values, *device_keys;

	values = (int *)malloc(numKeys * sizeof(int));

	cudaMalloc(&device_values, numKeys * sizeof(int));
	cudaMalloc(&device_keys, numKeys * sizeof(int));
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	int buckets_no = hashtable.size / BUCKET_SIZE;

	size_t blocks_no = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE)
		blocks_no++;

	
	kernel_get>>(hashtable.buckets, device_keys,
		device_values, numKeys, buckets_no);
	cudaDeviceSynchronize();

	
	cudaMemcpy(values, device_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(device_keys);
	cudaFree(device_values);

	return values;
}


float GpuHashTable::loadFactor() {
	if (hashtable.size == 0)
		return 1;

	return (float)hashtable.occupied / hashtable.size;
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

using namespace std;

#define	KEY_INVALID	0
#define LOAD_FACTOR 0.85
#define BLOCK_SIZE 1024
#define BUCKET_SIZE 4
#define SUCCESS 1
#define FAILED 0

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

typedef struct {
		int key;
		int value;
} Entry;

typedef struct {
	Entry entries[BUCKET_SIZE];
} Bucket;

typedef struct {
		Bucket *buckets;
		int size;
		int occupied;
} Hashtable;




class GpuHashTable
{
	Hashtable hashtable;
	bool just_created = true;

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
