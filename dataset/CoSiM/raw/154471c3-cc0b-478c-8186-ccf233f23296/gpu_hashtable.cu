
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"



static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in "
            << file << " at line " << line << endl;
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__
void gpu_reshape(int n, int nn, int *x, int *y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;

	
	for (int i = 0; i < SLOTS; i++) {
		int key = x[2 * SLOTS * idx + 2 * i];
		if (key == KEY_INVALID)
			continue;

		int value = x[2 * SLOTS * idx + 2 * i + 1];

		int pos[HASHES];
		pos[0] = hash1(key, nn);
		pos[1] = hash2(key, nn);
		pos[2] = hash3(key, nn);

		int tmp;
		
		for (int j = 0; j < HASHES; j++) {
			
			for (int k = 0; k < SLOTS; k++) {
				tmp = atomicCAS(y + 2 * SLOTS * pos[j] + 2 * k, KEY_INVALID, key);

				
				if (tmp == KEY_INVALID) {
					y[2 * SLOTS * pos[j] + 2 * k + 1] = value;
					goto next_key;
				}
			}
		}

		
		int j = pos[HASHES - 1];
		do {
			
			for (int k = 0; k < SLOTS; k++) {
				tmp = atomicCAS(y + 2 * SLOTS * j + 2 * k, KEY_INVALID, key);

				
				if (tmp == KEY_INVALID) {
					y[2 * SLOTS * j + 2 * k + 1] = value;
					goto next_key;
				}
			}

			j = (j + 1) % nn;
		} while (j != pos[HASHES - 1]);

next_key:
		continue;
	}

}

__global__
void gpu_insert(int *e, int nn, int n, int *k, int *v, int *ins) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;

	int key = k[idx];
	int value = v[idx];

	int pos[HASHES];
	pos[0] = hash1(key, nn);
	pos[1] = hash2(key, nn);
	pos[2] = hash3(key, nn);

	int tmp;

	
	for (int i = 0; i < HASHES; i++) {
		
		for (int j = 0; j < SLOTS; j++) {
			tmp = atomicCAS(&e[2 * SLOTS * pos[i] + 2 * j], KEY_INVALID, key);

			if (tmp == KEY_INVALID || tmp == key) {
				e[2 * SLOTS * pos[i] + 2 * j + 1] = value;

				
				if (tmp != key)
					atomicAdd(ins, 1);

				return;
			}
		}
	}

	
	int i = pos[HASHES - 1] + 1;
	do {
		
		for (int j = 0; j < SLOTS; j++) {
			tmp = atomicCAS(&e[2 * SLOTS * i + 2 * j], KEY_INVALID, key);

			
			if (tmp == KEY_INVALID) {
				e[2 * SLOTS * i + 2 * j + 1] = value;

				
				if (tmp != key)
					atomicAdd(ins, 1);


				return;
			}
		}

		i = (i + 1) % nn;
	} while (i != pos[HASHES - 1]);
}

__global__
void gpu_get(int *e, int nn, int n, int *k, int *v) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;

	int pos2[HASHES];
	pos2[0] = 0; pos2[1] = 0; pos2[2] = 0;
	pos2[0] = hash1(k[idx], nn);
	pos2[1] = hash2(k[idx], nn);
	pos2[2] = hash3(k[idx], nn);

	
	for (int i = 0; i < HASHES; i++) {
		
		for (int j = 0; j < SLOTS; j++) {
			if (e[2 * SLOTS * pos2[i] + 2 * j] == k[idx]) {
				v[idx] = e[2 * SLOTS * pos2[i] + 2 * j + 1];
				return;
			}
		}
	}

	
	int i = pos2[HASHES - 1] + 1;
	do {
		
		for (int j = 0; j < SLOTS; j++) {
			if (e[2 * SLOTS * i + 2 * j] == k[idx]) {
				v[idx] = e[2 * SLOTS * i + 2 * j + 1];
				return;
			}
		}

		i = (i + 1) % nn;
	} while (i != pos2[HASHES - 1]);
}


GpuHashTable::GpuHashTable(int size) {
	this->size = size;
	this->used = 0;
	this->entries = NULL;

	int num_bytes = 2 * SLOTS * size * sizeof(int);

	cudaMalloc((void **)&(this->entries), num_bytes);
	HANDLE_ERROR(cudaGetLastError());

	cudaMemset(this->entries, 0, num_bytes);
	HANDLE_ERROR(cudaGetLastError());
}


GpuHashTable::~GpuHashTable() {
	cudaFree(this->entries);
	HANDLE_ERROR(cudaGetLastError());

	cudaDeviceReset();
}


void GpuHashTable::reshape(int numBucketsReshape) {
	int *new_entries = NULL;
	int num_bytes = numBucketsReshape * 2 * SLOTS * sizeof(int);

	cudaMalloc((void **)&new_entries, num_bytes);
	HANDLE_ERROR(cudaGetLastError());

	cudaMemset(new_entries, 0, num_bytes);
	HANDLE_ERROR(cudaGetLastError());

	int blocks_no = this->size / BLOCK_SIZE;
	if (this->size % BLOCK_SIZE != 0)
		blocks_no++;

	if (this->used != 0)
		gpu_reshape>>(this->size, numBucketsReshape, this->entries, new_entries);
	HANDLE_ERROR(cudaGetLastError());

	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaGetLastError());

	this->size = numBucketsReshape;

	cudaFree(this->entries);
	HANDLE_ERROR(cudaGetLastError());

	this->entries = new_entries;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	
	if (1.0f * (numKeys + this->used) / this->size > MAX_LOAD)
		this->reshape((int)((numKeys + this->used) / MIN_LOAD));

	int num_bytes = numKeys * sizeof(int);
	int *device_keys = NULL;
	int *device_values = NULL;
	int *device_inserted = NULL;
	int *host_inserted = 0;

	host_inserted =(int *) malloc(sizeof(int));
	*host_inserted = 0;

	cudaMalloc((void **)&device_keys, num_bytes);
	HANDLE_ERROR(cudaGetLastError());

	cudaMalloc((void **)&device_values, num_bytes);
	HANDLE_ERROR(cudaGetLastError());

	cudaMalloc((void **)&device_inserted, sizeof(int));
	HANDLE_ERROR(cudaGetLastError());

	DIE(device_keys == NULL || device_values == NULL || device_inserted == NULL, "aaa cuda random fail");

	cudaMemcpy(device_keys, keys, num_bytes, cudaMemcpyHostToDevice);
	HANDLE_ERROR(cudaGetLastError());

	cudaMemcpy(device_values, values, num_bytes, cudaMemcpyHostToDevice);
	HANDLE_ERROR(cudaGetLastError());

	cudaMemcpy(device_inserted, host_inserted, sizeof(int), cudaMemcpyHostToDevice);
	HANDLE_ERROR(cudaGetLastError());

	int blocks_no = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE != 0)
		blocks_no++;



	gpu_insert>>(this->entries, this->size, numKeys, device_keys, device_values, device_inserted);
	HANDLE_ERROR(cudaGetLastError());

	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaGetLastError());

	cudaMemcpy(host_inserted, device_inserted, sizeof(int), cudaMemcpyDeviceToHost);
	HANDLE_ERROR(cudaGetLastError());

	this->used += *host_inserted;

	cudaFree(device_keys);
	HANDLE_ERROR(cudaGetLastError());

	cudaFree(device_values);
	HANDLE_ERROR(cudaGetLastError());

	cudaFree(device_inserted);
	HANDLE_ERROR(cudaGetLastError());

	
	if (this->loadFactor() < MIN_LOAD)
		this->reshape((int)(this->size * this->loadFactor() / MIN_LOAD));

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *host_result = NULL;
	int *device_result = NULL;
	int *device_keys = NULL;

	int num_bytes = numKeys * sizeof(int);

	cudaMalloc((void **)&device_result, num_bytes);
	cudaMalloc((void **)&device_keys, num_bytes);
	cudaMemcpy(device_keys, keys, num_bytes, cudaMemcpyHostToDevice);
	HANDLE_ERROR(cudaGetLastError());

	cudaMemset(device_result, 0, num_bytes);
	HANDLE_ERROR(cudaGetLastError());

	host_result = (int *) malloc(num_bytes);
	DIE(host_result == NULL, "brah");

	int blocks_no = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE != 0)
		blocks_no++;



	gpu_get>>(this->entries, this->size, numKeys, device_keys, device_result);
	HANDLE_ERROR(cudaGetLastError());

	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaGetLastError());

	cudaMemcpy(host_result, device_result, num_bytes, cudaMemcpyDeviceToHost);
	HANDLE_ERROR(cudaGetLastError());

	cudaFree(device_result);
	HANDLE_ERROR(cudaGetLastError());

	return host_result;
}


float GpuHashTable::loadFactor() {
	return 1.0f * this->used / this->size ; 
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
#define SLOTS			2
#define HASHES			3
#define MIN_LOAD		.8f
#define MAX_LOAD		.95f
#define BLOCK_SIZE		256 


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

__device__ int hash1(int data, int limit) {
	return ((long)abs(data) * 151llu) % 3452434812973llu % limit;
}

__device__ int hash2(int data, int limit) {
	return ((long)abs(data) * 797llu) % 8699590588571llu % limit;
}

__device__ int hash3(int data, int limit) {
	return ((long)abs(data) * 4027llu) % 21921594616111llu % limit;
}





class GpuHashTable
{
	public:
		int size;
		int used;

		int *entries;

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

