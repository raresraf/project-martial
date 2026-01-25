
#include 
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define HASH_COMPUTE(k, a, b, l) ((((long)k * a) % b) % l)
#define HASH_A 13169977llu
#define HASH_B 5351951779llu
#define MAX_LOAD_FACTOR 0.9f
#define MIN_LOAD_FACTOR 0.8f




__global__ void reshape_table(struct Pair *old_table, struct Pair *new_table, int old_size, int new_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int old_key, new_key;
	int hash, i, count;

	if (idx > old_size || old_table[idx].key == KEY_INVALID) {
		return;
	}

	new_key = old_table[idx].key;
	hash = HASH_COMPUTE(new_key, HASH_A, HASH_B, new_size);
	for (i = hash, count = 0; count < new_size; i = (i + 1) % new_size, count++) {
		old_key = atomicCAS(&(new_table[i].key), KEY_INVALID, new_key);
		if (old_key == KEY_INVALID) {
			new_table[i].value = old_table[idx].value;
			return;
		}
	}
}


__global__ void insert_table(struct Pair *table, int *keys, int *values, int *num_updates, int num_pairs, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j, count;
	int old_key;
	int hash;

	if (idx >= num_pairs)
		return;

	hash = HASH_COMPUTE(keys[idx], HASH_A, HASH_B, size);

	for (i = 0; i < num_pairs; i++) {
		for (j = hash, count = 0; count < size; j = (j + 1) % size, count++) {
			old_key = atomicCAS(&(table[j].key), KEY_INVALID, keys[idx]);
			if (old_key == KEY_INVALID || old_key == keys[idx]) {
				table[j].value = values[idx];
				
				if (old_key != KEY_INVALID) {
					
					atomicAdd(&(num_updates[0]), 1);
				}
				return;
			}
		}
	}
}



__global__ void get_table(struct Pair *table, int *keys, int *values, int num_pairs, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j, count;
	int hash;

	if (idx >= num_pairs)
		return;

	hash = HASH_COMPUTE(keys[idx], HASH_A, HASH_B, size);

	for (i = 0; i < num_pairs; i++) {
		for (j = hash, count = 0; count < size; j = (j + 1) % size, count++) {
			if (table[j].key == keys[idx]) {
				values[idx] = table[j].value;
				return;
			}
		}
	}
}



GpuHashTable::GpuHashTable(int size) {
	hash_table = NULL;
	table_pairs = 0;
	table_size = size;
	cudaError_t rc;

	
	rc = cudaMalloc((void **) &hash_table, size * sizeof(struct Pair));
	if (rc != cudaSuccess) {
		cout << "Mem alloc error" << endl;
		return;
	}
	rc = cudaMemset(hash_table, 0, size * sizeof(struct Pair));
	if (rc != cudaSuccess) {
		cout << "Mem set error" << endl;
		return;
	}
}


GpuHashTable::~GpuHashTable() {
	
	cudaFree(hash_table);
	hash_table = NULL;
}


void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t rc;
	int num_blocks;

	if (hash_table == NULL) {
		cout << "Table not created" << endl;
		return;
	}

	struct Pair *new_hash_table = NULL;
	int new_size = numBucketsReshape;
	
	rc = cudaMalloc((void **) &new_hash_table, numBucketsReshape * sizeof(struct Pair));
	if (rc != cudaSuccess) {
		cout << "Reshape alloc error" << endl;
		return;
	}

	rc = cudaMemset(new_hash_table, 0, numBucketsReshape * sizeof(struct Pair));
	if (rc != cudaSuccess) {
		cout << "Reshape set error" << endl;
		return;
	}
	
	num_blocks = table_size/THREADS_PER_BLOCK;
	if (table_size % THREADS_PER_BLOCK != 0)


		num_blocks++;
	reshape_table>>(hash_table, new_hash_table, table_size, new_size);
	cudaDeviceSynchronize();

	
	cudaFree(hash_table);
	hash_table = new_hash_table;
	table_size = new_size;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *devKeys, *devValues = 0;
	int num_blocks = 0;
	int *num_updates = 0;
	int *host_updates = 0;

	if (hash_table == NULL) {
		cout << "Table not created" << endl;
		return false;
	}

	host_updates = (int *) malloc(sizeof(int));

	
	cudaMalloc((void **) &devKeys, numKeys * sizeof(int));
    cudaMalloc((void **) &devValues, numKeys * sizeof(int));
    cudaMalloc((void **) &num_updates, sizeof(int));
    if (devKeys == 0 || devValues == 0 || num_updates == 0) {
    	cout << "Insert alloc error" << endl;
    	return false;
    }

    cudaMemset(num_updates, 0, sizeof(int));

    cudaMemcpy(devKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);


    num_blocks = numKeys/THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK != 0)
		num_blocks++;

	if ((float)(numKeys + table_pairs)/table_size >= MAX_LOAD_FACTOR) {
		
		int new_size = (int)((numKeys + table_pairs)/MIN_LOAD_FACTOR);
		reshape(new_size);
	}

	
	num_blocks = numKeys/THREADS_PER_BLOCK;


	if (numKeys % THREADS_PER_BLOCK != 0)
		num_blocks++;
	insert_table>>(hash_table, devKeys, devValues, num_updates, numKeys, table_size);
	cudaDeviceSynchronize();

	cudaMemcpy(host_updates, num_updates, sizeof(int), cudaMemcpyDeviceToHost);
	
    table_pairs = table_pairs + numKeys - host_updates[0];
    

    if ((float)table_pairs/table_size < MIN_LOAD_FACTOR) {
    	
    	int new_size = (int)(table_pairs/MIN_LOAD_FACTOR);
		reshape(new_size);
    }
	
	
	cudaFree(devKeys);
	cudaFree(devValues);
	cudaFree(num_updates);
	free(host_updates);
	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *hostValues = 0;
	int *devValues = 0;
	int *devKeys = 0;
	int num_blocks;

	if (hash_table == NULL) {
		cout << "Table not created" << endl;
		return NULL;
	}
	
	hostValues = (int *) malloc(numKeys * sizeof(int));
	cudaMalloc((void **) &devKeys, numKeys * sizeof(int));
    cudaMalloc((void **) &devValues, numKeys * sizeof(int));
	if (hostValues == 0 || devValues == 0 || devKeys == 0) {
    	cout << "Get alloc error" << endl;
    	return NULL;
    }

    cudaMemcpy(devKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    num_blocks = numKeys/THREADS_PER_BLOCK;
    if (numKeys % THREADS_PER_BLOCK != 0)
    	num_blocks++;
    get_table>>(hash_table, devKeys, devValues, numKeys, table_size);
    cudaDeviceSynchronize();

    cudaMemcpy(hostValues, devValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

    
    cudaFree(devKeys);
    cudaFree(devValues);
	return hostValues;
}


float GpuHashTable::loadFactor() {
	if (hash_table == NULL) {
		cout << "Table not created" << endl;
		return 0.f;
	}
	if (table_size == 0) {
		cout << "Table is empty" << endl;
		return 0.f;
	}
	if (table_pairs > table_size) {
		cout << "Error: nr of pairs exceed size" << endl;
		return 0.f;
	}
	float load_factor = (float) table_pairs/table_size;
	return load_factor; 
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
#define THREADS_PER_BLOCK 1024

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

struct Pair {
	int key;
	int value;
};




class GpuHashTable
{
	struct Pair *hash_table;
	long table_size;
	long table_pairs;

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

