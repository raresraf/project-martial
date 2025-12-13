
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include "gpu_hashtable.hpp"



__device__
int hash_func(int data, int limit) {
	const size_t val_1 = 26339969llu;
	const size_t val_2 = 6743036717llu;
	return ((long long)abs(data) * val_1) % val_2 % limit;
}



__global__ 
void insert(int *keys, int *values, int numKeys, HashTable *table, int table_dim, int insertedElements)
{
	unsigned int thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
	int inserted = 0;
	int old;
	int key;
	int val;
	int pos;

	key   = keys[thread_idx];
	val   = values[thread_idx];
	pos   = hash_func(key, table_dim);
	old   = atomicCAS(&table[pos].key, -1, key);
	
	if (thread_idx >= insertedElements) {
		return;
	} else {

		if (old == -1 || old == key) {
			atomicExch(&table[pos].value, val);
			return;
		} else {
			while (!inserted) {
				pos += 1;
				if (pos == table_dim)
					pos = 0;
				old = atomicCAS(&table[pos].key, -1, key);
				if (old == -1 || old == key) {
					atomicExch(&table[pos].value, val);
					inserted = 1;
				}
			} 
		}
		
	}
}


__global__ void initialize (HashTable *table, int table_dim)
{
	unsigned int thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (thread_idx < table_dim) {
		table[thread_idx].key   = -1;
		table[thread_idx].value = -1;
	}
	return;
}


GpuHashTable::GpuHashTable(int size) {
	
	int num_blocks = size / 512;
	if (size % 512 != 0)
		num_blocks += 1;
	
	insertedElements = 0;
	table_dim = size;
	firstTime = 0;
	cnt = 0;
	cudaMalloc((void **)&table, sizeof(HashTable)*size);
	
	initialize>>(table, table_dim);
}




GpuHashTable::~GpuHashTable() {
	cudaFree(table);
}



__global__ void reshape_table(HashTable *new_table, HashTable *table, int table_dim, int insertedElements)
{
	unsigned int thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
 	int inserted = 0;	
	int old;
	int key;
	int val;
	int pos;

		
	if (thread_idx >= insertedElements) {
		return;
	} else {
		key = table[thread_idx].key;
		val  = table[thread_idx].value;
		pos  = hash_func(key, table_dim);
		
		if (key == -1 || key == 0)
			return;		

        old = atomicCAS(&new_table[pos].key, -1, key);

		if (old == -1 || old == key) {
			atomicExch(&new_table[pos].value, val);
			return;
		} else {
			while (!inserted) {
				pos += 1;
				if (pos == table_dim)
					pos = 0;
					old = atomicCAS(&new_table[pos].key, -1, key);
					if (old == -1 || old == key) {
						atomicExch(&new_table[pos].value, val);
						inserted = 1;
					}
				}
			}

    }
	
}


void GpuHashTable::reshape(int numBucketsReshape)
{
	int new_dim = numBucketsReshape;
	int num_blocks_init = (new_dim) / 512; 
	
	if ((new_dim) % 512 != 0)
		num_blocks_init += 1; 	
	
	table_dim = new_dim;

	cudaFree(table);
	cudaMalloc(&table, sizeof(HashTable) * new_dim);
	initialize>>(table, new_dim);
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	
	int new_dim;
	int *device_k;
	int *device_v;
	HashTable *new_table_device;
	int num_blocks = (numKeys) / 512;
	
	if (numKeys % 512 != 0)
		num_blocks += 1;



	cnt = numKeys;
	cudaMalloc((void **) &device_k, numKeys * sizeof(int));
    cudaMalloc((void **) &device_v, numKeys * sizeof(int));
	
	
	if (device_v == 0 || device_k == 0) {
		printf("[HOST] Couldn't allocate memory_1\n");
        return false;
    }
	
	cudaMemcpy(device_k, keys,   numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
		
	if (firstTime == 0) {
		new_dim = (numKeys / table_dim) * numKeys + 0.2 * numKeys;
		int num_blocks_init = new_dim / 512;
		if (new_dim % 512 != 0)
			num_blocks_init += 1;
		table_dim = new_dim;
		cudaFree(table);
		cudaMalloc((void **) &table, sizeof(HashTable) * new_dim);
		initialize>>(table, new_dim);
        insertedElements += numKeys;
	} else
	
	if (((table_dim - insertedElements) < numKeys)) {
		int old_dim = table_dim;
		table_dim = table_dim + numKeys;
        cudaMalloc((void **) &new_table_device, sizeof(HashTable) * table_dim);
        int num_blocks_init = (table_dim) / 512;
		if (table_dim % 512 != 0)
			num_blocks_init += 1;
       		
        initialize>>(new_table_device, table_dim);
        cudaDeviceSynchronize();
               	
		
		reshape_table>> (new_table_device, table, table_dim, old_dim);
        cudaFree(table);
		cudaDeviceSynchronize();
		cudaMalloc((void **) &table, sizeof(HashTable) * table_dim);
		cudaMemcpy(table, new_table_device, table_dim * sizeof(HashTable), cudaMemcpyDeviceToDevice);
		cudaFree(new_table_device);
		insertedElements += numKeys;
        }
	
	insert>> (device_k, device_v, numKeys, table, table_dim, cnt);
	cudaDeviceSynchronize();
    firstTime += 1;
	
	
	cudaFree(device_k);
	cudaFree(device_v);
	
	return true;
	

}





__global__ void get_values(HashTable *table,int *vec, int *keys, int table_dim, int numKeys, int insertedElements) {
	
	int inserted = 0;
	unsigned int thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
	int key;
	int pos;
	key   = keys[thread_idx];
    pos   = hash_func(key, table_dim);
	
	if (thread_idx >= numKeys) {
		return;
	} else {        	
		if (table[pos].key == key) {
			atomicExch(&vec[thread_idx], table[pos].value);
			return;
		} else {
			while (!inserted) {
               	pos += 1;
				if (pos == table_dim)
					pos = 0;
				if (table[pos].key == key) {
					atomicExch(&vec[thread_idx], table[pos].value);
					inserted = 1;
                }
            }
        }
    }
}

int* GpuHashTable::getBatch(int* keys, int numKeys) {
	
	int *vec;
	int *device_keys;
	int *local_vec;
	int num_blocks = (numKeys) / 512;
	
	if (numKeys % 512 != 0)
		num_blocks += 1;



	local_vec = (int *)malloc(numKeys * sizeof(int));
	cudaMalloc((void **) &vec, numKeys * sizeof(int));
	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	
	
	if (vec == 0 || device_keys == 0) {
		printf("[HOST] Couldn't allocate memory_2\n");
		return NULL;
	}
	cudaMemcpy(device_keys, keys,  numKeys * sizeof(int), cudaMemcpyHostToDevice);
	get_values>> (table, vec, device_keys, table_dim, numKeys, insertedElements);
	
	cudaDeviceSynchronize();
	
	cudaMemcpy(local_vec, vec, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(vec);
	cudaFree(device_keys);
	return local_vec;
}


float GpuHashTable::loadFactor() {
	return (float)insertedElements / table_dim; 
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





struct HashTable {
	int value;
	int key;
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
	int table_dim;
	int insertedElements;
	int firstTime;
	int cnt;	
	HashTable *table;
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

