
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

__device__ unsigned int hash1(int data, int limit) {
	return ((long)data * 67965551447llu) % 441911656067171llu % limit;
}

__device__ unsigned int hash2(int data) {
	return  23llu - ((long)data * 16991387857llu) % 23llu;
}

__global__ void kernel_get(hash_table *gpu_hash, int *keys, int *values, int n_keys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int index1, index2, search;
	int i;

	if (idx < n_keys) {
		if(keys[idx] > 0) {
			index1 = hash1(keys[idx], gpu_hash->load.capacity);
			index2 = hash2(keys[idx]);
			i = 0;

			search = (index1 + i * index2) % gpu_hash->load.capacity;
			while (gpu_hash->table[search].key != keys[idx]) {
				if (gpu_hash->table[search].key == KEY_INVALID) {
					values[idx] = 0;
					return;
				}

				i++;
				search = (index1 + i * index2) % gpu_hash->load.capacity;
			}

			values[idx] = gpu_hash->table[search].value;
		} else {
			values[idx] = 0;
		}
	}
}



__global__ void kernel_reshape(hash_table *gpu_hash, pairkv *new_table, int limit) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	unsigned int index1, index2, search, old_key;
	pairkv to_add;

	if (idx load.capacity && gpu_hash->table[idx].key != KEY_INVALID) {
		to_add.key = gpu_hash->table[idx].key;
		to_add.value = gpu_hash->table[idx].value;

		index1 = hash1(to_add.key, limit);
		index2 = hash2(to_add.key);
		i = 0;

		search = (index1 + i * index2) % limit;
		old_key = atomicCAS(&new_table[search].key, KEY_INVALID, to_add.key);
		while (old_key != KEY_INVALID) {
			i++;
			search = (index1 + i * index2) % limit;
			old_key = atomicCAS(&new_table[search].key, KEY_INVALID, to_add.key);
		}

		new_table[search].value = to_add.value;
	}
}

__global__ void kernel_insert(hash_table *gpu_hash, int *keys, int *values, int n_keys) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	unsigned int index1, index2, search, old_key;

	if (idx  0 && values[idx] > 0) {
		index1 = hash1(keys[idx], gpu_hash->load.capacity);
		index2 = hash2(keys[idx]);
		i = 0;

		search = (index1 + i * index2) % gpu_hash->load.capacity;
		old_key = atomicCAS(&gpu_hash->table[search].key, KEY_INVALID, keys[idx]);

		while (old_key != KEY_INVALID) {
			if(old_key == keys[idx]) {
				gpu_hash->table[search].value = values[idx];
				return;
			}

			i++;
			search = (index1 + i * index2) % gpu_hash->load.capacity;
			old_key = atomicCAS(&gpu_hash->table[search].key, KEY_INVALID, keys[idx]);
		}

		gpu_hash->table[search].value = values[idx];
		atomicAdd(&gpu_hash->load.elements, 1);
	}
}

int find_prime(int n, int start)
{
	int i;

	for (i = start; i < PRIME_SIZE; i++) {
		if (primeList[i] >= n)
			return i;
	}

	return PRIME_SIZE - 1;
}


GpuHashTable::GpuHashTable(int size) {
	cudaMallocManaged(&gpu_hash, sizeof(hash_table));
	cudaMallocManaged(&gpu_hash->table, size * sizeof(pairkv));
	cudaMemset(gpu_hash->table, 0, size * sizeof(pairkv));

	gpu_hash->load.capacity = size;
	gpu_hash->load.elements = 0;

	prime_index = find_prime(size, 0);
}


GpuHashTable::~GpuHashTable() {
	cudaFree(gpu_hash->table);
	cudaFree(gpu_hash);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	pairkv *new_table;
	int new_size;
	size_t blocks_no = gpu_hash->load.capacity / BLOCK_SIZE;
 
	if (gpu_hash->load.capacity % BLOCK_SIZE)
		blocks_no++;

	prime_index = find_prime(numBucketsReshape, prime_index);
	new_size = primeList[prime_index];

	cudaMallocManaged(&new_table, new_size * sizeof(pairkv));
	cudaMemset(new_table, 0, new_size * sizeof(pairkv));



	kernel_reshape>>(gpu_hash, new_table, new_size);
	cudaDeviceSynchronize();

	cudaFree(gpu_hash->table);
	gpu_hash->table = new_table;
	gpu_hash->load.capacity = new_size;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys;
	int *device_values;
	int new_size;
	size_t blocks_no = numKeys / BLOCK_SIZE;
 
	if (numKeys % BLOCK_SIZE)
		blocks_no++;

	cudaMalloc((void **)&device_keys, numKeys * sizeof(int));
	cudaMalloc((void **)&device_values, numKeys * sizeof(int));

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	new_size = gpu_hash->load.elements + numKeys;
	if (new_size >= gpu_hash->load.capacity)


		reshape(new_size);

	kernel_insert>>(gpu_hash, device_keys, device_values, numKeys);
	cudaDeviceSynchronize();

	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *host_values;
	int *device_values;
	int *device_keys;
	size_t blocks_no = numKeys / BLOCK_SIZE;
 
	if (numKeys % BLOCK_SIZE)
		blocks_no++;

	host_values = (int *)malloc(numKeys * sizeof(int));
	cudaMalloc((void **)&device_values, numKeys * sizeof(int));
	cudaMalloc((void **)&device_keys, numKeys * sizeof(int));

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	kernel_get>>(gpu_hash, device_keys, device_values, numKeys);
	cudaDeviceSynchronize();

	cudaMemcpy(host_values, device_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(device_keys);
	cudaFree(device_values);

	return host_values;
}


float GpuHashTable::loadFactor() {
	return (float)gpu_hash->load.elements / gpu_hash->load.capacity; 
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

#define	KEY_INVALID 0
#define BLOCK_SIZE 256 
#define PRIME_SIZE 186

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






typedef struct pairkv {
	unsigned int key, value;
} pairkv;

typedef struct load_info {
	unsigned int capacity, elements;
} load_info;

typedef struct hash_table {
	pairkv *table;
	load_info load;
} hash_table;




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

	private:
		hash_table *gpu_hash;
		int prime_index;
};

#endif
