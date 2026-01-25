
#include 
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"




__global__ void kernel_hmap_insert(int *keys, int* values, unsigned int input_size, GpuHashTable::skeyval *hmap_data, int HMAP_SIZE, int *updated) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= input_size)
		return;

	unsigned int mykey = keys[idx];
	unsigned int myval = values[idx];
	unsigned int idx_hmap = (((long)mykey * 13169977) % 5351951779) % HMAP_SIZE;
	GpuHashTable::skeyval *data;
	int cyc = 0;
	while (1) {
		if (idx_hmap == HMAP_SIZE) {
			idx_hmap = 0;
			cyc++;
			if (cyc == 2)
				break;
		}
		data = &(hmap_data[idx_hmap]);
		if (data->key == mykey) {
			data->val = myval;
			atomicAdd(updated, 1);
			break;
		}
		atomicCAS(&(data->key), 0, mykey);
		if (data->key == mykey) {
			data->val = myval;
			break;
		}
		idx_hmap++;
	}
}

__global__ void kernel_hmap_get(int *keys, int* values, unsigned int input_size, GpuHashTable::skeyval *hmap_data, int HMAP_SIZE) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= input_size)
		return;

	unsigned int mykey = keys[idx];
	unsigned int myval = 0;
	unsigned int idx_hmap = (((long)mykey * 13169977) % 5351951779) % HMAP_SIZE;
	GpuHashTable::skeyval *data;
	int cyc = 0;
	while (1) {
		if (idx_hmap == HMAP_SIZE) {
			idx_hmap = 0;
			cyc++;
			if (cyc == 2)
				break;
		}
		data = &(hmap_data[idx_hmap]);
		if (data->key == mykey) {
			myval = data->val;
			break;
		}
		idx_hmap++;
	}
	values[idx] = myval;
}

__global__ void kernel_hmap_reshape(GpuHashTable::skeyval *hmap_data_old, GpuHashTable::skeyval *hmap_data_new, int HMAP_SIZE_OLD, int HMAP_SIZE_NEW) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= HMAP_SIZE_OLD)
		return;

	GpuHashTable::skeyval *data_old = &(hmap_data_old[idx]);
	unsigned int mykey = data_old->key;
	if (mykey == 0)
		return;
	unsigned int myval = data_old->val;
	unsigned int idx_hmap = (((long)mykey * 13169977) % 5351951779) % HMAP_SIZE_NEW;
	GpuHashTable::skeyval *data;
	int cyc = 0;
	while (1) {
		if (idx_hmap == HMAP_SIZE_NEW) {
			idx_hmap = 0;
			cyc++;
			if (cyc == 2)
				break;
		}
		data = &(hmap_data_new[idx_hmap]);
		if (data->key == mykey) {
			data->val = myval;
			break;
		}
		atomicCAS(&(data->key), 0, mykey);
		if (data->key == mykey) {
			data->val = myval;
			break;
		}
		idx_hmap++;
	}
}


GpuHashTable::GpuHashTable(int size) {
	cudaMalloc((void **) &hmap_data, size * sizeof(skeyval));
	hsize = 0;
	crt_num = 0;
    if (hmap_data == 0) {
        printf("[HOST] Couldn't allocate memory\n");
		return;
    }
	hsize = size;
	cudaMemset(hmap_data, 0, hsize * sizeof(skeyval));
}


GpuHashTable::~GpuHashTable() {
	cudaFree(hmap_data);
}


void GpuHashTable::reshape(int numBucketsReshape) {
	unsigned int elems = numBucketsReshape; 
	unsigned int new_size = ceil(elems / targetLoadFactor); 
	skeyval *hmap_data_new;

	cudaMalloc((void **) &hmap_data_new, new_size * sizeof(skeyval));
	if (hmap_data_new == 0) {
        printf("[HOST] Couldn't allocate memory\n");
		return;
    }
	cudaMemset(hmap_data_new, 0, new_size * sizeof(skeyval));
	unsigned int num_blocks = hsize / num_threads;
	if (num_blocks % num_threads != 0)


		num_blocks++;

	kernel_hmap_reshape>> (hmap_data, hmap_data_new, hsize, new_size);
	cudaDeviceSynchronize();
	
	cudaFree(hmap_data);
	hmap_data = hmap_data_new;
	hsize = new_size;
}


bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	if (numKeys + crt_num > hsize)
		reshape(numKeys + crt_num);

	int *device_keys = 0;
	int *device_values = 0;
	int *updated;

	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	cudaMallocManaged(&updated, sizeof(int));

	if (device_keys == 0 || device_values == 0 || updated == 0) {
        printf("[HOST] Couldn't allocate GPU memory\n");
    	return false;
    }

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	*updated = 0;

	unsigned int num_blocks = numKeys / num_threads;


	if (num_blocks % num_threads != 0)
		num_blocks++;

	kernel_hmap_insert>> (device_keys, device_values, numKeys, hmap_data, hsize, updated);
	cudaDeviceSynchronize();

	crt_num += numKeys - *updated;


	if (loadFactor() < minLoadFactor)
		reshape(crt_num);
	
	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}


int* GpuHashTable::getBatch(int *keys, int numKeys) {
	int *device_keys = 0;
	int *device_values = 0;

	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	if (device_keys == 0 || device_values == 0) {
        printf("[HOST] Couldn't allocate GPU memory\n");
    	return NULL;
    }

	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	unsigned int num_blocks = numKeys / num_threads;
	if (num_blocks % num_threads != 0)
		num_blocks++;

	kernel_hmap_get>> (device_keys, device_values, numKeys, hmap_data, hsize);
	cudaDeviceSynchronize();

	int *values = (int *)malloc(numKeys * sizeof(int));
	if (values == 0) {
        printf("[HOST] Couldn't allocate CPU memory\n");
    	return NULL;
    }
	cudaMemcpy(values, device_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(device_keys);
	cudaFree(device_values);
	return values;
}


float GpuHashTable::loadFactor() {
	return (float)crt_num / hsize; 
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




class GpuHashTable
{
	
	
	public:
	struct skeyval {
		unsigned int key;
		unsigned int val;
	};

	private:
	skeyval *hmap_data;
	unsigned int hsize;
	unsigned int crt_num;
	unsigned int num_threads = 64;
	float minLoadFactor = 0.8f;
	float targetLoadFactor = 0.85f;

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

