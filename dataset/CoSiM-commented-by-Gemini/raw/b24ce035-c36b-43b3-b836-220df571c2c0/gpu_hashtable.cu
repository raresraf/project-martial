
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


GpuHashTable::GpuHashTable(int size)
{
	cudaError_t error;

	
	error = cudaMalloc(&(GpuHashTable::hashtable), size * sizeof(hT));
	DIE(error != cudaSuccess || GpuHashTable::hashtable == NULL, "cudaMalloc hashtable error");
	error = cudaMemset(GpuHashTable::hashtable, 0, size * sizeof(hT));
	DIE(error != cudaSuccess, "cudaMemset hashtable error");

	
	GpuHashTable::currentTableSize = 0;
	GpuHashTable::tableSize = size;
}




GpuHashTable::~GpuHashTable()
{
	cudaError_t error;

	
	error = cudaFree(GpuHashTable::hashtable);
	DIE(error != cudaSuccess, "cudaFree hashtable error");

	
	GpuHashTable::currentTableSize = 0;
	GpuHashTable::tableSize = 0;
}



__global__ void copyForReshape(hT *hashtable, int tableSize,
								int *device_keys, int *device_values,
								int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if(idx < tableSize) {
		if (hashtable[idx].key != 0) {
			
			int index = atomicAdd(counter, 1);

			device_keys[index] = hashtable[idx].key;
			device_values[index] = hashtable[idx].value;
		}
	}
}




void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t error;
	hT *newHashTable = NULL;
	int new_size = 1.2f * numBucketsReshape;

	
	error = cudaMalloc(&newHashTable, new_size * sizeof(hT));
	DIE(error != cudaSuccess || newHashTable == NULL, "cudaMalloc new hashtable error");
	error = cudaMemset(newHashTable, 0, new_size * sizeof(hT));
	DIE(error != cudaSuccess, "cudaMemset new hashtable error");


	
	if(GpuHashTable::currentTableSize != 0) {
		const size_t block_size = 1024;
		size_t blocks_no = GpuHashTable::tableSize / block_size;
 
		if (GpuHashTable::tableSize % block_size) 
			++blocks_no;

		int *device_keys = NULL;
		int *counter = NULL;
		int *device_values = NULL;


		error = cudaMalloc(&device_keys, GpuHashTable::currentTableSize * sizeof(int));
		DIE(error != cudaSuccess || device_keys == NULL, "cudaMalloc device_keys error");

		error = cudaMalloc(&device_values, GpuHashTable::currentTableSize * sizeof(int));
		DIE(error != cudaSuccess || device_values == NULL, "cudaMalloc device_values error");

		
		error = cudaMalloc(&counter, sizeof(int));
		DIE(error != cudaSuccess || counter == NULL, "cudaMalloc counter error");
		error = cudaMemset(counter, 0, sizeof(int));
		DIE(error != cudaSuccess, "cudaMemset counter error");


		int *host_keys = (int *)malloc(GpuHashTable::currentTableSize * sizeof(int));
		DIE(host_keys == NULL, "malloc host_keys error");

		int *host_values = (int *)malloc(GpuHashTable::currentTableSize * sizeof(int));
		DIE(host_values == NULL, "malloc host_values error");

		
		
		
		copyForReshape>>(GpuHashTable::hashtable,
												GpuHashTable::tableSize,
												device_keys, device_values,
												counter);
		error = cudaDeviceSynchronize();
		DIE(error != cudaSuccess, "cudaDeviceSynchronize error");


		
		error = cudaMemcpy(host_keys, device_keys,
							GpuHashTable::currentTableSize * sizeof(int),
							cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_keys error");

		error = cudaMemcpy(host_values, device_values,
							GpuHashTable::currentTableSize * sizeof(int),
							cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_values error");


		
		GpuHashTable::~GpuHashTable();

		
		GpuHashTable::tableSize = new_size;
		GpuHashTable::hashtable = newHashTable;
		GpuHashTable::currentTableSize = 0;
		

		int numKeys = 0;
		error = cudaMemcpy(&numKeys, counter, sizeof(int), cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy numKeys error");
		
		
		insertBatch(host_keys, host_values, numKeys);


		error = cudaFree(device_keys);
		DIE(error != cudaSuccess, "cudaFree device_keys error");

		error = cudaFree(device_values);
		DIE(error != cudaSuccess, "cudaFree device_values error");

		error = cudaFree(counter);
		DIE(error != cudaSuccess, "cudaFree counter error");

		free(host_keys);
		free(host_values);

		return;

	}

	
	GpuHashTable::~GpuHashTable();

	
	GpuHashTable::tableSize = new_size;

	GpuHashTable::hashtable = newHashTable;
}






__global__ void getHashCode(int *keys, int *hashcodes, int numkeys, int tablesize)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numkeys)
		hashcodes[idx] = (((long)keys[idx]) * 1402982055436147llu)
								% 452517535812813007llu % tablesize;
}







__global__ void insertKeysandValues(int *hashcodes, int *keys, int *values,
									int numKeys, hT *hashtable,
									int *currentTableSize, int tablesize,
									int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numKeys) {
		if (keys[idx] == 0)
			return;

		int key = atomicCAS(&hashtable[hashcodes[idx]].key, 0, keys[idx]);
		
		if (key == 0) {
			hashtable[hashcodes[idx]].value = values[idx];
			
			keys[idx] = 0;
			atomicAdd(currentTableSize, 1);
			atomicAdd(counter, 1);
		} else if (key == keys[idx]) {
			
			hashtable[hashcodes[idx]].value = values[idx];
			atomicAdd(counter, 1);
			
			keys[idx] = 0;
		} else {
			
			
			hashcodes[idx] += 1;
			hashcodes[idx] %= tablesize;
		}
	}
}






bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys = NULL;
	int *device_values = NULL;
	cudaError_t error;


	
	
	if ((GpuHashTable::currentTableSize + numKeys) > GpuHashTable::tableSize)
		reshape((GpuHashTable::currentTableSize + numKeys));


	
	error = cudaMalloc(&device_keys, numKeys * sizeof(int));
	DIE(error != cudaSuccess || device_keys == NULL, "cudaMalloc device_keys error");

	error = cudaMalloc(&device_values, numKeys * sizeof(int));


	DIE(error != cudaSuccess || device_values == NULL, "cudaMalloc device_values error");


	
	error = cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(error != cudaSuccess, "cudaMemcpy keys error");

	error = cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(error != cudaSuccess, "cudaMemcpy values error");


	
	int *hashcodes = NULL;
	error = cudaMalloc(&hashcodes, numKeys * sizeof(int));
	DIE(error != cudaSuccess || hashcodes == NULL, "cudaMalloc hashcodes error");

	const size_t block_size = 1024;
	size_t blocks_no = numKeys / block_size;
 
	if (numKeys % block_size) 
		++blocks_no;

	getHashCode>>(device_keys, hashcodes,
											numKeys, GpuHashTable::tableSize);
	error = cudaDeviceSynchronize();
	DIE(error != cudaSuccess, "cudaDeviceSynchronize error");



	
	
	int *device_current = NULL;


	error = cudaMalloc(&device_current, sizeof(int));
	DIE(error != cudaSuccess, "cudaMalloc device_current error");
	error = cudaMemset(device_current, 0, sizeof(int));
	DIE(error != cudaSuccess, "cudaMemset device_current error");



	
	
	int *device_counter = NULL;
	error = cudaMalloc(&device_counter, sizeof(int));
	DIE(error != cudaSuccess || device_counter == NULL, "cudaMalloc device_counter error");
	error = cudaMemset(device_counter, 0, sizeof(int));
	DIE(error != cudaSuccess, "cudaMemset device_counter error");


	int host_counter = 0;
	int old_counter = 0;
	while(1) {
		
		insertKeysandValues>>(hashcodes, device_keys,
													device_values, numKeys,
													GpuHashTable::hashtable,
													device_current,
													GpuHashTable::tableSize,
													device_counter);
		error = cudaDeviceSynchronize();
		DIE(error != cudaSuccess, "cudaDeviceSynchronize error");

		old_counter = host_counter;
		error = cudaMemcpy(&host_counter, device_counter, sizeof(int), cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_counter error");

		
		GpuHashTable::currentTableSize += host_counter - old_counter;

		if(host_counter == numKeys)
			break;
	}
	

	
	error = cudaFree(device_keys);
	DIE(error != cudaSuccess, "cudaFree device_keys error");

	error = cudaFree(device_values);
	DIE(error != cudaSuccess, "cudaFree device_values error");

	error = cudaFree(hashcodes);
	DIE(error != cudaSuccess, "cudaFree hashcodes error");

	error = cudaFree(device_current);
	DIE(error != cudaSuccess, "cudaFree device_current error");

	error = cudaFree(device_counter);
	DIE(error != cudaSuccess, "cudaFree device_counter error");

	return true;
}





__global__ void getbatch(int *values, int *hashcodes, int *keys, int numKeys,
						hT *hashtable, int tablesize, int *counter)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numKeys) {
		if (keys[idx] == 0)
			return;

		if(hashtable[hashcodes[idx]].key == keys[idx]) {
			
			values[idx] = hashtable[hashcodes[idx]].value;
			atomicAdd(counter, 1);
			keys[idx] = 0;
		} else {
			
			hashcodes[idx] += 1;
			hashcodes[idx] %= tablesize;
		}
	}
}





int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *ret_values = (int *)malloc(numKeys * sizeof(int));
	cudaError_t error;

	
	int *device_ret_values = NULL;
	error = cudaMalloc(&device_ret_values, numKeys * sizeof(int));
	DIE(error != cudaSuccess || device_ret_values == NULL, "cudaMalloc device_ret_values error");

	
	int *device_keys = NULL;
	error = cudaMalloc(&device_keys, numKeys * sizeof(int));
	DIE(error != cudaSuccess || device_keys == NULL, "cudaMalloc device_keys error");
	error = cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(error != cudaSuccess, "cudaMemcpy keys error");


	
	int *hashcodes = NULL;

	error = cudaMalloc(&hashcodes, numKeys * sizeof(int));
	DIE(error != cudaSuccess || hashcodes == NULL, "cudaMalloc hashcodes error");

	const size_t block_size = 1024;
	size_t blocks_no = numKeys / block_size;
 
	if (numKeys % block_size) 
		++blocks_no;

	getHashCode>>(device_keys, hashcodes, numKeys, GpuHashTable::tableSize);
	error = cudaDeviceSynchronize();
	DIE(error != cudaSuccess, "cudaDeviceSynchronize error");



	int *device_counter = NULL;
	error = cudaMalloc(&device_counter, sizeof(int));
	DIE(error != cudaSuccess || device_counter == NULL, "cudaMalloc device_counter error");
	error = cudaMemset(device_counter, 0, sizeof(int));
	DIE(error != cudaSuccess, "cudaMemset device_counter error");

	while(1) {
		getbatch>>(device_ret_values, hashcodes,
											device_keys, numKeys,
											GpuHashTable::hashtable,
											GpuHashTable::tableSize, device_counter);
		error = cudaDeviceSynchronize();
		DIE(error != cudaSuccess, "cudaDeviceSynchronize error");

		int host_counter = 0;
		error = cudaMemcpy(&host_counter, device_counter, sizeof(int), cudaMemcpyDeviceToHost);
		DIE(error != cudaSuccess, "cudaMemcpy host_counter error");

		if(host_counter == numKeys)
			break;
	}


	


	error = cudaMemcpy(ret_values, device_ret_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	DIE(error != cudaSuccess, "cudaMemcpy ret_values error");


	
	error = cudaFree(device_keys);
	DIE(error != cudaSuccess, "cudaFree device_keys error");

	error = cudaFree(device_ret_values);
	DIE(error != cudaSuccess, "cudaFree device_ret_values error");

	error = cudaFree(hashcodes);
	DIE(error != cudaSuccess, "cudaFree hashcodes error");

	error = cudaFree(device_counter);
	DIE(error != cudaSuccess, "cudaFree device_counter error");

	return ret_values;
}




float GpuHashTable::loadFactor() {
	return ((float)GpuHashTable::currentTableSize) / ((float)GpuHashTable::tableSize);
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

typedef struct hashtableCell{
	int key;
	int value;
} hT;




class GpuHashTable
{
	public:
		hT *hashtable;
		int tableSize;
		int currentTableSize;
		

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

