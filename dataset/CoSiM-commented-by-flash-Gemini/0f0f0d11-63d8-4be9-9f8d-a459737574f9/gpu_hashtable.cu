
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"
#define LOAD_FACTOR 0.8f


__global__ void insertHash(unsigned long long *hashmap, int capacity, int *keys, int *values, int numKeys, unsigned int *num_inserted) {

	unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;

	if(id < numKeys)
	{	
		if(keys[id] <= 0 || values[id] <= 0)
		{
			return;
		}
		


		unsigned long long new_kv;
		
		new_kv = keys[id];
		new_kv = new_kv << 32;
		new_kv += values[id];

		int index = hash1(keys[id], capacity);
		
		
		while(true)
		{
			


			atomicCAS(&hashmap[index], (unsigned long long) 0, new_kv);
			
			if(hashmap[index] != new_kv)
			{
				unsigned long long k = hashmap[index] >> 32;

				if(k == new_kv >> 32)
				{
					atomicCAS(&hashmap[index], hashmap[index], new_kv);
					return;
				}
				index = (index + 1) % capacity;
			} else 
			{
				
				atomicInc(num_inserted, 4294967295);
				return;
			}			

		}			
	}	
}

__global__ void getHash(unsigned long long *hash, int capacity, int *keys, int *values, int numKeys) {
	
	unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
	if(id < numKeys)
	{
		int index = hash1(keys[id], capacity);
	
		int start = index;
			
		while(true)
		{
			
			if(hash[index] >> 32 == keys[id])
			{
				unsigned long long temp = hash[index];
				temp = temp << 32;
				temp = temp >> 32;
				values[id] =(int) temp;
				return;
			
			} else if( hash[index] >> 32 == 0)
			{
				values[id] = 0;
				return;
			} else 
			{
				index = (index + 1) % capacity;
				if(start == index)
				{
					values[id] = 0;
					return;
				}	
			}
						
			
		}
	}
	
}
 

GpuHashTable::GpuHashTable(int size) {
	cudaMalloc((void**) &hashmap, size * sizeof(unsigned long long));
	if(hashmap == NULL)
	{
		printf("Allocation failed[hashmap]\n");
		return;
	}


	cudaMemset(hashmap, 0x00, size * sizeof(unsigned long long));
	capacity = size;
	occupied = 0; 
}


GpuHashTable::~GpuHashTable() {
	cudaFree(hashmap);
	hashmap = NULL;
	capacity = 0;
	occupied = 0;
}


void GpuHashTable::reshape(int numBucketsReshape) {
	
	unsigned long long *tempHash;
	int threads_num_block = 1024;
	int num_blocks = capacity / threads_num_block + 1;

	if(cudaMalloc((void**) &tempHash, numBucketsReshape * sizeof(unsigned long long)) != cudaSuccess)
	{
		printf("Allocation failed [tempHash]\n");
		return;
	}
	cudaMemset(tempHash, 0x00, numBucketsReshape * sizeof(unsigned long long));
	
	if(occupied == 0)
	{
		cudaFree(hashmap);
		hashmap = tempHash;
		capacity = numBucketsReshape;
		return;
	}
	int num_el = 0;
	int *k_h = (int *) malloc (occupied * sizeof(int));
	int *v_h = (int *) malloc (occupied * sizeof(int));
	unsigned long long *hashmap_h = (unsigned long long *) malloc ( capacity * sizeof(unsigned long long));
	cudaMemcpy(hashmap_h, hashmap, capacity * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	if( k_h == NULL || v_h == NULL || hashmap_h == NULL)
	{
		printf("Allocation failed [k_h v_h hashmap_h]\n");
		return;
	}
	
	for(int i = 0; i < capacity; i++)
	{


		unsigned long long temp = hashmap_h[i];
		if(temp >> 32 != 0)
		{
			k_h[num_el] = temp >> 32;
			
			temp = temp << 32;
			temp = temp >> 32;

			v_h[num_el] = temp;
			num_el++;
		}
	}
	
	int *k_d;
	int *v_d;


	if(cudaMalloc((void **) &k_d, num_el * sizeof(int)) != cudaSuccess)
	{
		printf("Allocation failed\n");
	}
	if(cudaMalloc((void **) &v_d, num_el * sizeof(int)) != cudaSuccess)
	{
		printf("Allocation failed\n");
	} 
	
	cudaMemcpy(k_d, k_h, num_el * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(v_d, v_h, num_el * sizeof(int), cudaMemcpyHostToDevice);
	unsigned int *num_inserted;


	cudaMalloc((void **) &num_inserted, sizeof(unsigned int));
	cudaMemset(num_inserted, 0, sizeof(unsigned int));
	
	insertHash>>(tempHash, numBucketsReshape, k_d, v_d, num_el, num_inserted);

	cudaDeviceSynchronize();
		

	capacity = numBucketsReshape;

	
	cudaFree(k_d);
	cudaFree(v_d);
	cudaFree(hashmap);
	cudaFree(num_inserted);
	free(k_h);
	free(v_h);
	hashmap = tempHash;
	
	tempHash = NULL;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
			
	int *k_d;
	int *v_d;
	int acceptable_kv = 0;
	int threads_num_block = 1024;	
	int num_blocks = numKeys / threads_num_block + 1;

	cudaMalloc((void**) &k_d, numKeys * sizeof(int));
	if(k_d == NULL)
	{
		printf("Allocation failed[key]\n");
		return false;
	}

	cudaMalloc((void**) &v_d, numKeys * sizeof(int));
	if(v_d == NULL)
	{


		printf("Allocation failed[value]\n");
		return false;
	}	
	cudaMemcpy(k_d, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(v_d, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);	
	
	for(int i = 0; i < numKeys; i++)
	{
		if(keys[i] > 0 && values[i] > 0)
		{
			acceptable_kv++;		
		}
	}
	
	if( (acceptable_kv + occupied) / capacity > LOAD_FACTOR) {
		int new_capacity = (acceptable_kv + occupied) / LOAD_FACTOR;
		reshape((int) 1.5 * new_capacity);
	}
	unsigned int *num_inserted_d;
	unsigned int *num_inserted_h = (unsigned int *) malloc (sizeof(unsigned int));
	(*num_inserted_h) = 0;



	cudaMalloc((void **) &num_inserted_d, sizeof(unsigned int));
	cudaMemcpy(num_inserted_d, num_inserted_h, sizeof(unsigned int), cudaMemcpyHostToDevice);
		
	
	insertHash>>(hashmap, capacity, k_d, v_d, numKeys, num_inserted_d);
	
	cudaDeviceSynchronize();

	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		fprintf(stderr, "ERR: %s \n", cudaGetErrorString(error));
	}
	cudaMemcpy(num_inserted_h, num_inserted_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);	
	occupied += (*num_inserted_h);
	
	cudaFree(k_d);
	cudaFree(v_d);
	cudaFree(num_inserted_d);
	free(num_inserted_h);
	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *k_d;


	int *v_d;
	int threads_num_block = 1024;
	int num_blocks = numKeys / threads_num_block + 1;
	if(cudaMalloc((void **) &k_d, numKeys * sizeof(int)) != cudaSuccess)
	{
		printf("Allocation error\n");
		return NULL;
	}
	
	if(cudaMalloc((void **) &v_d, numKeys * sizeof(int)) != cudaSuccess)
	{
		printf("Allocation error\n");
		return NULL;
	}
	cudaMemcpy(k_d, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	getHash>>(hashmap,capacity,k_d, v_d, numKeys);

	int *v_h;
	v_h = (int *) malloc (numKeys * sizeof(int));
	cudaMemcpy(v_h, v_d, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(k_d);
	cudaFree(v_d);
	
	return v_h;
}


float GpuHashTable::loadFactor() {
	return (1.0f * occupied) / capacity; 
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
	
__device__ const size_t primeList[] =
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
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);	
		float loadFactor();
		void occupancy();
		void print(string info);
		unsigned long long *hashmap;
		int occupied;
		int capacity;
		~GpuHashTable();
};

#endif

