
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

int nb_threads = 64;

__global__ void insert_element(int *keys, int *values, int n, struct Nod *hashmap, int size_hash, int *new_inserted)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx >= n)
        return;
    int index = ((long)abs(keys[idx]) * 13169977) % 5351951779 % size_hash;
    unsigned int val = values[idx];
    unsigned int key = keys[idx];
    int old;
    while(1)
    {
        if(hashmap[index].key == key)
        {
            hashmap[index].value = val;
            atomicAdd(new_inserted, 1);
            break;
        }
        old = atomicCAS(&hashmap[index].key, 0, key);
        if (old == 0)
        {
            hashmap[index].value = val;
            break;
        }
        index++;
        if (index == size_hash)
            index = 0;
    }  
}

__global__ void get_elements(int *keys, int *values, int n, struct Nod *hashmap, int size_hash)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    int index = ((long)abs(keys[idx]) * 13169977) % 5351951779 % size_hash;
    while(1)
    {
        if(hashmap[index].key == keys[idx])
        {
            values[idx] = hashmap[index].value;
            break;
        }
        index++;
        if (index == size_hash)
            index = 0;
    }
}

__global__ void reshapeHashmap(struct Nod *oldHashmap, struct Nod *newHashmap, int old_n, int new_n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= old_n || oldHashmap[idx].key == 0)
        return;
    int index = ((long)abs((int)oldHashmap[idx].key) * 13169977) % 5351951779 % new_n;
    int old;
    while(1)
    {
        old = atomicCAS(&newHashmap[index].key, 0, oldHashmap[idx].key);
        if (old == 0)
        {
            newHashmap[index].value = oldHashmap[idx].value;
            break;
        }
        index++;
        if (index == new_n)
            index = 0;
    }
}


GpuHashTable::GpuHashTable(int size2) {
    size = size2;
    batch_size = 10; 
    used_size = 0;
    cudaMalloc((void **) &hashmap, size * sizeof(struct Nod));
    cudaMalloc((void **) &new_inserted, 1 * sizeof(int));
    cudaMemset(hashmap, 0, size * sizeof(struct Nod));
    cudaMalloc((void **) &keys_to_insert, batch_size * sizeof(int));
    cudaMalloc((void **) &values_to_insert, batch_size * sizeof(int));
    if(hashmap == 0 || keys_to_insert == 0 || values_to_insert == 0 || new_inserted == 0)
    {
        printf("problema la alocare\n");
        
    }
}


GpuHashTable::~GpuHashTable() {
    cudaFree(hashmap);
    cudaFree(keys_to_insert);
    cudaFree(values_to_insert);
    cudaFree(new_inserted);
}


void GpuHashTable::reshape(int numBucketsReshape) {
    numBucketsReshape = ((used_size + numBucketsReshape) * 1.178);
    struct Nod *new_hashmap;
    cudaMalloc((void **) &new_hashmap, numBucketsReshape * sizeof(struct Nod));
    if (new_hashmap == 0)
    {
        printf("eroare la alocare in reshape\n");
        return;
    }
    cudaMemset(new_hashmap, 0, numBucketsReshape * sizeof(struct Nod));
    int nrblocks = size / nb_threads;
    if(size % nb_threads != 0)


        nrblocks++;
    reshapeHashmap>> (hashmap, new_hashmap, size, numBucketsReshape);
    cudaDeviceSynchronize();
    cudaFree(hashmap);
    hashmap = new_hashmap;
    size = numBucketsReshape; 
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {

    if (used_size + numKeys > size)
        reshape(numKeys);
    if (batch_size < numKeys)
    {
        cudaFree(keys_to_insert);
        cudaFree(values_to_insert);
        batch_size = numKeys;
        cudaMalloc((void **) &keys_to_insert, batch_size * sizeof(int));
        cudaMalloc((void **) &values_to_insert, batch_size * sizeof(int));
    }
    int nr_noi = 0;
    cudaMemset(new_inserted, 0, sizeof(int));
    if(hashmap == 0 || keys_to_insert == 0 || values_to_insert == 0)


        return false;
    cudaMemcpy(keys_to_insert, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(values_to_insert, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    int nrblocks = numKeys / nb_threads;


    if(numKeys % nb_threads != 0)
        nrblocks++;
    insert_element>> (keys_to_insert, values_to_insert, numKeys, hashmap, size, new_inserted);
    cudaDeviceSynchronize();
    cudaMemcpy(&nr_noi, new_inserted, sizeof(int), cudaMemcpyDeviceToHost);
    used_size += (numKeys - nr_noi);
    if (loadFactor() < 0.8)
        reshape(ceil(used_size / 0.85));
	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys) {
    
    if (batch_size < numKeys)
    {
        cudaFree(keys_to_insert);
        cudaFree(values_to_insert);
        batch_size = numKeys;
        cudaMalloc((void **) &keys_to_insert, batch_size * sizeof(int));
        cudaMalloc((void **) &values_to_insert, batch_size * sizeof(int));
    }
 
    if(hashmap == 0 || keys_to_insert == 0 || values_to_insert == 0)
        return NULL;
    cudaMemcpy(keys_to_insert, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    int nrblocks = numKeys / nb_threads;
    if(numKeys % nb_threads != 0)
        nrblocks++;
    get_elements>>(keys_to_insert, values_to_insert, numKeys, hashmap, size);
    cudaDeviceSynchronize();
    int *result = (int *)malloc(numKeys * sizeof(int));
    cudaMemcpy(result, values_to_insert, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
    
	return result;
}



float GpuHashTable::loadFactor() {
	return (float) used_size / size;
    
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





struct Nod
{
    uint32_t key, value;
};

class GpuHashTable
{
    struct Nod *hashmap;
    uint32_t size, used_size;
    uint32_t batch_size;
    int *keys_to_insert, *values_to_insert;
    int *new_inserted;
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

