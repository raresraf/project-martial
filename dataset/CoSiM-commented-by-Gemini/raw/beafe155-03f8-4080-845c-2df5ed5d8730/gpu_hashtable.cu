/**
 * @file gpu_hashtable.cu
 * @brief CUDA implementation of a cuckoo hash table.
 *
 * This file provides a GPU-based implementation of a hash table using the
 * cuckoo hashing algorithm. It is designed for high-throughput insertions and
 * lookups on the GPU. The implementation includes kernels for `get`, `insert`,
 * and `rehash` operations, as well as a host-side class to manage the hash
 * table.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define THREADS_PER_BLOCK 1024
#define MIN_LOAD_FACTOR 0.6
#define MAX_LOAD_FACTOR 0.8

/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param keys A device pointer to the array of keys to look up.
 * @param values A device pointer to the array where retrieved values will be stored.
 * @param NumIntrari The number of keys to look up.
 * @param hashmap The hash table to search in.
 *
 * This kernel implements a parallel lookup in the cuckoo hash table. Each thread
 * is responsible for one key. It computes the hash for the key and searches
 * for it in the two possible locations in the hash table.
 */
__global__ void get (int * keys, int * values, int NumIntrari, hash_table hashmap) {
	int i, j, idx, key, hash;
	idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	hash = (abs(keys [idx]) * 100000) % 999999999999 % hashmap.size;
	
	int inceput [2] = {hash, 0};
	int sfarsit [2] = {hashmap.size, hash};
	key = keys [idx];
	// Pre-condition: Ensure the thread index is within the bounds of the input arrays.
	if (idx >= NumIntrari)
		return;
	// Invariant: The loop iterates through the two possible locations for the key
	// in the cuckoo hash table.
	for (i = 0; i < 2; i++) {
		for (j = inceput [i]; j < sfarsit [i]; j++) {
			
			if (hashmap.map [0][j].key == key) {
				values [idx] = hashmap.map [0][j].value;
				return;
			}
			else
			
			if (hashmap.map [1][j].key == key) {
				values [idx] = hashmap.map [1][i].value;
				return;
			}
		}
 	}
}

/**
 * @brief CUDA kernel to rehash the table.
 * @param Veche The old hash table.
 * @param Noua The new, larger hash table.
 *
 * This kernel is responsible for re-inserting all elements from an old hash
 * table into a new one when the load factor exceeds a certain threshold. Each
 * thread handles one element from the old table.
 */
__global__ void rehash (hash_table Veche, hash_table Noua) {
	int i, j, poz, idx, key, hash, OldKey, NewKey;
	
	bool reusit = false;
	idx = blockDim.x * blockIdx.x + threadIdx.x;
	// Invariant: Iterates through both hash functions' tables to find a valid key.
	for (poz = 0; poz < 2; poz++) {
		if (Veche.map [poz][idx].key == KEY_INVALID) {
			continue;
		}
	}
	NewKey = Veche.map [poz][idx].key;
	
	hash = (abs(NewKey) * 100000) % 999999999999 % Noua.size;
	
	int inceput [2] = {hash, 0};
	int sfarsit [2] = {hashmap.size, hash};
	// Invariant: Tries to insert the key into the new hash table, handling
	// potential collisions using atomic operations.
	for (i = 0; i < 2; i++) {
		if (reusit) {
			for (j = inceput [i]; j < sfarsit [i]; j++) {
				
				OldKey = atomicCAS(&Noua.map [0][j].key, KEY_INVALID, NewKey);
				if (OldKey == KEY_INVALID) {
					Noua.map [0][j].value = Veche.map [poz][idx].value;
					break;
				}
				else {
				
				OldKey = atomicCAS(&Noua.map [1][j].key, KEY_INVALID, NewKey);
				if (OldKey == KEY_INVALID) {
					Noua.map [1][j].value = Veche.map [poz][idx].value;
					break;
				}
				reusit = TRUE;
			}
		}
	}
}

/**
 * @brief CUDA kernel to insert a batch of key-value pairs.
 * @param keys A device pointer to the array of keys to insert.
 * @param values A device pointer to the array of values to insert.
 * @param NumIntrari The number of pairs to insert.
 * @param hashmap The hash table.
 *
 * This kernel implements a parallel insertion into the cuckoo hash table.
 * Each thread handles one key-value pair and uses atomic compare-and-swap
 * (atomicCAS) to handle potential race conditions during insertion.
 */
__global__ void insert (int * keys, int * values, int NumIntrari, hash_table hashmap) {

	int i, j, hash, OldKey, NewKey, idx;
	idx = blockDim.x * blockIdx.x + threadIdx.x;
	NewKey = keys[idx];
	
	hash = (abs(NewKey) * 100000) % 999999999999 % hashmap.size;
	
	int inceput [2] = {hash, 0};
	int sfarsit [2] = {hashmap.size, hash};
	
	if (idx >= NumIntrari)
		return;
	
	// Invariant: The loop attempts to insert the key-value pair into one of the
	// two possible locations, using atomicCAS to ensure thread safety.
	for (i = 0; i < 2; i++) {
		for (j = inceput [i]; j < sfarsit [i]; j++) {
			
			OldKey = atomicCAS(&hashmap.map[0][i].key, KEY_INVALID, NewKey);
			if (OldKey == KEY_INVALID || OldKey == NewKey) {
				
				hashmap.map [0][j].value = values [idx];
				return;
			}
			else {
			
				OldKey = atomicCAS(&hashmap.map[1][i].key, KEY_INVALID, NewKey); 
				if (OldKey == INVALID_KEY || OldKey == NewKey) {
					hashmap.[1][j].value = values [idx];
					return;
				}
			}
		}
	}
}


/**
 * @brief Constructor for the GpuHashTable class.
 * @param size The initial size of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	hashmap.size = size;
	int poz;
	NumarPerechiAdaugate = 0;
	// Invariant: Initializes both tables of the cuckoo hash map.
	for (poz = 0; poz < 2; poz++) {
		hashmap.map[poz] = nullptr;
		if (cudaMalloc (&hashmap.map [poz], size * sizeof (entre)) != cudaSuccess) {
			std:cerr << "Memory allocation error\n";
			return;
		}
		cudaMemset(hashmap.map[poz], 0, size * sizeof (entry));
	} 
}


/**
 * @brief Destructor for the GpuHashTable class.
 */
GpuHashTable::~GpuHashTable() {
	int poz;
	for (poz = 0; poz < 2; poz++) {
		cudaFree (hashmap.map[poz]);
	}
}


/**
 * @brief Resizes and rehashes the hash table.
 * @param numBucketsReshape The new size of the hash table.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int poz;
	hash_table NewHash;
	NewHash.size = numBucketsReshape;
	int nrElemente = hashmap.size / THREADS_PER_BLOCK;
	
	for (poz = 0; poz < 2; poz++) {
		if (cudaMalloc (&NewHash.map[poz], numBucketsReshape * sizeof (entry)) != cudaSuccess) {
			std:cerr << "RESHAPE Memory allocation error\n";
			return;
		}
		cudaMemset(NewHash.map[poz], 0, numBucketsReshape * sizeof (entry));
	}

	
	if (hashmap.size % THREADS_PER_BLOCK != 0)
		nrElemente = nrElemente + 1;
	rehash >> (hashmap, NewHash);
	cudaDeviceSynchronize();
	
	cudaFree (hashmap.map[0]);
	cudaFree (hashmap.map[1]);
	hashmap = NewHash;
}


/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys A host pointer to the array of keys.
 * @param values A host pointer to the array of values.
 * @param numKeys The number of pairs to insert.
 * @return True on success, false on failure.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *nrChei, *nrValori; 
	int nrElemente = numKeys / THREADS_PER_BLOCK;
	
	cudaMalloc (&nrChei, numKeys * sizeof (int));
	cudaMalloc (&nrValori, numKeys * sizeof (int));
	if (!nrChei || !nrValori) {
		std::cerr << "Memory allocation error\n";
		return false;
	}
	// Pre-condition: If the load factor will exceed the maximum, resize the table.
	if (float (NumarPerechiAdaugate + numKeys) / hashmap.size >= MAX_LOAD_FACTOR)
		reshape (int (( NumarPerechiAdaugate + numKeys) / MIN_LOAD_FACTOR));
	
	cudaMemcpy (nrChei, keys, numKeys * sizeof (int), cudaMemcpyHostToDevice);
	cudaMemcpy (nrValori, keys, numKeys * sizeof (int), cudaMemcpyHostToDevice);
	if (numKeys % THREADS_PER_BLOCK != 0)
		nrElemente++;
	insert >> (nrChei, nrValori, numKeys, hashmap);
	cudaDeviceSynchronize();
	NumarPerechiAdaugate = NumarPerechiAdaugate + numKeys;
	
	cudaFree (nrChei);
	cudaFree (nrValori);
	return true;
}


/**
 * @brief Retrieves values for a batch of keys.
 * @param keys A host pointer to the array of keys.
 * @param numKeys The number of keys to look up.
 * @return A host pointer to the array of retrieved values.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int valori, *nrChei;
	int nrElemente = numKeys / THREADS_PER_BLOCK;
	
	cudaMallocManaged(&valori, numKeys * sizeof(int));
	cudaMalloc(&nrChei, numKeys * sizeof(int));
	
	
	if (!nrChei || !valori) {
		std::cerr << "Memory allocation error\n";
		return nullptr;
	}
	
	
	cudaMemcpy(nrChei, valori, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	if (numKeys % THREADS_PER_BLOCK != 0)
		nrElemente = nrElemente + 1;
	get >> (nrChei, valori, numKeys, hashmap);
	cudaDeviceSynchronize();
	
	cudaFree(nrChei);
	return valori;
}




float GpuHashTable::loadFactor() {
	
	if (hashmap.size == 0)
		return 0;
	return float(NumarPerechiAdaugate) / hashmap.size;
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
	do { \
		if (assertion) { \
		fprintf(stderr, "(%s, %d): ", \
		__FILE__, __LINE__); \
		perror(call_description); \
		exit(errno); \
	} \
} while (0)
	
/**
 * @brief A list of prime numbers used for hashing.
 *
 * This list is used to provide good distribution in the hash functions.
 */
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

struct entry {
	int key, value;
};

struct hash_table {
	int size;


	entry *map[2];
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
	int NumarPerechiAdaugate;
	hash_table hashmap;
	
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