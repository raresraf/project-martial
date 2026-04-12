

/**
 * @file gpu_hashtable.cu
 * @brief This file contains a CUDA implementation of a GPU-based hash table.
 * @raw/dabc7569-a3e3-4315-a67a-fd2d97d99943/gpu_hashtable.cu
 *
 * This implementation uses open addressing with linear probing for collision resolution.
 * Key-value pairs are stored in a struct called `Celula`. The main operations (insert, get, reshape)
 * are parallelized on the GPU using CUDA kernels. Variable names such as `Celula`, `cheie`, and `valoare`
 * are in Romanian, meaning "cell", "key", and "value" respectively.
 *
 * Algorithm: Open Addressing with Linear Probing
 * Time Complexity:
 *  - Insertion: Average O(1), Worst O(N)
 *  - Retrieval: Average O(1), Worst O(N)
 *  - Reshape: O(N)
 * Space Complexity: O(N)
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @brief Constructor for the GpuHashTable class.
 * @param siz The initial capacity of the hash table.
 *
 * Allocates an array of `Celula` structs on the GPU and initializes memory to zero.
 */
GpuHashTable::GpuHashTable(int siz) {
	if (siz <= 0){
		std::cerr <<"size < 0
";
		exit(-1);
	}
	int err;

	
	inserari = 0;
	size = siz;
	
	err = cudaMalloc(&vect, size * sizeof(Celula));
	if (err != cudaSuccess || vect == NULL){
		std::cerr << "Memoria init
";
		exit(-1);
	}
	err = cudaMemset(vect, 0, size * sizeof(Celula));
	if (err != cudaSuccess) {
		std::cerr<<"Memset
";
		exit(-1);
	}
}


/**
 * @brief Destructor for the GpuHashTable class.
 *
 * Frees the GPU memory allocated for the hash table vector.
 */
GpuHashTable::~GpuHashTable() {
	int err;
	
	
	err = cudaFree(vect);
	if (err != cudaSuccess)
		exit(-1);
}


/**
 * @brief Computes the hash of a given key using a multiplicative method.
 * @param data The key to hash.
 * @param limit The size of the hash table, used to wrap the hash value.
 * @return The hash value for the given key.
 */
__device__ int dhash1(int data, int limit) {


	return ((long long)abs(data) * 10453007) % 3452434812973 % limit;
}

/**
 * @brief CUDA kernel for resizing and re-hashing the hash table.
 * @param vect Pointer to the old `Celula` array on the device.
 * @param newvect Pointer to the new `Celula` array on the device.
 * @param size The size of the old hash table.
 * @param newsize The size of the new hash table.
 *
 * Each thread processes one entry from the old hash table. If the entry is valid,
 * it re-hashes the key and inserts the `Celula` into the new table using linear probing
 * and `atomicCAS` for thread-safe insertion.
 */
__global__ void reshapeAux(Celula *vect, Celula *newvect, int size, int newsize) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j;
	
	// Each thread checks one cell from the old vector.
	if (i < size && vect[i].cheie != 0) {
		int cheie = vect[i].cheie;
		
		// Re-hash the key for the new table size and start linear probing.
		int begin = dhash1(cheie, newsize), cc;
		for (j = begin; j < newsize; j++){
			
			// Attempt to atomically claim an empty slot.
			cc = atomicCAS(&newvect[j].cheie, 0, cheie);
			if (cc == 0){
				
				newvect[j].valoare = vect[i].valoare;
				return;
			}
		}
		
		// If not inserted, wrap around and continue probing from the start.
		for (j = 0; j < begin; j++) {
			cc = atomicCAS(&newvect[j].cheie, 0, cheie);
			if (cc == 0){
				newvect[j].valoare = vect[i].valoare;
				return;
			}
		}
	}
}

/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The new capacity of the hash table.
 *
 * This host function orchestrates the resize operation. It allocates a new `Celula`
 * vector on the GPU and launches the `reshapeAux` kernel to re-hash the elements.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	if(numBucketsReshape <= 0)
		exit(-1);
	if(vect == NULL){
		std::cerr << "null vect
";
		exit(-1);
	}
	int err;
	size_t block_size = 1024, num_blocks;
	Celula *newvect;
	
	
	err = cudaMalloc(&newvect, numBucketsReshape * sizeof(Celula));
	if (err != cudaSuccess || newvect == NULL){
		std::cerr << "Memoria reshape
";
		exit(-1);
	}
	err = cudaMemset(newvect, 0, numBucketsReshape * sizeof(Celula));
	if (err != cudaSuccess) {
		std::cerr << "Memset reshape
";
		exit(-1);
	}

	num_blocks = (size + block_size - 1)/ block_size;
	reshapeAux>>(vect, newvect, size, numBucketsReshape);
	cudaDeviceSynchronize();
	err = cudaFree(vect);
	if (err != cudaSuccess) {
		std::cerr <<size<< " free reshape
";
		exit(-1);
	}

	
	size = numBucketsReshape;
	vect = newvect;
}



/**
 * @brief CUDA kernel for inserting a batch of key-value pairs.
 * @param sent_keys Pointer to the batch of keys to insert (on the device).
 * @param sent_values Pointer to the batch of values to insert (on the device).
 * @param vect The hash table's `Celula` array.
 * @param nr The number of key-value pairs in the batch.
 * @param size The total capacity of the hash table.
 *
 * Each thread handles one key-value pair, using linear probing and `atomicCAS`
 * to find an empty slot or update an existing key.
 */
__global__ void insert(int *sent_keys, int *sent_values, Celula *vect, int nr, int size) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j;

	if (i < nr && sent_keys[i] != 0) {
		int key = sent_keys[i], cc; 
		int begin = dhash1(key, size);

		// Start linear probing from the hashed index.
		for(j = begin; j < size; j++) {
			// Atomically check if the slot is empty (0) or contains the same key.
			cc = atomicCAS(&vect[j].cheie, 0, key);
			if(cc == 0 || cc == key) {
				vect[j].valoare = sent_values[i];
				return;
			}
		}
		
		// Wrap around and continue probing from the beginning.
		for (j = 0; j < begin; j++) {
			cc = atomicCAS(&vect[j].cheie, 0, key);
			if (cc == 0 || cc == key) {
				vect[j].valoare = sent_values[i];
				return;
			}
		}
	}
}

/**
 * @brief CUDA kernel for retrieving a batch of values.
 * @param sent_keys Pointer to the batch of keys to search for (on the device).
 * @param vect The hash table's `Celula` array.
 * @param nr The number of keys in the batch.
 * @param size The total capacity of the hash table.
 * @param ret_values Pointer to an array where the results will be stored.
 *
 * Each thread searches for one key using linear probing.
 * @warning This kernel may enter an infinite loop if a key is not present in the table,
 * as there is no stop condition for the search if the key is not found.
 */
__global__ void bashget(int *sent_keys, Celula *vect, int nr, int size, int *ret_values) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j;

	if (i < nr && sent_keys[i] != 0) {
		int key = sent_keys[i];
		int begin = dhash1(key, size);

		// Linearly probe for the key.
		for (j = begin; j < size; j++) {
			
			if (key == vect[j].cheie) {
				
				ret_values[i] = vect[j].valoare;
				return;
			}
		}
		
		// Wrap around and continue probing.
		for(j = 0; j < begin; j++) {
			if(key == vect[j].cheie) {
				ret_values[i] = vect[j].valoare;
				return;
			}
		}
	}
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys Pointer to the host array of keys.
 * @param values Pointer to the host array of values.
 * @param numKeys The number of key-value pairs to insert.
 * @return `true` if the insertion process was initiated successfully.
 *
 * This function handles host-to-device memory transfers, triggers a `reshape`
 * if the load factor exceeds a threshold (0.9), and launches the `insert` kernel.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	if (!keys || !values || numKeys <= 0)
		return false;
	int *sent_keys = NULL , *sent_values = NULL, err;
	size_t block_size = 1024, num_blocks;

	err = cudaMalloc(&sent_keys, numKeys * sizeof(int));
	if (err != cudaSuccess || sent_keys == NULL) {
		std::cerr << "Memoria insert 1
";
		exit(-1);
	}
	err = cudaMalloc(&sent_values, numKeys * sizeof(int));
	if (err != cudaSuccess || sent_values == NULL){
		std::cerr << "Memoria insert 2
";
		exit(-1);
	}

	err = cudaMemcpy(sent_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		std::cerr << "Memcpy insert
";
		exit(-1);
	}
	err = cudaMemcpy(sent_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		std::cerr << "memcpy insert
";
		exit(-1);
	}
	
	
	// Check load factor and reshape if it exceeds the threshold.
	int next = inserari + numKeys;
	int newSize = (int) (next / 0.8);
	if(((float)next) / size >= 0.9)
		reshape(newSize);
	num_blocks = (numKeys + block_size - 1)/ block_size;
	insert>>(sent_keys, sent_values, vect, numKeys, size);
	cudaDeviceSynchronize();
	
	inserari += numKeys;
	err = cudaFree(sent_values);
	if (err != cudaSuccess){
		std::cerr << "free insert 1
";
		exit(-1);
	}
	err = cudaFree(sent_keys);
	if (err != cudaSuccess){
		std::cerr << "free insert 2
";
		exit(-1);
	}
	return true;
}


/**
 * @brief Retrieves the values for a batch of keys.
 * @param keys Pointer to the host array of keys to search for.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to a host array containing the retrieved values.
 *
 * This function handles memory transfers and launches the `bashget` kernel.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	if (!keys || numKeys <= 0)
		return NULL;
	int *sent_keys, *ret_values, err;
	size_t block_size = 1024, num_blocks;

	err = cudaMalloc(&sent_keys, numKeys * sizeof(int));
	if (err != cudaSuccess || sent_keys == NULL){
		std::cerr << "Memoria get
";
		exit(-1);
	}
	err = cudaMalloc(&ret_values, numKeys * sizeof(int));
	if (err != cudaSuccess || ret_values == NULL){
		std::cerr << "calloc get
";
		exit(-1);
	}
	err = cudaMemset(ret_values, 0, numKeys * sizeof(int));
	if (err != cudaSuccess) {
		std::cerr << "memcpy get";
		exit(-1);
	}
	err = cudaMemcpy(sent_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		std::cerr << "memcpy get
";
		exit(-1);
	}
	Celula *vl = (Celula *)calloc(size, sizeof(Celula));
	cudaMemcpy(vl, vect, size * sizeof(Celula), cudaMemcpyDeviceToHost);
	num_blocks = (numKeys + block_size - 1)/ block_size;
	bashget>>(sent_keys, vect, numKeys, size, ret_values);
	cudaDeviceSynchronize();
	err = cudaFree(sent_keys);
	if (err != cudaSuccess){
		std::cerr << "free get
";
		exit(-1);
	}

	
	int *res = (int *) calloc(numKeys, sizeof(int));
	if (res == NULL) {
		exit(-1);
	}
	cudaMemcpy(res, ret_values, sizeof(int) * numKeys, cudaMemcpyDeviceToHost);
	return res;
}


/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float.
 */
float GpuHashTable::loadFactor() {
	if (size == 0)
		return 0.f;
	float load = (float)inserari/size;

	return load;
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

#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
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



// Alternative hash functions, not used in the current implementation.
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}


int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}



/**
 * @brief Defines the structure for a hash table entry.
 * 'cheie' is Romanian for 'key'.
 * 'valoare' is Romanian for 'value'.
 */
typedef struct
{
	int cheie;
	int valoare;
}Celula;

class GpuHashTable
{
	int inserari;
	int size;
	Celula *vect;
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
