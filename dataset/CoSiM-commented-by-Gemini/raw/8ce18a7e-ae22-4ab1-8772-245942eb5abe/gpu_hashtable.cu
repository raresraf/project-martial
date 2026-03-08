/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 * @details This file defines a hash table that operates on the GPU, leveraging CUDA kernels for
 * parallel batch insertion, retrieval, and resizing operations. The collision resolution
 * strategy is linear probing with atomic operations to ensure thread safety.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

/**
 * @struct Bucket
 * @brief Represents a single key-value entry in the hash table.
 */
typedef struct Bucket {
	int cheie;   // The key (cheie) of the entry.
	int valoare; // The value (valoare) associated with the key.
} Bucket;

// Global variables for the hash table state.
int  lungime = 0;        // Represents the total capacity (length) of the hash table.
int numar_elemente = 0; // Represents the current number of elements stored.
Bucket *buckets;        // Device pointer to the array of buckets on the GPU.


/**
 * @brief CUDA kernel for parallel batch insertion of key-value pairs.
 * @param buckets Device pointer to the hash table buckets.
 * @param marime The number of key-value pairs to insert (size of the batch).
 * @param lungime The total capacity of the hash table.
 * @param chei Device pointer to the array of keys to insert.
 * @param valori Device pointer to the array of values to insert.
 */
__global__ void inserare(Bucket *buckets, int marime, int lungime, int *chei, int *valori) {
	// Each thread is responsible for inserting one key-value pair.
	unsigned int id_thread = threadIdx.x + blockDim.x * blockIdx.x;

	if (id_thread < marime) {

		// Hash calculation: A multiplicative hash function using prime numbers to distribute keys.
		long pozitie = ((long)abs(chei[id_thread]) * 163307llu) % 668993977llu % lungime;

		int cheie_n = chei[id_thread]; // The new key to be inserted.
		int cheie_v; // Variable to store the value read from the bucket's key.

		/**
		 * @block Collision Resolution: Linear Probing with Atomic Compare-and-Swap (CAS).
		 * @logic The loop attempts to find an empty bucket (where key is 0) or an existing bucket with the same key.
		 * `atomicCAS` ensures that only one thread can claim a bucket in case of a hash collision.
		 * If `atomicCAS` fails (returns a non-zero value that is not our key), it means another thread
		 * has written to this bucket, so we probe the next position.
		 */
		while ((cheie_v = atomicCAS(&(buckets[pozitie].cheie), 0, cheie_n)) != 0 && cheie_v != cheie_n ) {
			pozitie++;
			// Wrap around to the beginning if the end of the table is reached.
			if (pozitie == lungime) {
				pozitie = 0;
			}
		}
		// Once an empty or matching bucket is secured, write the value.
		buckets[pozitie].valoare = valori[id_thread];
	}
	return;
}

/**
 * @brief CUDA kernel for rehashing the table into a new, larger array of buckets.
 * @param buckets Device pointer to the old hash table buckets.
 * @param new_buckets Device pointer to the new, larger hash table buckets.
 * @param marime The size of the old hash table.
 * @param lungime The capacity of the new hash table.
 */
__global__ void reformare(Bucket *buckets, Bucket *new_buckets ,int marime, int lungime) {
	// Each thread is responsible for rehashing one bucket from the old table.
	unsigned int id_thread = threadIdx.x + blockDim.x * blockIdx.x;

	if (id_thread < marime) {
		// Ignore empty buckets from the old table.
		if (buckets[id_thread].cheie == 0)
			return;

		// Re-calculate the hash for the key in the context of the new table's larger size.
		long pozitie = ((long)abs(buckets[id_thread].cheie) * 163307llu) % 668993977llu % lungime;
		int cheie_v;


		/**
		 * @block Collision Resolution (Rehashing): Linear Probing with Atomic CAS.
		 * @logic Similar to the insertion kernel, this finds an empty slot in the new table
		 * for the element from the old table, handling potential collisions during the rehash process.
		 */
		while ((cheie_v = atomicCAS(&(new_buckets[pozitie].cheie), 0, buckets[id_thread].cheie)) != 0 && cheie_v != buckets[id_thread].cheie ) {
			pozitie++;
			if (pozitie == lungime) {
				pozitie = 0;
			}
		}

		// Copy the value to the new bucket location.
		new_buckets[pozitie].valoare = buckets[id_thread].valoare;
		return;
	}
}


/**
 * @brief CUDA kernel for parallel batch retrieval of values based on keys.
 * @param buckets Device pointer to the hash table buckets.
 * @param marime The number of keys to look up.
 * @param lungime The total capacity of the hash table.
 * @param chei Device pointer to the array of keys to find.
 * @param valori Device pointer to an array where the found values will be written.
 */
__global__ void retur(Bucket *buckets, int marime, int lungime, int *chei, int *valori) {
	unsigned int id_thread = threadIdx.x + blockDim.x * blockIdx.x;

	if (id_thread < marime) {

		// Calculate the initial hash position for the key.
		long pozitie = ((long)abs(chei[id_thread]) * 163307llu) % 668993977llu % lungime;
		int cheie_c = chei[id_thread]; // The current key to search for.

		/**
		 * @block Search: Linear Probing.
		 * @logic Starting from the initial hash position, probe linearly until the bucket
		 * with the matching key is found. This assumes the key exists in the table.
		 */
		while (buckets[pozitie].cheie != cheie_c) {
			pozitie++;
			// Wrap around if the end of the table is reached.
			if (pozitie == lungime) {
				pozitie = 0;
			}
		}

		// Write the found value to the output array at the corresponding thread's index.
		valori[id_thread] = buckets[pozitie].valoare;
		return;
	}
	return;
}


/**
 * @brief Constructor for the GpuHashTable.
 * @param size Initial capacity of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	// Allocate memory on the GPU for the buckets.
	cudaMalloc((void **) &(buckets), size * sizeof(Bucket));
	// Initialize the allocated memory to zero.
	cudaMemset(buckets, 0, size * sizeof(Bucket));

	lungime = size;
}

/**
 * @brief Destructor for the GpuHashTable.
 */
GpuHashTable::~GpuHashTable() {
	// Free the allocated GPU memory.
	cudaFree(buckets);
}

/**
 * @brief Resizes and rehashes the table if the load factor exceeds a threshold.
 * @param numBucketsReshape The number of new elements being added, used to calculate the new size.
 */
void GpuHashTable::reshape(int numBucketsReshape) {

	numar_elemente += numBucketsReshape;
	int lungime_v = lungime; // Old length/capacity.

	// Check if the load factor exceeds the 0.8 threshold.
	if (numar_elemente / (float)lungime_v > 0.8) {
		// Calculate a new size, providing some headroom.
		lungime = numar_elemente * 10 / 8;

		// Allocate a new, larger bucket array on the device.
		Bucket *new_buckets;
		cudaMalloc((void **) &(new_buckets),lungime * sizeof(Bucket));
		cudaMemset(new_buckets, 0, lungime * sizeof(Bucket));

		// Configure and launch the rehashing kernel.
		const size_t block_size = 1024;
		size_t blocks_no = ceil((float)lungime_v / (float)block_size);
 
  		reformare>>(blocks_no, block_size, marime, lungime);
  		cudaDeviceSynchronize();
	 
  		cudaFree(buckets); // Free the old bucket array.

  		buckets = new_buckets; // Point to the new bucket array.
	}
}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys Host pointer to an array of keys.
 * @param values Host pointer to an array of values.
 * @param numKeys The number of pairs to insert.
 * @return Always returns true.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *chei, *valori; // Device pointers for keys and values.

	// Allocate memory on the device for the batch.
	cudaMalloc((void **) &(chei), numKeys * sizeof(int));
	cudaMalloc((void **) &(valori), numKeys * sizeof(int));

	// Copy data from host to device.
	cudaMemcpy(chei, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(valori, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Check if a reshape is needed before insertion.
	reshape(numKeys);

	// Configure and launch the insertion kernel.
	const size_t block_size = 1024;
	size_t blocks_no = ceil((float)numKeys / (float)block_size);

  	inserare>>(blocks_no, block_size, numKeys, lungime, chei, valori);
  	
  	// Free the temporary device memory for the batch.
  	cudaFree(chei);
  	cudaFree(valori);

	return true;
}

/**
 * @brief Retrieves a batch of values corresponding to a batch of keys.
 * @param keys Host pointer to an array of keys to look up.
 * @param numKeys The number of keys to retrieve.
 * @return A host pointer to an array containing the retrieved values.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *chei, *valori, *rezultate; // Device and host pointers.

	// Allocate host memory for the results.
	rezultate = (int *)malloc(numKeys * sizeof(int));

	// Allocate device memory for the lookup operation.
	cudaMalloc((void **) &(chei), numKeys * sizeof(int));
	cudaMalloc((void **) &(valori), numKeys * sizeof(int));

	// Copy keys from host to device.
	cudaMemcpy(chei, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	
	// Configure and launch the retrieval kernel.
	const size_t block_size = 512;
	size_t blocks_no = ceil((float)numKeys / (float)block_size);
  	
  	retur>>(blocks_no, block_size, numKeys, lungime, chei, valori);

	// Wait for kernel completion before copying results.
	cudaDeviceSynchronize();

  	// Copy the retrieved values from device to host.
  	cudaMemcpy(rezultate, valori, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	// Note: Device memory for `chei` and `valori` is not freed here, leading to a memory leak.
	return rezultate;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float.
 */
float GpuHashTable::loadFactor() {
	return (float)numar_elemente / (float)lungime; 
}


// The following section appears to be a separate test harness or a CPU implementation,
// included directly in the .cu file. It is not directly part of the GpuHashTable class logic.

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
	
// A large list of prime numbers, likely for use in various hashing schemes.
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



// A series of hash functions, likely for experimentation or for a different (CPU) hash table variant.
int hash1(int data, int limit) {
	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;
}
int hash2(int data, int limit) {
	return ((long)abs(data) * primeList[67]) % primeList[91] % limit;
}
int hash3(int data, int limit) {
	return ((long)abs(data) * primeList[70]) % primeList[93] % limit;
}



// This appears to be a forward declaration for a CPU-based HashTable class, not the GpuHashTable.
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
};

#endif
