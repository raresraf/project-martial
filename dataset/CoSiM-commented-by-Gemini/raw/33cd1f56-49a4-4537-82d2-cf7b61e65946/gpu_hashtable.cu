
/**

 * @file gpu_hashtable.cu

 * @brief Implements a concurrent, lock-free hash table on the GPU using CUDA.

 *

 * This hash table supports parallel batch insertions and retrievals. It uses open

 * addressing with linear probing for collision resolution and relies on atomic

 * compare-and-swap (CAS) operations to ensure thread safety without traditional locks.

 */



#include 

#include 

#include 

#include 

#include 

#include 



#include "gpu_hashtable.hpp"





/**

 * @brief CUDA kernel to insert key-value pairs into the hash table in parallel.

 * Each thread is responsible for inserting one key-value pair.

 * @param keys      Device pointer to the array of keys to insert.

 * @param values    Device pointer to the array of values to insert.

 * @param numKeys   The number of key-value pairs to insert.

 * @param hmap      The hash map structure containing device pointers and size.

 */

__global__ void thread_insert(int *keys, int *values, int numKeys, Hmap hmap) {



  	// Functional Utility: Calculate a unique global index for each thread.

  	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

  	if(i >= numKeys)

  		return;



    // A key of 0 is treated as invalid/empty.

  	if( keys[i] == 0)

  		return;



  	int hashed_idx = hash1(keys[i], hmap.size);



    /**

     * Block Logic: Implements linear probing to find a slot for the key.

     * The loop continues until the key is successfully inserted or updated.

     */

  	while(1) {



        /**

         * Concurrency Control: Attempt to claim an empty slot.

         * `atomicCAS` atomically compares the value at `&hmap.keys[hashed_idx]` with 0.

         * If they are equal, it writes `keys[i]` to that location. The operation

         * returns the old value (0 on success), indicating the thread has won the slot.

         */

		if( atomicCAS(&hmap.keys[hashed_idx], 0, keys[i]) == 0 ){

			// Safely write the value since this thread now owns the slot.

			atomicExch(&hmap.values[hashed_idx], values[i]);

			break;

		}



        /**

         * Concurrency Control: Handle the case where the key already exists (update).

         * `atomicCAS` checks if the key at the current slot is the one we are trying

         * to insert. If so, we can proceed to update its value.

         */

		if( atomicCAS(&hmap.keys[hashed_idx], keys[i], keys[i]) == keys[i]){

			// The key exists, so atomically update its value.

			atomicExch(&hmap.values[hashed_idx], values[i]);

			break;

		}



        // Collision detected, move to the next slot (linear probing).

    	hashed_idx++;

     	hashed_idx %= hmap.size; // Wrap around to the beginning of the table if necessary.

    }

}





/**

 * @brief CUDA kernel to retrieve values for a batch of keys in parallel.

 * @param keys      Device pointer to the array of keys to look up.

 * @param numKeys   The number of keys to look up.

 * @param hmap      The hash map structure.

 * @param results   Device pointer to an array where the results will be stored.

 */

__global__ void thread_get(int *keys, int numKeys, Hmap hmap, int *results) {



  	// Calculate a unique global index for each thread.

  	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;





  	if(i >= numKeys)

  		return;

  	int hashed_idx = hash1(keys[i], hmap.size);



    /**

     * Block Logic: Implements linear probing to find the specified key.

     */

  	while(1) {



        /**

         * Concurrency Control: Safely check if the key at the current slot matches.

         * Using `atomicCAS` here prevents reading a key that might be in the process

         * of being written by another thread in a concurrent insertion.

         */

    	if(atomicCAS(&hmap.keys[hashed_idx], keys[i], keys[i]) == keys[i]){

    		// Key found, atomically read its value.

    		atomicExch(&results[i], hmap.values[hashed_idx]);

    		break;

    	}



        // Key not found at this slot, move to the next one.

    	hashed_idx++;

     	hashed_idx %= hmap.size;

    }

}





/**

 * @brief Host-side constructor for the GpuHashTable.

 * Allocates memory on the GPU for the keys and values arrays.

 * @param size The initial number of buckets in the hash table.

 */

GpuHashTable::GpuHashTable(int size) {

	hm.current_no_of_pairs = 0;

	hm.size = size;

	cudaMalloc((void **) &hm.keys, size * sizeof(int));

	cudaMalloc((void **) &hm.values, size * sizeof(int));



    // Initialize all keys and values to 0 (empty).

	cudaMemset(hm.keys, 0, size * sizeof(int));

	cudaMemset(hm.values, 0, size * sizeof(int));

}



/**

 * @brief Host-side destructor. Frees the GPU memory.

 */

GpuHashTable::~GpuHashTable() {

	cudaFree(hm.keys);

	cudaFree(hm.values);

}





/**

 * @brief Resizes the hash table and re-hashes all existing elements.

 * @param numBucketsReshape The new size of the hash table.

 */

void GpuHashTable::reshape(int numBucketsReshape) {



	Hmap new_hm;

	new_hm.size = numBucketsReshape;



	// Allocate new, larger arrays on the GPU.

	cudaMalloc((void **) &new_hm.keys, numBucketsReshape * sizeof(int));

	cudaMalloc((void **) &new_hm.values, numBucketsReshape * sizeof(int));

	cudaMemset(new_hm.keys, 0, numBucketsReshape * sizeof(int));

	cudaMemset(new_hm.values, 0, numBucketsReshape * sizeof(int));



	unsigned int no_of_threads_in_block = 256;

	unsigned int no_of_blocks = hm.size / no_of_threads_in_block;

	if(hm.size % no_of_threads_in_block != 0)

		no_of_blocks ++;



	// Launch a kernel to perform a parallel re-hash from the old table to the new one.

	thread_insert>>>(hm.keys, hm.values, hm.size, new_hm);

	cudaDeviceSynchronize();



	new_hm.current_no_of_pairs += hm.current_no_of_pairs;



	// Free the old GPU memory.

	cudaFree(hm.keys);

	cudaFree(hm.values);



	hm = new_hm;

}



/**

 * @brief Inserts a batch of key-value pairs into the hash table.

 * @return True on success.

 */

bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {

	int *device_keys, *device_values;



	unsigned int no_of_threads_in_block = 256;

	unsigned int size = numKeys * sizeof(int);

	unsigned int no_of_blocks = numKeys / no_of_threads_in_block;



	if(numKeys % no_of_threads_in_block != 0)

		no_of_blocks++;



	cudaMalloc(&device_keys, size);

	cudaMalloc(&device_values, size);



	// Check load factor and reshape if necessary before insertion.

	if (float(hm.current_no_of_pairs + numKeys) / hm.size >= 0.9)

		reshape(int((hm.current_no_of_pairs + numKeys) / 0.8));



	// Copy data from host to device.

	cudaMemcpy(device_keys, keys, size, cudaMemcpyHostToDevice);

	cudaMemcpy(device_values, values, size, cudaMemcpyHostToDevice);



	// Launch the insertion kernel.

	thread_insert>>>(device_keys, device_values, numKeys, hm);

	cudaDeviceSynchronize(); // Wait for the kernel to complete.



	hm.current_no_of_pairs += numKeys;



	cudaFree(device_keys);

	cudaFree(device_values);



	return true;

}



/**

 * @brief Retrieves values for a batch of keys.

 * @return A host-accessible pointer to an array containing the results.

 *         Note: uses `cudaMallocManaged` for unified memory.

 */

int* GpuHashTable::getBatch(int* keys, int numKeys) {



	int *device_keys;

	unsigned int no_of_threads_in_block = 256;

	unsigned int size = numKeys * sizeof(int);

	unsigned int no_of_blocks = numKeys / no_of_threads_in_block;



	if(numKeys % no_of_threads_in_block != 0)

		no_of_blocks++;



	// Copy keys from host to device.

	cudaMalloc(&device_keys, size);

	cudaMemcpy(device_keys, keys, size, cudaMemcpyHostToDevice);





	int *results;

    // Allocate managed memory for results so it's accessible from both host and device.

	cudaMallocManaged(&results, size);



	// Launch the retrieval kernel.

	thread_get>>>(device_keys, numKeys, hm, results);

	cudaDeviceSynchronize(); // Wait for completion.



	cudaFree(device_keys);

	

	return results;

}





float GpuHashTable::loadFactor() {

	if(hm.size == 0)

		return 0;

	return float(hm.current_no_of_pairs)/hm.size;

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



// A list of large prime numbers used for hashing.

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

	87686378464759llu, 110477914016779llu, 139193449418173llu, 175372756929481llu,

	220955828033581llu, 278386898836457llu, 350745513859007llu, 441911656067171llu,

	556773797672909llu, 701491027718027llu, 883823312134381llu, 1113547595345903llu,

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





/**

 * @brief The hash function used by the kernels.

 * It uses a multiplicative hashing scheme with large prime numbers.

 */

__device__ int hash1(int data, int limit) {

	return ((long)abs(data) * primeList[64]) % primeList[90] % limit;

}



/**

 * @struct Hmap

 * @brief A host-side structure holding device pointers and metadata for the hash map.

 */

typedef struct hashmap {

	int *keys;      //!< Device pointer to the keys array.

	int *values;    //!< Device pointer to the values array.

	unsigned int current_no_of_pairs; //!< The number of elements in the table.

	unsigned int size;      //!< The total capacity (number of buckets) of the table.

} Hmap;





/**

 * @class GpuHashTable

 * @brief The host-side interface for managing the GPU hash table.

 */

class GpuHashTable

{

	Hmap hm;

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



