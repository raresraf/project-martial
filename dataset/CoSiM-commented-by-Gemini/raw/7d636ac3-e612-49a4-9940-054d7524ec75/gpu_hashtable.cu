/**
 * @file gpu_hashtable.cu
 * @brief Implements a concurrent, open-addressing hash table on the GPU using CUDA.
 *
 * This implementation uses linear probing for collision resolution and atomic
 * operations to ensure thread-safe insertions and lookups, making it suitable
 * for highly parallel environments. The hash table supports dynamic resizing
 * (rehashing) to maintain performance as the load factor increases.
 *
 * @note The file includes an unusual structure with a C-style API wrapper
 * and a re-definition of the class and helpers within an #ifndef block.
 */

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"
#define UNDEFINED 	0
#define HASH_A 		2074129llu
#define HASH_B 		1061961721llu


/**
 * @brief CUDA kernel for inserting a batch of key-value pairs into the hash table.
 * @param map_stored Pointer to the hash table's storage (device memory).
 * @param keys Array of keys to insert (device memory).
 * @param values Array of values to insert (device memory).
 * @param stored_pairs A single-element array (counter) for the number of stored pairs (device memory).
 * @param map_size The total capacity of the hash table.
 * @param total The number of key-value pairs to process in this batch.
 *
 * Each thread processes one key-value pair. It computes the initial hash index
 * and then uses linear probing to find an available slot. The `atomicCAS`
 * (Compare-and-Swap) operation is used to ensure that only one thread can claim
 * an empty slot or update an existing key, thus preventing race conditions.
 */
__global__ void kernel_insert(Pair* map_stored, int* keys, int *values,
		int *stored_pairs, int map_size, int total) 
{
	// Assign a unique ID to each thread, corresponding to an element in the input arrays.
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int hashed_key;
	int key;
	int value;
	
	if (i < total) {
		
		// Hash function: Multiplicative hashing with large primes.
		hashed_key = ((long)abs(keys[i]) * HASH_A) % HASH_B % map_size;
		key = keys[i];
		value = values[i];
		
		// Begin linear probing sequence.
		while (true) {
			
			// Atomically attempt to claim a slot.
			// If map_stored[hashed_key].key is UNDEFINED, set it to `key`.
			int prev = atomicCAS(&map_stored[hashed_key].key, UNDEFINED, key);

			// Case 1: The slot was empty (UNDEFINED).
			// The thread successfully claimed it.
			if (prev == UNDEFINED) {
				atomicAdd(&stored_pairs[0], 1); // Increment the total count of stored pairs.
				map_stored[hashed_key].value = value; // Set the value.
				return;
			// Case 2: The key already exists.
			// Update the value associated with the key.
			} else if (prev == key) {
				map_stored[hashed_key].value = value;
				return;
			}

			// Case 3: Collision. The slot is occupied by a different key.
			// Move to the next slot in the linear probe sequence.
			hashed_key = (hashed_key + 1) % map_size;
		}
	}
}


/**
 * @brief CUDA kernel to rehash elements from an old table into a new, larger one.
 * @param map_stored Pointer to the new hash table's storage (device memory).
 * @param old_map Pointer to the old hash table's storage (device memory).
 * @param map_size The capacity of the new hash table.
 * @param total The capacity of the old hash table.
 *
 * This kernel is called during the `reshape` operation. Each thread takes an
 * element from the old map and re-inserts it into the new map using the same
 * linear probing and atomic logic as `kernel_insert`.
 */
__global__ void kernel_insert_reshape(Pair *map_stored, Pair* old_map,
		int map_size, int total)
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int hashed_key;
	int key;
	int value;
	if (i < total) {
		// Skip empty slots in the old map.
		if (old_map[i].key == UNDEFINED) {
			return;
		}

		// Re-hash the key for the new table size.
		hashed_key = ((long)abs(old_map[i].key) * HASH_A) % HASH_B % map_size;
		key = old_map[i].key;
		value = old_map[i].value;

		// Begin linear probing in the new table.
		while (true) {
			
			// Atomically claim a slot in the new table.
			int prev = atomicCAS(&map_stored[hashed_key].key, UNDEFINED, key);

			// If the slot was empty, the insertion is successful.
			if (prev == UNDEFINED) {
				map_stored[hashed_key].value = value;

				return;
			}
			// If a collision occurs, probe the next slot.
			hashed_key = (hashed_key + 1) % map_size;
		}
	}
}




/**
 * @brief CUDA kernel to look up a batch of keys in the hash table.
 * @param keys Array of keys to look up (device memory).
 * @param values Array where the found values will be stored (device memory).
 * @param stored_map Pointer to the hash table's storage (device memory).
 * @param map_size The capacity of the hash table.
 * @param total The number of keys to look up.
 *
 * Each thread searches for one key. It computes the initial hash index and
 * follows the linear probing path until the key is found. It assumes all
 * keys exist in the table.
 */
__global__ void kernel_lookup(int* keys, int* values, Pair* stored_map,
		int map_size, int total) 
{
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int hashed_key;
	unsigned int idx;
	
	if (i < total) {
		// Calculate the initial position using the same hash function as insertion.
		hashed_key = ((long)abs(keys[i]) * HASH_A) % HASH_B % map_size;
		idx = hashed_key;
		
		// Traverse the linear probe sequence.
		while (true) {
			// If the key is found, store its value and exit.
			if (stored_map[idx].key == keys[i]) {
				values[i] = stored_map[idx].value; 
				break;
			} else {
				// Move to the next slot.
				idx = (idx + 1) % map_size;
			}
		}
	}
}

/**
 * @brief Constructs a GpuHashTable.
 * @param size Initial capacity of the hash table.
 */
GpuHashTable::GpuHashTable(int size) {
	this->size = size;

	// Allocate storage for key-value pairs on the GPU.
	cudaMalloc((void **) &map, size * sizeof(Pair));
	DIE(map == NULL, "cudaMalloc");

	// Allocate a counter for the number of elements on the GPU.
	cudaMalloc((void **) &stored_pairs, sizeof(int));
	DIE(stored_pairs == NULL, "cudaMalloc");
	
	// Initialize the counter and the map memory to zero (UNDEFINED).
	cudaMemset(stored_pairs, 0, sizeof(int));
	cudaMemset(map, 0, size * sizeof(Pair));
}

/**
 * @brief Destroys the GpuHashTable, freeing all allocated GPU memory.
 */
GpuHashTable::~GpuHashTable() {
	
	cudaFree(stored_pairs);
	cudaFree(map);
}

/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The new capacity of the hash table.
 *
 * This function orchestrates the rehashing process by allocating a new, larger
 * table and launching `kernel_insert_reshape` to migrate all elements.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int prev_size;
	Pair *new_map;
	Pair *prev_map;

	
	// Allocate new, larger storage on the GPU.
	cudaMalloc((void **)&new_map, numBucketsReshape * sizeof(Pair));
	DIE(new_map == NULL, "cudaMalloc realloc new values");
	cudaMemset(new_map, 0, size * sizeof(Pair));
	
	
	// Swap the old map with the new one.
	prev_map = this->map;
	prev_size = this->size;
	
	this->map = new_map;

	// Launch the reshape kernel to re-hash elements into the new map.
	kernel_insert_reshape>>(new_map, prev_map, numBucketsReshape, prev_size);
	cudaDeviceSynchronize();

	// Free the old map storage.
	cudaFree(prev_map);
	this->size = numBucketsReshape;

}

/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys Pointer to an array of keys on the host.
 * @param values Pointer to an array of values on the host.
 * @param numKeys The number of pairs to insert.
 * @return Returns true on success.
 *
 * Before insertion, it checks the load factor. If adding the new keys would
 * exceed a 90% load factor, it triggers a `reshape` operation to grow the table.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys;
	int *device_values;
	int *host_stored_pairs;
	Pair *host_pairs;

	host_stored_pairs = (int *)malloc(sizeof(int));
	cudaMemcpy(host_stored_pairs, stored_pairs, sizeof(int), cudaMemcpyDeviceToHost);
	
	// Check if the load factor will exceed 0.9. If so, resize the table.
	if ((float)(host_stored_pairs[0] + numKeys) > 0.9 * (float)this->size) {
		// Grow the table by a factor of 1.2 relative to the new required size.
		reshape((host_stored_pairs[0] + numKeys) * 1.2);
	}
	host_pairs = (Pair *) malloc(this->size * sizeof(Pair));
	DIE(host_pairs == NULL, "malloc pairs");
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	DIE(device_values == NULL, "cudaMalloc");
	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	DIE(device_keys == NULL, "cudaMalloc");

	
	// Copy data from host to device for processing.
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Launch the insertion kernel.
	kernel_insert>>(this->map, device_keys, device_values, stored_pairs, this->size, numKeys);
	cudaDeviceSynchronize();

	free(host_pairs);
	cudaFree(host_stored_pairs);
	cudaFree(device_keys);
	cudaFree(device_values);
	return true;
}

/**
 * @brief Retrieves the values for a batch of keys.
 * @param keys Pointer to an array of keys on the host.
 * @param numKeys The number of keys to look up.
 * @return A host pointer to an array of found values. The caller is responsible for freeing this memory.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *values;
	int *device_keys;
	int *device_values;

	values = (int *) malloc(numKeys * sizeof(int));
	DIE(values == NULL, "malloc values get batch");
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	DIE(device_values == NULL, "cudaMalloc device values get batch");
	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	DIE(device_keys == NULL, "cudaMalloc device keys get batch");
	
	
	// Copy keys to device and launch the lookup kernel.
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	kernel_lookup>>(device_keys, device_values, this->map, this->size, numKeys);
	cudaDeviceSynchronize();

	// Copy the results back to the host.
	cudaMemcpy(values, device_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(device_values);
	cudaFree(device_keys);
	return values;
}

/**
 * @brief Calculates the current load factor of the hash table.
 * @return The load factor as a float (number of stored pairs / capacity).
 */
float GpuHashTable::loadFactor() {
	int *host_stored_pairs;
	
	host_stored_pairs = (int *)malloc(sizeof(int));
	// Copy the count of stored pairs from device to host.
	cudaMemcpy(host_stored_pairs, stored_pairs, sizeof(int), cudaMemcpyDeviceToHost);
	return (float) host_stored_pairs[0] / (float) size;
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

typedef struct {
	int key;
	int value;
} Pair;




class GpuHashTable
{
	public:
		int size;
		int *stored_pairs;
		Pair *map;

		int *stored_keys;
		int *stored_values;

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

