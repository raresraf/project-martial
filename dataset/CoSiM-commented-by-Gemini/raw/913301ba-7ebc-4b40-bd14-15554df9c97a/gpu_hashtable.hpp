/**
 * @file gpu_hashtable.hpp
 * @brief Header file for the GPU-accelerated hash table.
 * @details This file defines the data structures, constants, and the host-side
 * class definition for the GpuHashTable. It includes the structures for hash
 * elements and the main hash table controller, as well as constants for load factor
 * management and macros for error handling.
 */
#ifndef _HASHCPU_
#define _HASHCPU_

using namespace std;

// Defines the value used to represent an invalid or empty key in a hash bucket.
#define	KEY_INVALID		0

/**
 * @brief A macro for fatal error checking of system calls.
 * @details If the assertion (e.g., a function call returning an error) is true,
 * it prints an error message including the file and line number, the description
 * of the call, and then exits the program with the current errno.
 */
#define DIE(assertion, call_description) 
	do {	
		if (assertion) {	
		fprintf(stderr, "(%s, %d): ",	
		__FILE__, __LINE__);	
		perror(call_description);	
		exit(errno);	
	}	
} while (0)

/**
 * @brief A list of large prime numbers.
 * @details These primes are used in the hash functions to help ensure a good distribution
 * of hash values and reduce collisions. The selection of different primes can be used to
 * create multiple independent hash functions.
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



// Example hash functions; these appear to be intended for CPU-side use but are not used in the final GPU implementation.
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
 * @struct my_hashElem
 * @brief Represents a single key-value pair in the hash table.
 * @details This structure is the fundamental unit of storage in the hash table's bucket array.
 */
typedef struct {
	int elem_value;
	int elem_key;
}my_hashElem;

/**
 * @struct my_hashTable
 * @brief The main control structure for the hash table.
 * @details This structure holds pointers to the device memory where the hash buckets are stored,
 * along with metadata such as the table's capacity and the current number of items.
 * A pointer to this struct is passed to each CUDA kernel.
 */
typedef struct {
	my_hashElem* table;          // Pointer to the array of hash elements on the GPU.
	unsigned long long max_items; // The maximum capacity of the table.
	unsigned long long curr_nr_items; // The current number of elements in the table.
}my_hashTable;

// The maximum load factor (in percent) before a reshape/rehash is triggered.
#define MAX_LOAD 85
// The minimum load factor (in percent) to aim for after a reshape.
#define MIN_LOAD 75




/**
 * @class GpuHashTable
 * @brief A host-side class to manage a GPU-based hash table.
 * @details This class provides the public API for interacting with the hash table.
 * It handles memory management (allocation/deallocation), kernel launches, and
 * data transfers between the host and device.
 */
class GpuHashTable
{
public:
	/**
	 * @brief Constructor for the GpuHashTable.
	 * @param size The initial capacity of the hash table.
	 */
	GpuHashTable(int size);

	// A pointer (on the host) to the hash table control structure in device memory.
	my_hashTable* dev_hash;
	
	// A flag used to control reshape logic.
	int OK = 0;

	/**
	 * @brief Resizes the hash table to a new capacity.
	 * @param sizeReshape The target number of elements for the new table.
	 */
	void reshape(int sizeReshape);
	
	/**
	 * @brief Checks if adding a batch would exceed the maximum load factor.
	 * @param batchSize The size of the batch to be inserted.
	 * @return True if the batch can be inserted without rehashing, false otherwise.
	 */
	bool check_l(unsigned long long batchSize);

	/**
	 * @brief Inserts a batch of key-value pairs.
	 * @param keys A host pointer to an array of keys.
	 * @param values A host pointer to an array of values.
	 * @param numKeys The number of key-value pairs in the batch.
	 * @return True on success.
	 */
	bool insertBatch(int* keys, int* values, int numKeys);

	/**
	 * @brief Retrieves a batch of values corresponding to given keys.
	 * @param key A host pointer to an array of keys to look up.
	 * @param numItems The number of keys to get.
	 * @return A host pointer to an array with the corresponding values. The caller must free this memory.
	 */
	int* getBatch(int* key, int numItems);
	
	/**
	 * @brief Calculates and returns the current load factor of the hash table.
	 * @return The load factor as a float.
	 */
	float loadFactor();
	
	void occupancy();
	void print(string info);

	/**
	 * @brief Destructor for the GpuHashTable.
	 */
	~GpuHashTable();
};

#endif
