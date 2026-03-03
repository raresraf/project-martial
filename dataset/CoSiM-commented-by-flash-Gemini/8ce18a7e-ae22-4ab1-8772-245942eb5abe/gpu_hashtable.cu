/**
 * @file gpu_hashtable.cu
 * @brief Implements a GPU-accelerated hash table using CUDA, with a C-style global state.
 *
 * This module provides a `GpuHashTable` class that offers insertion, retrieval,
 * and dynamic resizing (reshaping) capabilities for a hash table on a CUDA-enabled GPU.
 * It leverages CUDA kernels for parallel operations and atomic functions
 * for thread-safe access and collision resolution in a parallel environment.
 * The hash table employs open addressing with linear probing for collision handling.
 *
 * A notable design choice is the use of global variables (`lungime`, `numar_elemente`, `buckets`)
 * to store the hash table's state. This effectively makes the `GpuHashTable` class
 * manage a single, global hash table instance across the entire application, rather
 * than having independent hash table objects.
 *
 * Key features:
 * - Parallel `inserare` (insert), `retur` (get), and `reformare` (reshape) operations
 *   implemented as CUDA kernels.
 * - `atomicCAS` for thread-safe updates to hash table slots.
 * - Dynamic resizing of the hash table based on load factor thresholds.
 * - Host-side interface (`GpuHashTable` class) for managing GPU memory and launching kernels.
 * - Uses Romanian identifiers for some key variables and functions.
 *
 * @attention This implementation assumes integer keys and values. The global state
 *            management implies that only one active hash table exists at a time,
 *            even if multiple `GpuHashTable` objects are instantiated.
 *
 * HPC & Parallelism Considerations:
 * - Memory Hierarchy: The main hash table (`buckets`) resides in global device memory.
 * - Thread Indexing: Standard CUDA linear indexing (`threadIdx.x + blockDim.x * blockIdx.x`) is used.
 * - Synchronization: `atomicCAS` is critical for ensuring correctness during concurrent
 *   insertions and rehashing by multiple threads accessing the same global memory locations.
 *   `cudaDeviceSynchronize()` is used to ensure host-side code waits for GPU computations to complete.
 * - Collision Resolution: Linear probing is used, which can suffer from primary clustering
 *   in parallel environments if not managed carefully (e.g., through load factor control).
 * - Global State: The use of global variables for hash table parameters (`lungime`, `numar_elemente`, `buckets`)
 *   means that all `GpuHashTable` instances operate on the same underlying GPU hash table.
 */

// Standard C++ headers
#include <iostream>   // For standard input/output operations (e.g., std::cerr)
#include <vector>     // For dynamic arrays (though not directly used in the .cu file)
#include <string>     // For string manipulation (e.g., std::string)
#include <algorithm>  // For algorithms like std::max (though not explicitly used here)
#include <numeric>    // For numerical operations (though not explicitly used here)
#include <cstdio>     // For C-style input/output (e.g., fprintf, perror)
#include <cstdlib>    // For general utilities (e.g., exit, abs)
#include <cstring>    // For memory manipulation (e.g., memset)
#include <cerrno>     // For error number definitions (e.g., errno)

// CUDA specific headers
#include <cuda_runtime.h> // For CUDA runtime API functions (e.g., cudaMalloc, cudaMemcpy, cudaFree)
#include <device_launch_parameters.h> // For __global__ and __device__ keywords

// Project-specific header
#include "gpu_hashtable.hpp" // Contains declarations for GpuHashTable class, HashItem, GPU_Table structs.

/**
 * @brief Structure representing a single entry (key-value pair) in the hash table.
 * Corresponds to 'HashItem' in previous versions.
 */
typedef struct Bucket {
	int cheie;   // The key of the item (Romanian: cheie).
	int valoare; // The value associated with the key (Romanian: valoare).
} Bucket;

// Global variables representing the state of the hash table on the host side.
// These variables are shared across all instances of GpuHashTable.
int  lungime = 0;        // Current size (number of buckets) of the hash table (Romanian: lungime).
int numar_elemente = 0; // Current number of elements stored in the hash table (Romanian: numar_elemente).
Bucket *buckets;         // Pointer to the array of Bucket structs in GPU global memory.


/**
 * @brief CUDA kernel to perform parallel insertions or updates in the hash table.
 * (Romanian: inserare - insertion)
 *
 * Each thread in the grid is responsible for inserting or updating a specific key-value pair.
 * It uses open addressing with linear probing and `atomicCAS` for thread-safe operations.
 * If the key already exists, its value is updated. If a slot is empty, the key is inserted.
 *
 * @param buckets Pointer to the hash table (array of Bucket structs) on the device (global memory).
 * @param marime The number of keys to process in this batch (Romanian: marime - size/quantity).
 * @param lungime The current total size (number of buckets) of the hash table (Romanian: lungime).
 * @param chei Pointer to an array of keys to insert/update on the device (global memory) (Romanian: chei).
 * @param valori Pointer to an array of values corresponding to the keys (global memory) (Romanian: valori).
 *
 * HPC & Parallelism:
 * - Thread Indexing: `id_thread = threadIdx.x + blockDim.x * blockIdx.x` computes a unique linear thread ID.
 * - Memory Access: Accesses `buckets`, `chei`, `valori` in global device memory.
 * - Synchronization: `atomicCAS` is crucial here. It atomically compares a memory location's
 *   value with an expected value (0 for empty) and, if they match, swaps it with a new key.
 *   This ensures that only one thread successfully writes to an empty slot or updates a key's
 *   value without race conditions.
 * - Collision Resolution: Linear probing means threads might contend for slots or follow
 *   long chains. `atomicCAS` helps manage concurrent access to these slots.
 */
__global__ void inserare(Bucket *buckets, int marime, int lungime, int *chei, int *valori) {
	unsigned int id_thread = threadIdx.x + blockDim.x * blockIdx.x; // Unique linear thread ID.

	if (id_thread < marime) { // Process only active threads within the batch size.
        // Compute the initial hash position for the key.
        // Uses a specific prime number (163307llu) for hashing.
		long pozitie = ((long)abs(chei[id_thread]) * 163307llu) % 668993977llu % lungime;

		int cheie_n = chei[id_thread]; // Current key to insert.
		int cheie_v; // Variable to store the old key during atomicCAS.

        // Block Logic: Linear probing search and insertion.
        // Loop continues until an empty slot is found (cheie_v == 0) or the key already exists (cheie_v == cheie_n).
        // Invariant: `pozitie` moves linearly through the hash table slots, wrapping around if needed.
		while ((cheie_v = atomicCAS(&(buckets[pozitie].cheie), 0, cheie_n)) != 0 && cheie_v != cheie_n ) {
			pozitie++; // Move to the next slot (linear probing).
			if (pozitie == lungime) {
				pozitie = 0; // Wrap around to the beginning of the table.
			}
		}
        // Once an appropriate slot is found or created, assign the value.
		buckets[pozitie].valoare = valori[id_thread];
	}
	return;
}

/**
 * @brief CUDA kernel to reshape (rehash) elements from an old hash table to a new one.
 * (Romanian: reformare - reshaping/reforming)
 *
 * Each thread in the grid is responsible for taking one item from the old hash table
 * and inserting it into the new hash table. This is used during resizing operations.
 * It uses open addressing with linear probing and `atomicCAS` to handle collisions
 * and ensure thread-safe insertion into the new table.
 *
 * @param buckets Pointer to the old hash table (array of Bucket structs) on the device.
 * @param new_buckets Pointer to the new, larger hash table on the device.
 * @param marime The number of elements to rehash from the old table (Romanian: marime).
 * @param lungime The total size (number of buckets) of the new hash table (Romanian: lungime).
 *
 * HPC & Parallelism:
 * - Thread Indexing: `id_thread = threadIdx.x + blockDim.x * blockIdx.x` computes a unique linear thread ID.
 * - Memory Access: Accesses `buckets` (old table) and `new_buckets` (new table) in global device memory.
 * - Synchronization: `atomicCAS` is essential for collision resolution when multiple
 *   threads attempt to insert their old items into potentially the same new hash table slots.
 *   This ensures only one thread succeeds in writing to an empty slot.
 * - Collision Resolution: Linear probing is used, similar to `inserare` kernel.
 */
__global__ void reformare(Bucket *buckets, Bucket *new_buckets ,int marime, int lungime) {
	unsigned int id_thread = threadIdx.x + blockDim.x * blockIdx.x; // Unique linear thread ID.

	if (id_thread < marime) { // Process only active threads within the batch size (marime here refers to old_table_size).

    // Compute the initial hash position for the old item's key in the new hash table.
    // Uses a specific prime number (163307llu) for hashing.
	long pozitie = ((long)abs(buckets[id_thread].cheie) * 163307llu) % 668993977llu % lungime;
	int cheie_v; // Variable to store the old key during atomicCAS.

    // Guard clause: only process valid (non-empty) keys from the old table.
	if (buckets[id_thread].cheie == 0) // Assuming KEY_INVALID (0) means empty.
		return;

    // Block Logic: Linear probing search and insertion into the new hash table.
    // Loop continues until an empty slot is found (cheie_v == 0) or the key already exists (cheie_v == old_key).
	while ((cheie_v = atomicCAS(&(new_buckets[pozitie].cheie), 0, buckets[id_thread].cheie)) != 0 && cheie_v != buckets[id_thread].cheie ) {
		pozitie++; // Move to the next slot (linear probing).
		if (pozitie == lungime) {
			pozitie = 0; // Wrap around to the beginning of the table.
		}
	}
    // Once an appropriate slot is found or created, assign the value.
	new_buckets[pozitie].valoare = buckets[id_thread].valoare;
	return;
	}
}


/**
 * @brief CUDA kernel to perform parallel lookups (get operations) in the hash table.
 * (Romanian: retur - return/get)
 *
 * Each thread in the grid is responsible for looking up a specific key.
 * It uses open addressing with linear probing to find the key.
 *
 * @param buckets Pointer to the hash table (array of Bucket structs) on the device (global memory).
 * @param marime The number of keys to lookup in this batch (Romanian: marime).
 * @param lungime The current total size (number of buckets) of the hash table (Romanian: lungime).
 * @param chei Pointer to an array of keys to lookup on the device (global memory) (Romanian: chei).
 * @param valori Pointer to an array where found values will be stored on the device (global memory) (Romanian: valori).
 *
 * HPC & Parallelism:
 * - Thread Indexing: `id_thread = threadIdx.x + blockDim.x * blockIdx.x` computes a unique linear thread ID.
 * - Memory Access: Accesses `buckets`, `chei`, `valori` in global device memory.
 * - Collision Resolution: Linear probing is implemented in a loop, potentially leading to
 *   divergence if threads encounter different collision chains.
 */
__global__ void retur(Bucket *buckets, int marime, int lungime, int *chei, int *valori) {
	unsigned int id_thread = threadIdx.x + blockDim.x * blockIdx.x; // Unique linear thread ID.

	if (id_thread < marime) { // Process only active threads within the batch size.
        // Compute the initial hash position for the key.
        // Uses a specific prime number (163307llu) for hashing.
		long pozitie = ((long)abs(chei[id_thread]) * 163307llu) % 668993977llu % lungime;
		int cheie_c = chei[id_thread]; // Current key to find.

        // Block Logic: Linear probing search for the key.
        // Loop continues until the key is found.
        // Invariant: `pozitie` moves linearly through the hash table slots, wrapping around if needed.
		while (buckets[pozitie].cheie != cheie_c) {
			pozitie++; // Move to the next slot (linear probing).
			if (pozitie == lungime) {
				pozitie = 0; // Wrap around to the beginning of the table.
			}
		}
        // Once the key is found, store its value.
		valori[id_thread] = buckets[pozitie].valoare;
		return;
	}
	return;
}


/**
 * @brief Host-side class providing an interface to a GPU-accelerated hash table.
 *
 * This class primarily manages GPU memory for the hash table and orchestrates the
 * launching of CUDA kernels for batch operations (insert, get) and reshaping.
 * It operates on global hash table state variables (`lungime`, `numar_elemente`, `buckets`).
 */
class GpuHashTable
{
	// No member variables here as the hash table state (`lungime`, `numar_elemente`, `buckets`)
	// is managed via global variables. This implies a single, global hash table.

	public:
		/**
		 * @brief Constructor for GpuHashTable.
		 * Allocates and initializes GPU memory for the global hash table state.
		 *
		 * @param size The desired initial size (number of buckets) for the hash table.
		 */
		GpuHashTable(int size) {
            // Allocate global device memory for the hash table (array of Bucket structs).
			cudaMalloc((void **) &(buckets), size * sizeof(Bucket));
            // Initialize all hash table slots by setting their keys to 0 (KEY_INVALID implicitly).
			cudaMemset(buckets, 0, size * sizeof(Bucket));

			lungime = size; // Update the global hash table size.
		}

		/**
		 * @brief Destructor for GpuHashTable.
		 * Frees the GPU memory allocated for the global hash table.
		 */
		~GpuHashTable() {
			cudaFree(buckets); // Free the global device memory.
		}

		/**
		 * @brief Reshapes (resizes) the global hash table to accommodate more elements.
		 *
		 * This function checks the load factor of the global hash table. If it exceeds
		 * a threshold (0.8), it calculates a new, larger size, allocates new GPU memory,
		 * transfers existing elements using the `reformare` CUDA kernel, and then frees
		 * the old GPU memory.
		 *
		 * @param numBucketsReshape (Ignored in this implementation) The new size is
		 *                          calculated based on the current number of elements
		 *                          and a fixed load factor. This parameter's value is not used.
		 */
		void reshape(int numBucketsReshape) { // numBucketsReshape is an unused parameter in this implementation.
            // BUG: `numar_elemente` is updated here, but `numar_elemente` is not consistently updated
            // by `insertBatch` in this implementation. This will lead to incorrect load factor calculations.
			numar_elemente += numBucketsReshape; // This line seems to incorrectly add the reshape parameter to total elements.
			int lungime_v = lungime; // Store current hash table size.

            // Check load factor: if `numar_elemente` / `lungime_v` > 0.8, trigger reshape.
			if (numar_elemente / lungime_v > 0.8) {
                // Calculate new hash table size based on current elements to maintain a load factor of ~0.8.
				lungime = numar_elemente * 10 / 8;

				Bucket *new_buckets;
                // Allocate global device memory for the new hash table.
				cudaMalloc((void **) &(new_buckets),lungime * sizeof(Bucket));
                // Initialize all slots in the new hash table to 0.
				cudaMemset(new_buckets, 0, lungime * sizeof(Bucket));

                // Determine grid configuration for kernel launch.
				const size_t block_size = 1024;
				size_t blocks_no = ceil((float)lungime_v / (float)block_size); // Uses old size to determine blocks for rehash.
 
                // Launch the `reformare` kernel to transfer elements from old `buckets` to `new_buckets`.
  				reformare<<>>(buckets, new_buckets, lungime_v, lungime);
  				cudaDeviceSynchronize(); // Synchronize device to ensure kernel completion.
	 
  				cudaFree(buckets); // Free the global device memory of the old hash table.

  				buckets = new_buckets; // Update global pointer to point to the new hash table.
			}
		}

		/**
		 * @brief Inserts a batch of key-value pairs into the global hash table on the GPU.
		 *
		 * This function handles memory allocation for keys/values on the device,
		 * performs data transfer from host to device, calls `reshape` (which
		 * checks load factor and resizes if needed), launches the `inserare` kernel,
		 * and finally frees temporary device memory.
		 *
		 * @param keys Pointer to an array of keys on the host to insert.
		 * @param values Pointer to an array of values on the host to insert.
		 * @param numKeys The number of key-value pairs in the batch.
		 * @return true on success.
		 */
		bool insertBatch(int *keys, int* values, int numKeys) {
			int *chei, *valori; // Pointers for keys and values arrays on the device.

            // Allocate global device memory for the keys and values arrays.
			cudaMalloc((void **) &(chei), numKeys * sizeof(int));
			cudaMalloc((void **) &(valori), numKeys * sizeof(int));

            // Copy keys and values from host memory to device global memory.
			cudaMemcpy(chei, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(valori, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

			reshape(numKeys); // Call reshape (which might resize based on global `numar_elemente`).
                              // Note: `numar_elemente` is not correctly updated by `insertBatch` itself,
                              // which could lead to incorrect reshape decisions.

            // Determine grid configuration for kernel launch.
			const size_t block_size = 1024;
			size_t blocks_no = ceil((float)numKeys / (float)block_size);

            // Launch the `inserare` kernel to perform parallel insertions/updates.
  			inserare<<>>(buckets, numKeys, lungime, chei, valori);
  			
            // Free the temporary global device memory allocated for keys and values.
  			cudaFree(chei);
  			cudaFree(valori);

			return true;
		}

		/**
		 * @brief Retrieves a batch of values for given keys from the global hash table on the GPU.
		 *
		 * This function allocates device memory for keys and values, transfers keys
		 * from host to device, launches the `retur` kernel, copies results back to
		 * the host, and finally frees temporary device memory.
		 *
		 * @param keys Pointer to an array of keys on the host to lookup.
		 * @param numKeys The number of keys in the batch.
		 * @return int* A pointer to an array of retrieved values on the host.
		 *         The caller is responsible for freeing this host memory using `free()`.
		 */
		int* getBatch(int* keys, int numKeys) {
			int *chei, *valori, *rezultate; // Pointers for device keys, values, and host results.

            // Allocate host memory for the retrieved values.
			rezultate = (int *)malloc(numKeys * sizeof(int));

            // Allocate global device memory for the keys and values arrays.
			cudaMalloc((void **) &(chei), numKeys * sizeof(int));
			cudaMalloc((void **) &(valori), numKeys * sizeof(int));

            // Copy keys from host memory to device global memory.
			cudaMemcpy(chei, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
			
            // Determine grid configuration for kernel launch.
			const size_t block_size = 512;
			size_t blocks_no = ceil((float)numKeys / (float)block_size);
  	
            // Launch the `retur` kernel to perform parallel lookups.
  			retur<<>>(buckets, numKeys, lungime, chei, valori);

            // Synchronize the device and copy results from device to host.
  			cudaMemcpy(rezultate, valori, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

            // Free the temporary global device memory allocated for keys and values.
			cudaFree(chei);
			cudaFree(valori);

			return rezultate;
		}


		/**
		 * @brief Calculates the current load factor of the global hash table.
		 *
		 * The load factor is the ratio of the number of elements (`numar_elemente`)
		 * to the total number of available hash table slots (`lungime`).
		 *
		 * @return float The current load factor.
		 */
		float loadFactor() {
			return (float)numar_elemente / (float)lungime; 
		}

		// Placeholder for future implementation.
		void occupancy();
		// Placeholder for future implementation.
		void print(string info);
};

// --- Convenience Macros for GpuHashTable Operations ---
// These macros provide a simplified interface to interact with the global GpuHashTable.
// Note: This pattern often wraps the GpuHashTable object within the macro.
#define HASH_INIT GpuHashTable GpuHashTable(1); // Initializes a GpuHashTable object named `GpuHashTable` with size 1.
#define HASH_RESERVE(size) GpuHashTable.reshape(size); // Reshapes the `GpuHashTable` (uses global state).

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys) // Inserts a batch of items.
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys) // Retrieves a batch of values.

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor() // Gets the current load factor.

// This include likely brings in test definitions for the hash table.
// In competitive programming or library development, tests might be
// included directly in the source file or a separate compilation unit.
#include "test_map.cpp"

// --- Global Constants and Helper Functions (ifndef block from gpu_hashtable.hpp expected) ---
// These definitions are typically found in a header file (e.g., gpu_hashtable.hpp)
// and are duplicated or fully defined here if the header is not included
// or for standalone compilation.
#ifndef _HASHCPU_ // Include guard to prevent multiple definitions.
#define _HASHCPU_

using namespace std; // Using standard namespace.

#define	KEY_INVALID		0 // Macro representing an invalid or empty key in a hash table slot.

/**
 * @brief Macro for error checking and program termination.
 * @param assertion A boolean expression. If true, an error is reported and the program exits.
 * @param call_description A string describing the failed call.
 */
#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)
	
// A list of prime numbers, potentially used for hash functions or to find
// suitable sizes for hash tables during resizing.
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
