/**
 * @file gpu_hashtable.cu
 * @brief A GPU-accelerated hash table implementation using CUDA.
 * @details This file defines a hash table that resides in GPU memory and is manipulated
 * through CUDA kernels. It uses linear probing with wrap-around for collision resolution
 * and atomic operations for thread safety.
 */
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define plus10percent(x) (int)((double)1.1*(double)x)

/**
 * @brief Computes a hash for a given integer key on the device.
 * @param data The integer key to hash.
 * @param limit The size of the hash table, used to scale the hash value.
 * @return The computed hash value.
 */
__global__ void insert(int *keys, int* values, int numKeys, 
                        dataType *data, int *max_size, int *crt_size) {

    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    
    if (i < numKeys) {

        
        int position = myHash(keys[i], *max_size);



        while(1){

            
            // If the key already exists, update its value.
            if(data[position].key == keys[i]){
                atomicExch(&data[position].value, values[i]);
                return;
            }

            
            // Attempt to insert the key in an empty slot.
            if(atomicCAS(&data[position].key, 0, keys[i]) == 0){

                atomicExch(&data[position].value, values[i]);
                atomicAdd(crt_size, 1);
                return;
            }

            
            // Linear probing: move to the next slot.
            position++;
            
            // Wrap around if the end of the table is reached.
            if(position >= *max_size){
                position = 0;
            }
        }
    }
}




/**
 * @brief CUDA kernel to retrieve values for a batch of keys.
 * @param keys Pointer to an array of keys to look up.
 * @param values Pointer to an array to store the retrieved values.
 * @param numKeys The number of keys to look up.
 * @param data Pointer to the hash table data.
 * @param max_size Pointer to the maximum size of the hash table.
 * @details Each thread searches for a key using linear probing with wrap-around.
 * If the key is found, the corresponding value is written to the output array;
 * otherwise, -1 is written.
 */
__global__ void get_values(int *keys, int* values, int numKeys, 
                            dataType *data, int *max_size) {

    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    
    if (i < numKeys) {

        
        int position = myHash(keys[i], *max_size);
        int first_position = position;

        while(1){
            
            // If the key is found, store the value and exit.
            if(data[position].key == keys[i]){
                atomicExch(&values[i], data[position].value);
                return;
            }

            position++;
            // Wrap around at the end of the table.
            if(position >= *max_size){
                position = 0;
            }

            
            // If the whole table has been searched, the key is not present.
            if(position == first_position){
                atomicExch(&values[i], -1);
                return;
            }
        }
    }
}


/**
 * @brief Constructs a GpuHashTable object.
 * @param size The initial capacity of the hash table.
 * @details Allocates managed memory on the GPU for the hash table data and its metadata,
 * allowing direct access from both host and device.
 */
GpuHashTable::GpuHashTable(int size) {

    data = 0;
    max_size = 0;
    crt_size = 0;
    
    
    // Allocate managed memory for the hash table structure and its data.
    cudaMallocManaged(&data, size * sizeof(dataType));
    cudaMallocManaged(&max_size, sizeof(int));
    cudaMallocManaged(&crt_size, sizeof(int));

    if(data == 0 || max_size == 0 || crt_size == 0){
	   printf("eroare init\n");
	   return;
    }

    
    // Initialize the hash table memory.
    memset(data, 0, size * sizeof(dataType));
    *max_size = size;
    *crt_size = 0;

    if(cudaGetLastError() != 0){
        printf("%s\n", cudaGetErrorName(cudaGetLastError()));
        exit(-1);
    }
}


/**
 * @brief Destroys the GpuHashTable object.
 * @details Frees the GPU memory allocated for the hash table.
 */
GpuHashTable::~GpuHashTable() {

    cudaFree(data);   

}


/**
 * @brief Resizes the hash table to a new capacity.
 * @param numBucketsReshape The new capacity of the hash table.
 * @details This function allocates a new, larger hash table and rehashes the elements
 * from the old table to the new one. The rehashing process is done on the host by
 * extracting all elements and then launching a kernel to insert them into the new table.
 */
void GpuHashTable::reshape(int numBucketsReshape) { 

    
    // Allocate a new table in managed memory.
    dataType *new_data;
    cudaMallocManaged(&new_data, numBucketsReshape * sizeof(dataType));

    if(new_data == NULL){
        printf("eroare reshape\n");
        return;
    }
    memset(new_data, 0, numBucketsReshape * sizeof(dataType));

    
    if(*crt_size == 0){
        cudaFree(data);
        data = new_data;
        *max_size = numBucketsReshape;
        return;
    }

    int numKeys = (*crt_size);

    
    const size_t block_size = 256;
    size_t blocks_no = numKeys / block_size;

    if (numKeys % block_size) 
        ++blocks_no;

    
    // Temporary storage for keys and values.
    int *keys, *values;
    cudaMallocManaged(&keys, numKeys * sizeof(int));
    cudaMallocManaged(&values, numKeys * sizeof(int));

    int idx = 0;


    // Extract key-value pairs from the old table (host-side).
    for(int i = 0; i < (*max_size); i++){
        if(data[i].key != 0){
            keys[idx] = data[i].key;
            values[idx] = data[i].value;
            idx++;
        }
    }

    (*max_size) = numBucketsReshape;
    (*crt_size) = 0;


    // Re-insert the elements into the new table using a kernel.
    insert>> (keys, values, numKeys, 
                                        new_data, max_size, crt_size);
    cudaDeviceSynchronize();

    cudaFree(data);
    cudaFree(keys);
    cudaFree(values);
    data = new_data;
    *max_size = numBucketsReshape;
    *crt_size = numKeys;

    if (cudaGetLastError() != 0){
        printf("%s\n", cudaGetErrorName(cudaGetLastError()));
        exit(-1);
    }
}


/**
 * @brief Inserts a batch of key-value pairs into the hash table.
 * @param keys An array of keys to insert.
 * @param values An array of values to insert.
 * @param numKeys The number of keys and values in the batch.
 * @return True if the insertion was successful, false otherwise.
 * @details If the table is close to full, it is resized. Then, keys and values are
 * copied to the GPU, and the insertion kernel is launched.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {

    int *device_keys = 0;
    int *device_values = 0;
    int num_bytes = numKeys * sizeof(int);

    
    const size_t block_size = 256;
    size_t blocks_no = numKeys / block_size;

    if (numKeys % block_size) 
    	++blocks_no;

    
    // Reshape if load factor is too high.
    if(numKeys >= *max_size - *crt_size){
        int x = numKeys + *crt_size;
        
        reshape(plus10percent(x));
    }

    cudaMalloc((void **) &device_keys, num_bytes);
    cudaMalloc((void **) &device_values, num_bytes);

    if (device_values == 0 || device_keys == 0) {
	    	printf("[HOST] Couldn't allocate memory\n");
    		return 1;
    }

    
    // Copy keys and values from host to device.
    cudaMemcpy(device_keys, keys, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_values, values, num_bytes, cudaMemcpyHostToDevice);

    
    // Launch the insertion kernel.
    insert>> (device_keys, device_values, numKeys, 
                                        data, max_size, crt_size);
    cudaDeviceSynchronize();

    
    cudaFree(device_keys);
    cudaFree(device_values);

    if (cudaGetLastError() != 0){
        printf("%s\n", cudaGetErrorName(cudaGetLastError()));
        exit(-1);
    }

    return true;
}


/**
 * @brief Retrieves values for a batch of keys.
 * @param keys An array of keys to look up.
 * @param numKeys The number of keys in the batch.
 * @return A pointer to a host array containing the retrieved values. The caller must free this memory.
 * @details This function copies the keys to the GPU, launches a retrieval kernel, and copies
 * the results back to the host.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {

    int *values;
    int *device_values;
    int *device_keys;
    int num_bytes = numKeys * sizeof(int);

    
    const size_t block_size = 256;
    size_t blocks_no = numKeys / block_size;

    if (numKeys % block_size) 
	   ++blocks_no;

    
    values = (int*)malloc(numKeys*sizeof(int));
    if(values == NULL){
    	printf("eroare malloc in getBatch\n");
    	return NULL;
    }

    
    // Allocate managed memory for device-side operations.
    cudaMallocManaged(&device_keys, num_bytes);
    cudaMallocManaged(&device_values, num_bytes);

    if(device_keys == NULL || device_values == NULL){
    	printf("[HOST] Couldn't allocate memory\n");
    	return NULL;
    }

    memcpy(device_keys, keys, num_bytes);

    
    // Launch the retrieval kernel.
    get_values>> (device_keys, device_values, 
                                            numKeys, data, max_size);
    cudaDeviceSynchronize();

    
    // Copy results back to host.
    memcpy(values, device_values, num_bytes);

    cudaFree(device_keys);
    cudaFree(device_values);

    return values;
}



/**
 * @brief Calculates the load factor of the hash table.
 * @return The current load factor (ratio of current size to maximum size).
 */
float GpuHashTable::loadFactor() {
    return (float)(*crt_size)/(float)(*max_size);
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

__device__ int myHash(int data, int limit){
	
 	size_t prime1 = 41812097llu;
 	size_t prime2 = 226258767906406483llu;

	return ((long)abs(data) * prime1) % prime2 % limit;
}

typedef struct {
	int key;
	int value;
} dataType;




class GpuHashTable
{
	public:
		
		dataType* data;
		int *max_size;
		int *crt_size;

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
