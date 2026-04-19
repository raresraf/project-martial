


#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"


/**
 * @brief Constructor: Initializes the device-side hash table.
 * 
 * @param siz Requested initial capacity.
 */
GpuHashTable::GpuHashTable(int siz) {
	if (siz <= 0){
		std::cerr <<"size < 0\n";
		exit(-1);
	}
	int err;

	
	inserari = 0;
	size = siz;
	
	// Memory Hierarchy: Global memory allocation for the entry array.
	err = cudaMalloc(&vect, size * sizeof(Celula));
	if (err != cudaSuccess || vect == NULL){
		std::cerr << "Memoria init\n";
		exit(-1);
	}
	err = cudaMemset(vect, 0, size * sizeof(Celula));
	if (err != cudaSuccess) {
		std::cerr<<"Memset\n";
		exit(-1);
	}
}



/**
 * @brief Destructor: Frees allocated device resources.
 */
GpuHashTable::~GpuHashTable() {
	int err;
	
	
	err = cudaFree(vect);
	if (err != cudaSuccess)
		exit(-1);
}



/**
 * @brief Device-side hash function using prime constant modular arithmetic.
 */
__device__ int dhash1(int data, int limit) {


	return ((long long)abs(data) * 10453007) % 3452434812973 % limit;
}


/**
 * @brief CUDA kernel for parallel data migration.
 * 
 * Functional Utility: Re-hashes valid entries from the old table into a 
 * new memory buffer using a two-pass circular probe.
 */
__global__ void reshapeAux(Celula *vect, Celula *newvect, int size, int newsize) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j;
	
	if (i < size && vect[i].cheie != 0) {
		int cheie = vect[i].cheie;
		
		int begin = dhash1(cheie, newsize), cc;
		/**
		 * Block Logic: Forward probe pass (hash to end).
		 */
		for (j = begin; j < newsize; j++){
			
			
			cc = atomicCAS(&newvect[j].cheie, 0, cheie);
			if (cc == 0){
				
				newvect[j].valoare = vect[i].valoare;
				return;
			}
		}
		
		
		/**
		 * Block Logic: Wrap-around probe pass (start to hash).
		 */
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
 * @brief Dynmically resizes the table to maintain low collision probability.
 * 
 * Logic: Implements proactive capacity management by migrating elements to 
 * a new device buffer using parallel re-hashing.
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	if(numBucketsReshape <= 0)
		exit(-1);
	if(vect == NULL){
		std::cerr << "null vect\n";
		exit(-1);
	}
	int err;
	size_t block_size = 1024, num_blocks;
	Celula *newvect;
	
	
	err = cudaMalloc(&newvect, numBucketsReshape * sizeof(Celula));
	if (err != cudaSuccess || newvect == NULL){
		std::cerr << "Memoria reshape\n";
		exit(-1);
	}
	err = cudaMemset(newvect, 0, numBucketsReshape * sizeof(Celula));
	if (err != cudaSuccess) {
		std::cerr << "Memset reshape\n";
		exit(-1);
	}

	num_blocks = (size + block_size - 1)/ block_size;
	reshapeAux>>(vect, newvect, size, numBucketsReshape);
	cudaDeviceSynchronize();
	err = cudaFree(vect);
	if (err != cudaSuccess) {
		std::cerr <<size<< " free reshape\n";
		exit(-1);
	}

	
	size = numBucketsReshape;
	vect = newvect;
}




/**
 * @brief CUDA kernel for parallel entry insertion.
 * 
 * functional Utility: Performs thread-parallel atomic updates with circular probing.
 */
__global__ void insert(int *sent_keys, int *sent_values, Celula *vect, int nr, int size) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j;

	if (i < nr && sent_keys[i] != 0) {
		int key = sent_keys[i], cc; 
		int begin = dhash1(key, size);

		/**
		 * Block Logic: Primary insertion probe pass.
		 */
		for(j = begin; j < size; j++) {
			cc = atomicCAS(&vect[j].cheie, 0, key);
			if(cc == 0 || cc == key) {
				vect[j].valoare = sent_values[i];
				return;
			}
		}
		
		/**
		 * Block Logic: Wrap-around insertion probe pass.
		 */
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
 * @brief CUDA kernel for parallel entry retrieval.
 */
__global__ void bashget(int *sent_keys, Celula *vect, int nr, int size, int *ret_values) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j;

	if (i < nr && sent_keys[i] != 0) {
		int key = sent_keys[i];
		int begin = dhash1(key, size);

		// Logic: Two-pass linear search.
		for (j = begin; j < size; j++) {
			
			if (key == vect[j].cheie) {
				
				ret_values[i] = vect[j].valoare;
				return;
			}
		}
		
		for(j = 0; j < begin; j++) {
			if(key == vect[j].cheie) {
				ret_values[i] = vect[j].valoare;
				return;
			}
		}
	}
}


/**
 * @brief Batch parallel insertion from host memory.
 * 
 * Logic: Transfers batches to the device and triggers expansion if load 
 * factor exceeds 90%.
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	if (!keys || !values || numKeys <= 0)
		return false;
	int *sent_keys = NULL , *sent_values = NULL, err;
	size_t block_size = 1024, num_blocks;

	err = cudaMalloc(&sent_keys, numKeys * sizeof(int));
	if (err != cudaSuccess || sent_keys == NULL) {
		std::cerr << "Memoria insert 1\n";
		exit(-1);
	}
	err = cudaMalloc(&sent_values, numKeys * sizeof(int));
	if (err != cudaSuccess || sent_values == NULL){
		std::cerr << "Memoria insert 2\n";
		exit(-1);
	}

	err = cudaMemcpy(sent_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		std::cerr << "Memcpy insert\n";
		exit(-1);
	}
	err = cudaMemcpy(sent_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		std::cerr << "memcpy insert\n";
		exit(-1);
	}
	
	
	
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
		std::cerr << "free insert 1\n";
		exit(-1);
	}
	err = cudaFree(sent_keys);
	if (err != cudaSuccess){
		std::cerr << "free insert 2\n";
		exit(-1);
	}
	return true;
}


/**
 * @brief Batch retrieval of values.
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	if (!keys || numKeys <= 0)
		return NULL;
	int *sent_keys, *ret_values, err;
	size_t block_size = 1024, num_blocks;

	err = cudaMalloc(&sent_keys, numKeys * sizeof(int));
	if (err != cudaSuccess || sent_keys == NULL){
		std::cerr << "Memoria get\n";
		exit(-1);
	}
	err = cudaMalloc(&ret_values, numKeys * sizeof(int));
	if (err != cudaSuccess || ret_values == NULL){
		std::cerr << "calloc get\n";
		exit(-1);
	}
	err = cudaMemset(ret_values, 0, numKeys * sizeof(int));
	if (err != cudaSuccess) {
		std::cerr << "memcpy get";
		exit(-1);
	}
	err = cudaMemcpy(sent_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		std::cerr << "memcpy get\n";
		exit(-1);
	}
	Celula *vl = (Celula *)calloc(size, sizeof(Celula));
	cudaMemcpy(vl, vect, size * sizeof(Celula), cudaMemcpyDeviceToHost);
	num_blocks = (numKeys + block_size - 1)/ block_size;
	bashget>>(sent_keys, vect, numKeys, size, ret_values);
	cudaDeviceSynchronize();
	err = cudaFree(sent_keys);
	if (err != cudaSuccess){
		std::cerr << "free get\n";
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
 * @brief Current saturation ratio.
 */
float GpuHashTable::loadFactor() {
	if (size == 0)
		return 0.f;
	float load = (float)inserari/size;

	return load;
}


// ... primeList documentation ...

/**
 * @brief Multi-tier hash functions utilizing prime number modular arithmetic.
 */
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
 * @struct Celula
 * @brief Representation of a key-value mapping on the device.
 */
typedef struct
{
	int cheie;
	int valoare;
}Celula;

/**
 * @class GpuHashTable
 * @brief Controller for managing device-side hash mapping.
 */
class GpuHashTable
{
	int inserari;   // Counter for elements inserted.
	int size;       // Total capacity of the buffer.
	Celula *vect;   // Device pointer to entry buffer.
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

