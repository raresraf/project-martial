
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define BLOCK_THREADS 1024


__device__ int keyHash(int key, int limit)
{
	
	const unsigned int prime = 16777619;

	
	unsigned int hash = 2166136261;

	int *k = &key;

	hash = hash ^ ((unsigned char *) k)[0];
	hash = hash * prime;
	hash = hash ^ ((unsigned char *) k)[1];
	hash = hash * prime;
	hash = hash ^ ((unsigned char *) k)[2];
	hash = hash * prime;
	hash = hash ^ ((unsigned char *) k)[3];
	hash = hash * prime;

	return hash % limit;
}



__global__ void kernel_insert_batch(int *srcKeys, int *srcValues, int srcSize,
				    int *dstKeys, int *dstValues, int dstSize)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (idx < srcSize) {
		
		int key = srcKeys[idx];

		
		if (key == KEY_INVALID)
			return;

		
		int hash = keyHash(key, dstSize);

		
		for (int i = 0; i < dstSize; ++i) {
			
			const int dstKeyIndex = (hash + i) % dstSize;

			
			const int dstKey = atomicCAS(&dstKeys[dstKeyIndex],
						     KEY_INVALID,
						     key);

			
			if (dstKey == KEY_INVALID || dstKey == key) {
				dstValues[dstKeyIndex] = srcValues[idx];
				return;
			}
		}
	}
}



__global__ void kernel_get_batch(int *srcKeys, int *dstValues, int dstSize,
				 int *htKeys, int *htValues, int htSize)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (idx < dstSize) {
		
		int key = srcKeys[idx];

		
		if (key == KEY_INVALID)
			return;

		
		int hash = keyHash(key, htSize);

		
		for (int i = 0; i < htSize; ++i) {
			
			const int htIndex = (hash + i) % htSize;

			
			if (htKeys[htIndex] == key) {
				dstValues[idx] = htValues[htIndex];
				return;
			}
		}
	}
}


GpuHashTable::GpuHashTable(int size) : size(size), numElements(0), minLoad(0.8f)
{
	const size_t numBytes = size * sizeof(int);
	cudaError_t error;

	error = cudaMallocManaged(&this->keys, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	error = cudaMallocManaged(&this->values, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	memset(this->keys, 0, numBytes);
	memset(this->values, 0, numBytes);
}


GpuHashTable::~GpuHashTable()
{
	cudaError_t error;

	error = cudaFree(this->keys);
	CUDA_DIE(error, "cudaFree");

	error = cudaFree(this->values);
	CUDA_DIE(error, "cudaFree");
}


void GpuHashTable::reshape(int numBucketsReshape)
{
	const size_t numBytes = numBucketsReshape * sizeof(int);
	cudaError_t error;

	int *newKeys;
	int *newValues;

	error = cudaMallocManaged(&newKeys, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	error = cudaMallocManaged(&newValues, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	memset(newKeys, 0, numBytes);
	memset(newValues, 0, numBytes);

	const int numBlocks = (this->size + BLOCK_THREADS - 1) / BLOCK_THREADS;
	kernel_insert_batch>>(this->keys,
							  this->values,
							  this->size,
							  newKeys,
							  newValues,
							  numBucketsReshape);
	CUDA_DIE(cudaGetLastError(), "kernel_insert_batch");

	CUDA_DIE(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

	this->size = numBucketsReshape;

	cudaFree(this->keys);
	cudaFree(this->values);
	this->keys = newKeys;
	this->values = newValues;
}


bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys)
{
	const int newSize = this->numElements + numKeys;
	const size_t numBytes = numKeys * sizeof(int);
	cudaError_t error;

	if (newSize > this->size)
		this->reshape(ceil((float) newSize / this->minLoad));

	int *kerKeys;
	int *kerValues;

	error = cudaMalloc(&kerKeys, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	error = cudaMalloc(&kerValues, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	error = cudaMemcpy(kerKeys, keys, numBytes, cudaMemcpyHostToDevice);
	CUDA_DIE(error, "cudaMemcpy");

	error = cudaMemcpy(kerValues, values, numBytes, cudaMemcpyHostToDevice);
	CUDA_DIE(error, "cudaMemcpy");



	const int numBlocks = (numKeys + BLOCK_THREADS - 1) / BLOCK_THREADS;
	kernel_insert_batch>>(kerKeys, kerValues,
							  numKeys, this->keys,
							  this->values,
							  this->size);
	CUDA_DIE(cudaGetLastError(), "kernel_insert_batch");

	CUDA_DIE(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

	this->numElements += numKeys;

	cudaFree(kerKeys);
	cudaFree(kerValues);

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys)
{
	const size_t numBytes = numKeys * sizeof(int);
	cudaError_t error;

	int *kerKeys;
	int *values;

	error = cudaMalloc(&kerKeys, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	error = cudaMemcpy(kerKeys, keys, numBytes, cudaMemcpyHostToDevice);
	CUDA_DIE(error, "cudaMemcpy");

	error = cudaMallocManaged(&values, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	const int numBlocks = (numKeys + BLOCK_THREADS - 1) / BLOCK_THREADS;
	kernel_get_batch>>(kerKeys, values, numKeys,
						       this->keys,
						       this->values,
						       this->size);
	CUDA_DIE(cudaGetLastError(), "kernel_get_batch");

	CUDA_DIE(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

	cudaFree(kerKeys);

	return values;
}


float GpuHashTable::loadFactor()
{
	return (float) this->numElements / this->size;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef GPU_HASHTABLE_
#define GPU_HASHTABLE_

#define	KEY_INVALID 0

#define CUDA_DIE(code, msg)                                                   \
do {                                                                          \
	if (code != cudaSuccess) {                                            \
		fprintf(stderr, "(%s, %d): %s: %s", __FILE__, __LINE__, msg,  \
			cudaGetErrorString(code));                            \
		exit(EXIT_FAILURE);                                           \
	}                                                                     \
} while(0)                                                                    \

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




class GpuHashTable
{
	private:
		int size;
		int numElements;
		float minLoad;
		int *keys;
		int *values;
	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		
		float loadFactor();
		void occupancy();
		void print(std::string info);
	
		~GpuHashTable();
};

#endif  

#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define BLOCK_THREADS 1024


__device__ int keyHash(int key, int limit)
{
	
	const unsigned int prime = 16777619;

	
	unsigned int hash = 2166136261;

	int *k = &key;

	hash = hash ^ ((unsigned char *) k)[0];
	hash = hash * prime;
	hash = hash ^ ((unsigned char *) k)[1];
	hash = hash * prime;
	hash = hash ^ ((unsigned char *) k)[2];
	hash = hash * prime;
	hash = hash ^ ((unsigned char *) k)[3];
	hash = hash * prime;

	return hash % limit;
}

__global__ void kernel_insert_batch(int *srcKeys, int *srcValues, int srcSize,
				    int *dstKeys, int *dstValues, int dstSize,
				    int *countElements)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (idx < srcSize) {
		
		int key = srcKeys[idx];

		
		if (key == KEY_INVALID)
			return;

		
		int hash = keyHash(key, dstSize);

		
		for (int i = 0; i < dstSize; ++i) {
			
			const int dstKeyIndex = (hash + i) % dstSize;

			
			const int dstKey = atomicCAS(&dstKeys[dstKeyIndex],
						     KEY_INVALID,
						     key);

			
			if (dstKey == KEY_INVALID) {
				dstValues[dstKeyIndex] = srcValues[idx];
				return;
			} else if (dstKey == key) {
				dstValues[dstKeyIndex] = srcValues[idx];
				atomicSub(countElements, 1);
				return;
			}
		}
	}
}

__global__ void kernel_get_batch(int *srcKeys, int *dstValues, int dstSize,
				 int *htKeys, int *htValues, int htSize)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (idx < dstSize) {
		
		int key = srcKeys[idx];

		
		if (key == KEY_INVALID)
			return;

		
		int hash = keyHash(key, htSize);

		
		for (int i = 0; i < htSize; ++i) {
			
			const int htIndex = (hash + i) % htSize;

			
			if (htKeys[htIndex] == key) {
				dstValues[idx] = htValues[htIndex];
				return;
			}
		}
	}
}


GpuHashTable::GpuHashTable(int size) : size(size), minLoad(0.8f)
{
	const size_t numBytes = size * sizeof(int);
	cudaError_t error;

	error = cudaMallocManaged(&this->numElements, sizeof(int));
	CUDA_DIE(error, "cudaMallocManaged");

	*this->numElements = 0;

	error = cudaMallocManaged(&this->keys, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	error = cudaMallocManaged(&this->values, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	memset(this->keys, 0, numBytes);
	memset(this->values, 0, numBytes);
}


GpuHashTable::~GpuHashTable()
{
	cudaError_t error;

	error = cudaFree(this->keys);
	CUDA_DIE(error, "cudaFree");

	error = cudaFree(this->values);
	CUDA_DIE(error, "cudaFree");
}


void GpuHashTable::reshape(int numBucketsReshape)
{
	const size_t numBytes = numBucketsReshape * sizeof(int);
	cudaError_t error;

	int *newKeys;
	int *newValues;

	error = cudaMallocManaged(&newKeys, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	error = cudaMallocManaged(&newValues, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	memset(newKeys, 0, numBytes);
	memset(newValues, 0, numBytes);

	const int numBlocks = (this->size + BLOCK_THREADS - 1) / BLOCK_THREADS;
	kernel_insert_batch>>(this->keys,
							  this->values,
							  this->size,
							  newKeys,
							  newValues,
							  numBucketsReshape,
							  this->numElements);
	CUDA_DIE(cudaGetLastError(), "kernel_insert_batch");

	CUDA_DIE(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

	this->size = numBucketsReshape;

	cudaFree(this->numElements);
	cudaFree(this->keys);
	cudaFree(this->values);
	this->keys = newKeys;
	this->values = newValues;
}


bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys)
{
	*this->numElements += numKeys;
	const size_t numBytes = numKeys * sizeof(int);
	cudaError_t error;

	if (*this->numElements > this->size)
		this->reshape(ceil((float) *this->numElements / this->minLoad));

	int *kerKeys;
	int *kerValues;

	error = cudaMalloc(&kerKeys, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	error = cudaMalloc(&kerValues, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	error = cudaMemcpy(kerKeys, keys, numBytes, cudaMemcpyHostToDevice);
	CUDA_DIE(error, "cudaMemcpy");

	error = cudaMemcpy(kerValues, values, numBytes, cudaMemcpyHostToDevice);
	CUDA_DIE(error, "cudaMemcpy");

	const int numBlocks = (numKeys + BLOCK_THREADS - 1) / BLOCK_THREADS;
	kernel_insert_batch>>(kerKeys, kerValues,
							  numKeys, this->keys,
							  this->values,
							  this->size,
							  this->numElements);
	CUDA_DIE(cudaGetLastError(), "kernel_insert_batch");

	CUDA_DIE(cudaDeviceSynchronize(), "cudaDeviceSynchronize");


	cudaFree(kerKeys);
	cudaFree(kerValues);

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys)
{
	const size_t numBytes = numKeys * sizeof(int);
	cudaError_t error;

	int *kerKeys;
	int *values;

	error = cudaMalloc(&kerKeys, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	error = cudaMemcpy(kerKeys, keys, numBytes, cudaMemcpyHostToDevice);
	CUDA_DIE(error, "cudaMemcpy");

	error = cudaMallocManaged(&values, numBytes);
	CUDA_DIE(error, "cudaMallocManaged");

	const int numBlocks = (numKeys + BLOCK_THREADS - 1) / BLOCK_THREADS;
	kernel_get_batch>>(kerKeys, values, numKeys,
						       this->keys,
						       this->values,
						       this->size);
	CUDA_DIE(cudaGetLastError(), "kernel_get_batch");

	CUDA_DIE(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

	cudaFree(kerKeys);

	return values;
}


float GpuHashTable::loadFactor()
{
	return (float) *this->numElements / this->size;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef GPU_HASHTABLE_
#define GPU_HASHTABLE_

#define	KEY_INVALID 0

#define CUDA_DIE(code, msg)                                                   \
do {                                                                          \
	if (code != cudaSuccess) {                                            \
		fprintf(stderr, "(%s, %d): %s: %s", __FILE__, __LINE__, msg,  \
			cudaGetErrorString(code));                            \
		exit(EXIT_FAILURE);                                           \
	}                                                                     \
} while(0)                                                                    \

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




class GpuHashTable
{
	private:
		int size;
		int *numElements;
		float minLoad;
		int *keys;
		int *values;
	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
		
		float loadFactor();
		void occupancy();
		void print(std::string info);
	
		~GpuHashTable();
};

#endif  

