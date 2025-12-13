
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

#define BLOCK_SIZE 1000
#define RESIZE_TRESHOLD 0.9f
#define OPTIMAL_LOAD 0.85f


#define cudaCheckErrors(msg) \
    { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Cuda error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(err), \
                __FILE__, __LINE__); \
            exit(1); \
        } \
    }


__device__ int hash_func(int data, int limit)
{
	return (data * 1087448823553llu) % 28282345988300791llu % limit;
}


__global__ void kernel_reshape(
	int *src, int src_size,


	int *dst, int dst_size
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int key, value, old_key, hash_value;

	if (idx >= src_size)
		return;

	key = src[2 * idx];

	if (key == KEY_INVALID)
		return;

	value = src[2 * idx + 1];

	hash_value = hash_func(key, dst_size);

	while (true) {
		old_key = atomicCAS(dst + 2 * hash_value, KEY_INVALID, key);
		if (old_key == KEY_INVALID) {
			dst[2 * hash_value + 1] = value;
			break;
		}
		hash_value = (hash_value + 1) % dst_size;
	}
}


__global__ void kernel_insert(
	int *buf, int size,
	int *keys, int *values, int count,
	int *new_elem
) {


	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int key, value, old_key, hash_value;

	if (idx >= count)
		return;

	key = keys[idx];

	if (key == KEY_INVALID)
		return;

	value = values[idx];

	hash_value = hash_func(key, size);
	while (true) {
		old_key = atomicCAS(buf + 2 * hash_value, KEY_INVALID, key);
		if (old_key == KEY_INVALID || old_key == key) {
			buf[2 * hash_value + 1] = value;
			if (old_key == KEY_INVALID)
				atomicAdd((unsigned int *)new_elem, 1);
			break;
		}
		hash_value = (hash_value + 1) % size;
	}
}


__global__ void kernel_get(
	int *buf, int size,


	int *keys, int *values, int count
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int key, hash_value;

	if (idx >= count)
		return;

	key = keys[idx];

	if (key == KEY_INVALID)
		return;

	hash_value = hash_func(key, size);
	while (1) {
		if (buf[2 * hash_value] == key) {
			values[idx] = buf[2 * hash_value + 1];
			break;
		}
		hash_value = (hash_value + 1) % size;
	}
}


GpuHashTable::GpuHashTable(int size)
{
	int buf_size = 2 * size * sizeof(int);

	cudaMalloc(&this->buf, buf_size);
	cudaCheckErrors("cudaMalloc");
	cudaMemset(this->buf, 0, buf_size);
	cudaCheckErrors("cudaMemset");

	this->size = size;
	this->count = 0;
}


GpuHashTable::~GpuHashTable()
{
	cudaFree(this->buf);
	cudaCheckErrors("cudaFree");
}


void GpuHashTable::reshape(int new_size)
{
	int *new_buf, new_buf_size = 2 * new_size * sizeof(int);

	cudaMalloc(&new_buf, new_buf_size);
	cudaCheckErrors("cudaMalloc");
	cudaMemset(new_buf, 0, new_buf_size);
	cudaCheckErrors("cudaMemset");

	kernel_reshapesize / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
		this->buf, this->size,
		new_buf, new_size
	);
	cudaDeviceSynchronize();
	cudaCheckErrors("cudaDeviceSynchronize");

	cudaFree(this->buf);
	cudaCheckErrors("cudaFree");

	this->buf = new_buf;
	this->size = new_size;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int num_keys)
{
	if (this->count + num_keys >= RESIZE_TRESHOLD * this->size) {
		this->reshape((this->count + num_keys) / OPTIMAL_LOAD);
	}

	int host_new_elem, *device_keys, *device_values, *device_new_elem;
	int buf_size = num_keys * sizeof(int);

	cudaMalloc(&device_keys, buf_size);
	cudaCheckErrors("cudaMalloc");
	cudaMalloc(&device_values, buf_size);
	cudaCheckErrors("cudaMalloc");
	cudaMalloc(&device_new_elem, sizeof(int));
	cudaCheckErrors("cudaMalloc");
	cudaMemcpy(device_keys, keys, buf_size, cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy");
	cudaMemcpy(device_values, values, buf_size, cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy");
	cudaMemset(device_new_elem, 0, sizeof(int));
	cudaCheckErrors("cudaMemset");

	kernel_insert>>(
		this->buf, this->size,
		device_keys, device_values, num_keys,
		device_new_elem
	);
	cudaDeviceSynchronize();
	cudaCheckErrors("cudaDeviceSynchronize");

	cudaMemcpy(&host_new_elem, device_new_elem, sizeof(int),
		cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy");

	cudaFree(device_keys);
	cudaCheckErrors("cudaFree");
	cudaFree(device_values);
	cudaCheckErrors("cudaFree");
	cudaFree(device_new_elem);
	cudaCheckErrors("cudaFree");

	this->count += host_new_elem;

	return true;
}


int* GpuHashTable::getBatch(int* keys, int num_keys)
{
	int *host_values, *device_keys, *device_values;
	int buf_size = num_keys * sizeof(int);

	cudaMalloc(&device_keys, buf_size);
	cudaCheckErrors("cudaMalloc");
	cudaMalloc(&device_values, buf_size);
	cudaCheckErrors("cudaMalloc");
	cudaMemcpy(device_keys, keys, buf_size, cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy");
	host_values = (int *)malloc(buf_size);
	DIE(!host_values, "malloc");

	kernel_get>>(
		this->buf, this->size,
		device_keys, device_values, num_keys
	);
	cudaDeviceSynchronize();
	cudaCheckErrors("cudaDeviceSynchronize");

	cudaMemcpy(host_values, device_values, buf_size, cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy");

	cudaFree(device_keys);
	cudaCheckErrors("cudaFree");
	cudaFree(device_values);
	cudaCheckErrors("cudaFree");

	return host_values;
}


float GpuHashTable::loadFactor()
{
	return this->count / (float)this->size;
}



#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
#ifndef _HASHCPU_
#define _HASHCPU_

#include 

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




class GpuHashTable
{
	private:
		int *buf;
		int size;
		int count;

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
