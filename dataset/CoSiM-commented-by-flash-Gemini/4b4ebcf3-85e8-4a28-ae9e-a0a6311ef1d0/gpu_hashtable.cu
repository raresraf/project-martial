
#include 
#include 
#include 
#include 
#include 
#include 
#include 

#include "gpu_hashtable.hpp"

__global__ void kernel_insert(HashTableEntry *hashtable_vec, int size,
		int *keys, int *values, int numKeys, unsigned int *entries_added)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int hash_index;

	if (index >= numKeys)
		return;
	
	hash_index = hash_func(keys[index], size);

	
	while(hashtable_vec[hash_index].key != keys[index] &&
		atomicCAS(&(hashtable_vec[hash_index].occupied), 0, 1)) {
		hash_index++;
		hash_index %= size;
	}

	
	if (hashtable_vec[hash_index].key != keys[index]) {
		hashtable_vec[hash_index].key = keys[index];
		atomicInc(entries_added, numKeys + 1);
	}
	hashtable_vec[hash_index].value = values[index];
	
}

__global__ void kernel_copy(HashTableEntry *hashtable_vec, int size,
		HashTableEntry *new_hashtable_vec, int new_size)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	HashTableEntry current_entry;

	if (index >= size)
		return;
	
	current_entry = hashtable_vec[index];

	
	if (current_entry.occupied == 1) {
		int hash_index = hash_func(current_entry.key, new_size);

		while (atomicCAS(&(new_hashtable_vec[hash_index].occupied), 0, 1)) {
			hash_index++;
			hash_index %= new_size;
		}

		new_hashtable_vec[hash_index].key = current_entry.key;
		new_hashtable_vec[hash_index].value = current_entry.value;
	}
}

__global__ void kernel_get(HashTableEntry *hashtable_vec, int size, int *keys,
		int *values, int numKeys)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int hash_index;

	if (index >= numKeys)
		return;
	
	hash_index = hash_func(keys[index], size);

	
	while (hashtable_vec[hash_index].occupied == 1 &&
		keys[index] != hashtable_vec[hash_index].key) {
		hash_index++;
		hash_index %= size;
	}

	if (hashtable_vec[hash_index].occupied == 1)
		values[index] = hashtable_vec[hash_index].value;
}


GpuHashTable::GpuHashTable(int size)
{
	HANDLE_ERROR(cudaSetDevice(1));

	HANDLE_ERROR(
		cudaMalloc(
			&(this->hashtable_vec),
			size * sizeof(HashTableEntry)
		)
	);

	this->size = size;
	this->num_elems = 0;
}


GpuHashTable::~GpuHashTable()
{
	HANDLE_ERROR(cudaFree(this->hashtable_vec));
}


void GpuHashTable::reshape(int numBucketsReshape)
{
	HashTableEntry *new_hashtable_vec;
	unsigned long new_size = numBucketsReshape / 0.8;
	
	HANDLE_ERROR(
		cudaMalloc(
			&new_hashtable_vec,
			new_size * sizeof(HashTableEntry)
		)
	);

	const size_t block_size = 16;
	size_t blocks_no = this->size/16 + (this->size % 16 == 0 ? 0 : 1);

	kernel_copy>>(
		this->hashtable_vec,
		this->size,
		new_hashtable_vec,
		new_size
	);

	HANDLE_ERROR(cudaDeviceSynchronize());

	HANDLE_ERROR(cudaFree(this->hashtable_vec));
	this->hashtable_vec = new_hashtable_vec;
	this->size = new_size;
}


bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys)
{
	const size_t block_size = 16;
	size_t blocks_no = numKeys/16 + (numKeys % 16 == 0 ? 0 : 1);
	unsigned int *new_entries_no;
	int *device_keys, *device_values;

	if (this->num_elems + numKeys >= this->size)
		reshape((this->num_elems + numKeys));

	HANDLE_ERROR(
		cudaMalloc(
			&device_keys,
			numKeys * sizeof(int)
		)
	);
	HANDLE_ERROR(
		cudaMalloc(
			&device_values,
			numKeys * sizeof(int)
		)
	);

	HANDLE_ERROR(
		cudaMallocManaged(
			&new_entries_no,
			sizeof(unsigned int)
		)
	);

	*new_entries_no = 0;

	HANDLE_ERROR(
		cudaMemcpy(
			device_keys,
			keys,
			numKeys * sizeof(int),
			cudaMemcpyHostToDevice
		)
	);
	HANDLE_ERROR(
		cudaMemcpy(
			device_values,
			values,
			numKeys * sizeof(int),
			cudaMemcpyHostToDevice
		)
	);
	


	kernel_insert>>(
		this->hashtable_vec,
		this->size,
		device_keys,
		device_values,
		numKeys,
		new_entries_no
	);
	HANDLE_ERROR(cudaDeviceSynchronize());

	this->num_elems += *new_entries_no;

	HANDLE_ERROR(cudaFree(device_keys));
	HANDLE_ERROR(cudaFree(device_values));

	return true;
}


int* GpuHashTable::getBatch(int* keys, int numKeys)
{
	const size_t block_size = 16;
	size_t blocks_no = numKeys/16 + (numKeys % 16 == 0 ? 0 : 1);
	int *device_keys, *device_values, *values;

	HANDLE_ERROR(
		cudaMalloc(
			&device_keys,
			numKeys * sizeof(int)
		)
	);
	HANDLE_ERROR(
		cudaMalloc(
			&device_values,
			numKeys * sizeof(int)
		)
	);

	HANDLE_ERROR(
		cudaMemcpy(
			device_keys,
			keys,
			numKeys * sizeof(int),
			cudaMemcpyHostToDevice
		)
	);

	values = (int *)calloc(numKeys, sizeof(int));
	if(values == NULL) {
		fprintf(stderr, "values get calloc\n");
		exit(ENOMEM);
	}



	kernel_get>>(
		this->hashtable_vec,
		this->size,
		device_keys,
		device_values,
		numKeys
	);
	HANDLE_ERROR(cudaDeviceSynchronize());

	HANDLE_ERROR(
		cudaMemcpy(
			values,
			device_values,
			numKeys * sizeof(int),
			cudaMemcpyDeviceToHost
		)
	);


	HANDLE_ERROR(cudaFree(device_keys));
	HANDLE_ERROR(cudaFree(device_values));
	
	return values;
}


float GpuHashTable::loadFactor()
{
	return (float)this->num_elems / this->size;
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

#define ENOMEM			12
#define	KEY_INVALID		0
#define FNV_OFFSET		2166136261llu
#define FNV_PRIME		16777619llu
#define PRIME_NUM		14480561146010017169llu

#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
			fprintf(stderr, "(%s, %d): ",	\
			__FILE__, __LINE__);	\
			perror(call_description);	\
			exit(errno);	\
		}	\
	} while (0)




static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        cerr << cudaGetErrorString(err) << " in " 
            << file << " at line " << line << endl;
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__device__ int hash_func_basic(int data, int limit) {
	return ((long)abs(data) * PRIME_NUM) % FNV_PRIME % limit;
}


__device__ int hash_func(int data, int limit) {
	unsigned long long hash = FNV_OFFSET;

	for (size_t i = 0; i < sizeof(int); i++) {
		char byte_data = data >> (i * 8) & 0xFF;
		hash = hash ^ byte_data;
		hash *= FNV_PRIME;
	}

	return hash % limit;
}

typedef struct {
	int key = 0;
	int value = 0;
	int occupied = 0;
} HashTableEntry;




class GpuHashTable
{
public:
	unsigned long size;
	unsigned long num_elems;
	HashTableEntry *hashtable_vec;

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

