
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *AB;
	double *first;
	double *second;
	double *C;
	AB = calloc(N * N, sizeof(double));
	if(AB == NULL){
		exit(-1);
	}
	first = calloc(N * N, sizeof(double));
	if(first == NULL){
		exit(-1);
	}
	second = calloc(N * N, sizeof(double));
	if(second == NULL){
		exit(-1);
	}
	C = calloc(N * N, sizeof(double));
	if(C == NULL){
		exit(-1);
	}
	int i, j, k = 0;
    
	for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			for(k = i; k < N; k++){
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	
    
	for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			for(k = 0; k < N; k++){
				first[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}
    
	for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			for(k = 0; k <= i; k++){
				second[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}
    
    for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			C[i * N + j] = first[i * N + j] + second[i * N + j];
		}
	}


	free(AB);
	free(first);
	free(second);
	return C;
}
