
#include "utils.h"

int min(int a, int b) {
	return (a < b) ? a : b;
}


double* my_solver(int N, double *A, double* B) {

	
	double *C = calloc(N * N, sizeof(double));
	if(C == NULL) {
		perror("Calloc failed!");
		exit(ENOMEM);
	}
	double *D = calloc(N * N, sizeof(double));
	if(D == NULL) {
		perror("Calloc failed!");
		exit(ENOMEM);
	}

	
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			
			for(int k = i; k < N; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}


	
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			for(int k = 0; k < N; k++) {
				D[i * N + j] += C[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			for(int k = 0; k <= min(i, j); k++) {
				D[i * N + j] += A[k * N + i] * A[k * N + j];
			}
			
		}
	}

	free(C);

	return D;
}
