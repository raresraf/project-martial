
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
    int i, j, k;
	double *AtA;
	AtA = malloc(N * N * sizeof(double));

	double *D; 
	D = malloc(N * N * sizeof(double));

	double *C; 
	C = malloc(N * N * sizeof(double));
	
	
	for (i = 0; i < N; i++) {
		for (j = i; j < N; j++) {
			AtA[i*N + j] = 0.0;
			for (k = 0; k < i+1; k++) {
				AtA[i * N + j] += A[k*N + i] * A[k*N + j];
			}
		}
	}

	
	for (i = 1; i < N; i++) {
		for (j = 0; j < i; j++) {
			AtA[i*N + j] = AtA[j*N + i];
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			D[i*N + j] = 0.0;
			for (k = i; k < N; k++) {
				D[i*N + j] += A[i*N + k] * B[k*N + j];
			}
		}
	}

	
    for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i*N + j] = 0.0;
			for (k = 0; k < N; k++) {
				C[i*N + j] += D[i*N + k] * B[j*N + k];
			}
		}
	}


	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i*N + j] += AtA[i*N + j];
		}
	}

	free(AtA);
	free(D);

	return C;
}
