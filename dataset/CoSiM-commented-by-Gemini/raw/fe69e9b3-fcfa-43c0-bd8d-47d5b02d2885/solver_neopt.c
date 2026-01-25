 
#include "utils.h"


double* my_solver(int N, double *A, double *B) {
	printf("NEOPT SOLVER\n");

	int i, j, k;
	double *C = calloc(N * N, sizeof(double));
	double *AUX1 = calloc(N * N, sizeof(double));
	double *AUX2 = calloc(N * N, sizeof(double));

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		} 
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				AUX1[i * N + j] += C[i * N + k] * B[j * N + k];
			}
		} 
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k <= i; k++) {
				AUX2[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		} 
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
				C[i * N + j] = AUX1[i * N + j] + AUX2[i * N + j];
		} 
	}

	free(AUX1);
	free(AUX2);
	
	return C;
}
