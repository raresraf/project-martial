
#include <stdlib.h>
#include "utils.h"



void transpose(int N, double *A, double *B) {
	int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
			B[i * N + j] = A[j * N + i];
		}
	}
}

double* my_solver(int N, double *A, double *B) {
	printf("NEOPT SOLVER\n");
	double *C = (double *) calloc(N * N, sizeof(double));
	double *ABBt = (double *) calloc(N * N, sizeof(double));
	double *At = (double *) calloc(N * N, sizeof(double));
	double *AtA = (double *) calloc(N * N, sizeof(double));

	double *BBt = (double *) calloc(N * N, sizeof(double));
	double *Bt = (double *) calloc(N * N, sizeof(double));
	
	transpose(N, B, Bt);
	int i, j, k;

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				BBt[i * N + j] += B[i * N + k] * Bt[k * N + j];
			}
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = i; k < N; ++k) {
				ABBt[i * N + j] += A[i * N + k] * BBt[k * N + j];
			}
		}
	}

	transpose(N, A, At);

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k <= i && k <= j; ++k) {
				AtA[i * N + j] += At[i * N + k] * A[k * N + j];
			}
		}
	}

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j]; 
		}
	}
	free(ABBt);
	free(At);
	free(AtA);
	free(BBt);
	free(Bt);

	return C;
}
