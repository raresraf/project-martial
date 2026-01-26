
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
	printf("OPT SOLVER\n");
	register double *C = (double *) malloc(N * N * sizeof(double));
	
	register double *ABBt = (double *) malloc(N * N * sizeof(double));
	register double *At = (double *) malloc(N * N * sizeof(double));
	register double *AtA = (double *) malloc(N * N * sizeof(double));

	register double *BBt = (double *) malloc(N * N * sizeof(double));
	register double *Bt = (double *) malloc(N * N * sizeof(double));

	transpose(N, B, Bt);
	register int i, j, k;

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			for (k = 0; k < N; ++k) {
				sum += B[i * N + k] * Bt[k * N + j];
			}
			BBt[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			for (k = i; k < N; ++k) {
				sum += A[i * N + k] * BBt[k * N + j];
			}
			ABBt[i * N + j] = sum;
		}
	}

	transpose(N, A, At);

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			register double sum = 0.0;
			for (k = 0; k <= i && k <= j; ++k) {
				sum += At[i * N + k] * A[k * N + j];
			}
			AtA[i * N + j] = sum;
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
