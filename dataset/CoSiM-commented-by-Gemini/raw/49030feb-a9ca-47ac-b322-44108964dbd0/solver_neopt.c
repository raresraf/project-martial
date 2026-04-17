
/* @raw/49030feb-a9ca-47ac-b322-44108964dbd0/solver_neopt.c: Unoptimized matrix solver */
#include "utils.h"



double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *C = malloc(N * N * sizeof(double));
	double *mat = malloc(N * N * sizeof(double));
	

	/* Pre-condition: mat array allocated. */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			mat[i * N + j] = 0.0;
			for (int k = 0; k < N; k++) {
				if(k >= i)
					mat[i * N + j] += A[i * N + k] * B[k * N + j];

			}
		}
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C[i * N + j] = 0.0;
			for (int k = 0; k < N; k++) {
					C[i * N + j] += mat[i * N + k] * B[j * N + k];

			}
		}
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				if(k <= i || k >= j)
					C[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}
	free(mat);
	return C;
}