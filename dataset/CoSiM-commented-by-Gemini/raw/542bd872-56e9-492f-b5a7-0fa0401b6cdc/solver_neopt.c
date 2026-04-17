
/*
 * @raw/542bd872-56e9-492f-b5a7-0fa0401b6cdc/solver_neopt.c
 * Module Level: Unoptimized matrix solver implementation (naive approach).
 */
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *C1 = calloc(N * N, sizeof(double));

	
	/* Pre-conditions: Matrix C1 is zero-allocated and Matrix B is accessible. Computes intermediate matrix multiplications. */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = i; k < N; k++) {
				C1[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	double *C2 = calloc(N * N, sizeof(double));

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				C2[i * N + j] += C1[i * N + k] * B[j * N + k];
			}
		}
	}

	double *C3 = calloc(N * N, sizeof(double));

	
	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k <= i; k++) {
				C3[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	for (int i = 0; i < N * N; i++)
		C2[i] += C3[i];

	free(C1);
	free(C3);

	return C2;
}
