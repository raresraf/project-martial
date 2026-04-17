/*
 * Module: @raw/3788f690-8af3-428b-a37c-3cca2425b7a1/solver_neopt.c
 * High-level purpose: Non-optimized solver.
 */
#include <string.h>
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *C = (double*) calloc(N * N, sizeof(double));
	if (!C)
		return NULL;

	double *aux = (double*) calloc(N * N, sizeof(double));
	if (!aux)
		return NULL;
	/* Pre-condition: C and aux are allocated. Invariant: N constraints. */
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				aux[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				C[i * N + j] += aux[i * N + k] * B[j * N + k];
			}
		}
	}

	memset(aux, 0, N * N * sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < i + 1; k++) {
				aux[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += aux[i * N + j];
		}
	}

	free(aux);

	return C;
}
