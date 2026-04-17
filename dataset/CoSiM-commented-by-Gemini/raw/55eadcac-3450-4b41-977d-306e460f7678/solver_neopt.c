
/*
 * Module: Unoptimized Solver
 * @raw/55eadcac-3450-4b41-977d-306e460f7678/solver_neopt.c
 * Purpose: Naive implementation of matrix math.
 */
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	int i, j, k;

	
	double* C = calloc(N * N, sizeof(double));
	double* AB = calloc(N * N, sizeof(double)); 
	double* prod1 = calloc(N * N, sizeof(double)); 
	double* prod2 = calloc(N * N, sizeof(double)); 
	if (C == NULL || AB == NULL || prod1 == NULL || prod2 == NULL) {
        perror("calloc failed\n");
        exit(EXIT_FAILURE);
    }

	/* Pre-conditions: Matrices A, B and AB allocated.
	 * Invariants: Computing partial product for AB. */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			
			for (k = i; k < N; k++) {
				
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				
				
				prod1[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				
				prod2[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			
			C[i * N + j] = prod1[i * N + j] + prod2[i * N + j];
		}
	}

	
	free(AB);
	free(prod1);
	free(prod2);
	return C;
}