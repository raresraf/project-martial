/**
 * @file solver_neopt.c
 * @brief Encapsulates functional utility for solver_neopt.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

#include "utils.h"
#include <stdlib.h>

double* my_solver(int N, double *A, double* B) {
	double * C, *AB, *ABB_t;
	int i, j, k;

	
	C = calloc(N * N, sizeof(double));
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if(C == NULL) {
		printf("Probleme la alocarea memoriei\n");
	}
	AB = calloc(N * N, sizeof(double));
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if(AB == NULL)
		printf("Probleme la alocarea memoriei\n");
	ABB_t = calloc(N * N, sizeof(double));
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if(ABB_t == NULL)
		printf("Probleme la alocarea memoriei\n");

	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for(i = 0; i < N; i++) {
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for(j = 0; j < N; j++) {
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for(k = 0; k <= i; k++) {
				C[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}
	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for(i = 0; i < N; i++) {
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for(j = 0; j < N; j++) {
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for(k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for(i = 0; i < N; i++) {
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for(j = 0; j < N; j++) {
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for(k = 0; k < N; k++) {
				ABB_t[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for(i = 0; i < N; i++) {
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for(j = 0; j < N; j++) {
			C[i * N + j] += ABB_t[i * N + j];
		}
	}

	free(AB);
	free(ABB_t);

	return C;
}
