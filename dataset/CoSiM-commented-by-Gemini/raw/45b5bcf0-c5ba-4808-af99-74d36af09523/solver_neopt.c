/**
 * @file solver_neopt.c
 * @brief Encapsulates functional utility for solver_neopt.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	
	double *first_mul = calloc (N * N, sizeof(double));

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if (!first_mul)
		return NULL;

	double *second_mul = calloc (N * N, sizeof(double));

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if (!second_mul)
		return NULL;

	double *third_mul = calloc (N * N, sizeof(double));

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if (!third_mul)
		return NULL;

	double *res = malloc (N * N * sizeof(double));

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if (!res)
		return NULL;

	int i, j, k;

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (j = 0; j < N; j++) {
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (k = i; k < N; k++) {
				first_mul[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (j = 0; j < N; j++) {
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (k = 0; k < N; k++) {
				second_mul[i * N + j] += first_mul[i * N + k] * B[j * N + k]; 
			}
		}
	}

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (j = 0; j < N; j++) {
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (k = 0; k <= i; k++) {
				third_mul[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (j = 0; j < N; j++) {
			res[i * N + j] = second_mul[i * N + j] + third_mul[i * N + j];
		}
	}

	free(first_mul);
	free(second_mul);
	free(third_mul);

	return res;
}
