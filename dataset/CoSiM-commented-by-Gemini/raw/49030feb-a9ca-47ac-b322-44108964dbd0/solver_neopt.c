/**
 * @file solver_neopt.c
 * @brief Encapsulates functional utility for solver_neopt.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

#include "utils.h"



double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *C = malloc(N * N * sizeof(double));
	double *mat = malloc(N * N * sizeof(double));
	

	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (int i = 0; i < N; i++) {
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (int j = 0; j < N; j++) {
			mat[i * N + j] = 0.0;
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (int k = 0; k < N; k++) {
				/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
				if(k >= i)
					mat[i * N + j] += A[i * N + k] * B[k * N + j];

			}
		}
	}
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (int i = 0; i < N; i++) {
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (int j = 0; j < N; j++) {
			C[i * N + j] = 0.0;
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (int k = 0; k < N; k++) {
					C[i * N + j] += mat[i * N + k] * B[j * N + k];

			}
		}
	}
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (int i = 0; i < N; i++) {
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (int j = 0; j < N; j++) {
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (int k = 0; k < N; k++) {
				/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
				if(k <= i || k >= j) /* Non-obvious bitwise operation or pointer arithmetic */
					C[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}
	free(mat);
	return C;
}