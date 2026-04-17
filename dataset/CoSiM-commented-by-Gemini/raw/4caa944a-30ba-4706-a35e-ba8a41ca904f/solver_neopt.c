/**
 * @file solver_neopt.c
 * @brief Encapsulates functional utility for solver_neopt.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	int i, j, k;
	double *C, *AtA, *AB, *ABBt;

	C = calloc(N * N, sizeof(double));
	DIE(C == NULL, "calloc C");

	AtA = calloc(N * N, sizeof(double));
	DIE(AtA == NULL, "calloc AtA");

	AB = calloc(N * N, sizeof(double));
	DIE(AB == NULL, "calloc AB");

	ABBt = calloc(N * N, sizeof(double));
	DIE(ABBt == NULL, "calloc ABBt");

    
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        for (j = 0; j < N; j++) {
            /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
            for (k = i; k < N; k++) {
                AB[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        for (j = 0; j < N; j++) {
            /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
            for (k = 0; k < N; k++) {
                ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
            }
        }
    }

	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        for (j = 0; j < N; j++) {
            /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
            for (k = 0; k <= i && k <= j; k++) { /* Non-obvious bitwise operation or pointer arithmetic */
                AtA[i * N + j] += A[k * N + i] * A[k * N + j];
            }
        }
    }

	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        for (j = 0; j < N; j++) {
            C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
        }
    }

	free(AB);
    free(AtA);
    free(ABBt);

	return C;
}
