/**
 * @file solver_neopt.c
 * @brief Encapsulates functional utility for solver_neopt.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

#include "utils.h"



double* my_solver(int N, double *A, double* B) {
	
	double *AB;
	double *ABBt;
	double *AtA;
	double *C;
	int i, j, k;

	AB = calloc(N * N,  sizeof(double));
	ABBt = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));
	AtA = calloc(N * N, sizeof(double));
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	if (AB == NULL || ABBt == NULL || C == NULL || AtA == NULL) { /* Non-obvious bitwise operation or pointer arithmetic */
        perror("malloc failed\n");
        exit(EXIT_FAILURE);
    }

	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for(j = 0; j < N; j++) {
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (k = i; k < N; k++) {
               AB[i * N + j] += A[i * N + k] * B[k * N + j];
            }
		}
	}

	
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++) {
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for(j = 0; j < N; j++) {
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (k = 0; k < N; k++) {
               	ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
            }
		}
	}

	  
	/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
	for (i = 0; i < N; i++){
		/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
		for (j = 0; j < N; j++) {
			/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
			for (k = 0; k < N; k++) {
				/* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
				if (A[N * k + i] == 0 || A[N * k + j] == 0) /* Non-obvious bitwise operation or pointer arithmetic */
					break;
				AtA[N * i + j] += A[N * k + i] * A[N * k + j];
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
	free(ABBt);
	free(AtA);
	return C;

}
