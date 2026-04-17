/**
 * @file solver_neopt.c
 * @brief Naive non-optimized matrix solver implementation.
 * Unoptimized memory access pattern. Baseline for performance comparison.
 */

#include "utils.h"


/**
 * @brief Computes C = At * A + A * B * Bt.
 * Allocates memory dynamically and executes matrix operations.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *C, *BBt, *ABBt, *AtA; 
	int i, j, k;
	C    = calloc(N * N, sizeof(double));
	BBt  = calloc(N * N, sizeof(double));
	ABBt = calloc(N * N, sizeof(double));
	AtA  = calloc(N * N, sizeof(double));

	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if((C == NULL) || (AtA == NULL) || (ABBt == NULL) || (BBt == NULL)) {
		return NULL;
	}

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i ++) {
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0 ;j < N; j ++) {
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = 0; k <= i; k ++) {
				AtA[i * N + j] += A[k * N + j] * A[k * N + i];
			}
		}
	}

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i ++) {
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0 ;j < N; j ++) {
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = 0; k < N; k ++) {
				BBt[i * N + j] += B[i * N + k] * B[j * N + k];
			}
		}
	}

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i ++) {
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0 ;j < N; j ++) {
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = i; k < N; k ++) {
				ABBt[i * N + j] += A[i * N + k] * BBt[k * N + j];
			}
		}
	}

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i ++) {
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; j ++) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}
	
	free(AtA);
	free(ABBt);
	free(BBt);

	return C;
}

