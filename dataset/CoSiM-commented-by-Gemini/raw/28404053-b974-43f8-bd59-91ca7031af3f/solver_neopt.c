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
	
	int i = 0;
	int j = 0;
	int k = 0;

	double *fst = malloc(N * N * sizeof(double));
	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i++) {
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; j++) {
			double sum = 0;
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = 0; k < N; k++) {
				/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
				if (k >= i)
					sum += A[i * N + k] * B[k * N + j];
			}
			fst[i * N + j] = sum;
		}
	}

	double *snd = malloc(N * N * sizeof(double));

	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i++) {
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; j++) {
			double sum = 0;
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = 0; k < N; k++) {
				sum += fst[i * N + k] * B[j * N + k];
			}
			snd[i * N + j] = sum;
		}
	}

	double *third = malloc(N * N * sizeof(double));
	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i++) {
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; j++) {
			double sum = 0;
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = 0; k < N; k++) {
				sum += A[k * N + i] * A[k * N + j];
			}
			third[i * N + j] = sum;
			third[i * N + j] += snd[i * N + j];
		}
	}

	free(fst);
	free(snd);
	
	return third;			
}
