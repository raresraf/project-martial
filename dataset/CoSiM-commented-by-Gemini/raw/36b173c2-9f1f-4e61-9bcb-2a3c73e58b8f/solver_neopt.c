/**
 * @file solver_neopt.c
 * @brief Naive non-optimized matrix solver implementation.
 * Unoptimized memory access pattern. Baseline for performance comparison.
 */

#include "utils.h"
#define MIN(x, y) (((x) < (y)) ? (x) : (y))





double* my_transpose(int N, double* to_be_transposed) {
	int i, j;
	double *transpose = (double*) calloc(N * N, sizeof(double));
	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i  = 0; i < N; ++i) {
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; ++j) {
			transpose[i * N + j] = to_be_transposed[j * N + i];
		}
	}
	return transpose;
}

/**
 * @brief Computes C = At * A + A * B * Bt.
 * Allocates memory dynamically and executes matrix operations.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	int i, j, k;
	int min = 0;

	printf("NEOPT SOLVER\n");
	double *C = (double*) calloc(N * N, sizeof(double));
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (C == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	double *At, *Bt;

	double *another_C = (double*) calloc(N * N, sizeof(double));
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (another_C == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	double *res_A = (double*) calloc(N * N, sizeof(double));
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (res_A == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	double *res = (double*) calloc(N * N, sizeof(double));
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (res == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	At = my_transpose(N, A);
	Bt = my_transpose(N, B);

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; ++i) {
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; ++j) {
			C[i * N + j] = 0.0;
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = i; k < N; ++k) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; ++i) {
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; ++j) {
			another_C[i * N + j] = 0.0;
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = 0; k < N; ++k) {
				another_C[i * N + j] += C[i * N + k] * Bt[k * N + j];
			}
		}
	}

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; ++i) {
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; ++j) {
			min = MIN(i, j);
			res_A[i * N + j] = 0.0;
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for(k = 0; k < min + 1; ++k) {
				res_A[i * N + j] += At[i * N + k] * A[k * N + j];
			}
		}
	}

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; ++i) {
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; ++j) {
			res[i * N + j] = another_C[i * N + j] + res_A[i * N + j];
		}
	}
	free(res_A);
	free(another_C);
	free(At);
	free(Bt);
	free(C);
	return res;
}
