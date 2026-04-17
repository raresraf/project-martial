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
	double *C;
	double *ABB_t;
	double *A_tA;
	double *AB;
	int i, j, k;

	C = malloc(N * N * sizeof(*C));
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (NULL == C)
		exit(EXIT_FAILURE);

	ABB_t = calloc(N * N, sizeof(*ABB_t));
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (NULL == ABB_t)
		exit(EXIT_FAILURE);

	A_tA = calloc(N * N, sizeof(*A_tA));
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (NULL == A_tA)
		exit(EXIT_FAILURE);

	AB = calloc(N * N, sizeof(*AB));
	/* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
	if (NULL == AB)
		exit(EXIT_FAILURE);

	
	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i++)
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; j++)
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = i; k < N; k++)
				AB[i * N + j] += A[i * N + k] * B[k * N + j];

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i++)
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; j++)
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = 0; k < N; k++)
				ABB_t[i * N + j] += AB[i * N + k] * B[j * N + k];

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i++)
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; j++)
			/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
			for (k = 0; k < N; k++)
				A_tA[i * N + j] += A[k * N + i] * A[k * N + j];

	
	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; i++)
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; j++)
			C[i * N + j] = ABB_t[i * N + j] + A_tA[i * N + j];

	free(ABB_t);
	free(A_tA);
	free(AB);

	return C;
}
