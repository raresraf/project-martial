/**
 * @file solver_neopt.c
 * @brief Naive, non-optimized implementation of a matrix equation solver.
 * @details This file provides a basic, loop-based implementation for computing
 * the matrix equation: result = (A * B) * B^T + A^T * A. It assumes that the
 * input matrix A is upper triangular. This version is intended for correctness
 * testing and as a baseline for performance comparison against optimized versions.
 * The implementation uses three nested loops for matrix multiplication, leading to
 * a time complexity of O(N^3).
 */
#include "utils.h"
#include <stdlib.h>

/**
 * @brief Computes the matrix expression (A * B) * B^T + A^T * A using naive loops.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N, row-major, upper triangular).
 * @param B A pointer to the second input matrix (N x N, row-major).
 * @return A pointer to the resulting N x N matrix (named C). The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double* B) {
	double *A_tA, *AB, *ABB_t, *C;
	int i, j, k;

	// Allocate and zero-initialize memory for intermediate and final results.
	// On failure, the program exits.
	A_tA = (double *)calloc(N * N, sizeof(double));
	if (NULL == A_tA)
		exit(EXIT_FAILURE);
	
	AB = (double *)calloc(N * N, sizeof(double));
	if (NULL == AB)
		exit(EXIT_FAILURE);
	
	ABB_t = (double *)calloc(N * N, sizeof(double));
	if (NULL == ABB_t)
		exit(EXIT_FAILURE);
	
	C = (double *)calloc(N * N, sizeof(double));
	if (NULL == C)
		exit(EXIT_FAILURE);

	
	/**
	 * Block Logic: Compute AB = A * B.
	 * Assumes A is an upper triangular matrix, so the inner loop starts from k = i.
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = i; k < N; k++)
				AB[i * N + j] += A[i * N + k] * B[k * N + j];

	
	/**
	 * Block Logic: Compute ABB_t = AB * B^T.
	 * The access pattern `B[j * N + k]` corresponds to accessing B transposed.
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++)
				ABB_t[i * N + j] += AB[i * N + k] * B[j * N + k];

	
	/**
	 * Block Logic: Compute A_tA = A^T * A.
	 * The access pattern `A[k * N + i]` corresponds to accessing A transposed.
	 * Assumes A is upper triangular, so k <= i is sufficient.
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k <= i; k++)
				A_tA[i * N + j] += A[k * N + i] * A[k * N + j];

	
	/**
	 * Block Logic: Final summation.
	 * C = ABB_t + A_tA => ((A * B) * B^T) + (A^T * A)
	 */
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			C[i * N + j] = ABB_t[i * N + j] + A_tA[i * N + j];
	
	// Free intermediate matrices.
	free(A_tA);
	free(AB);
	free(ABB_t);

	return C;
}
