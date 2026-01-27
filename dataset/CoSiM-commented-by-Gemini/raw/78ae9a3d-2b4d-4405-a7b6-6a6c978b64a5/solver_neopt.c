/**
 * @file solver_neopt.c
 * @brief Non-optimized, naive implementation of a matrix operation sequence.
 * @details This file provides a `my_solver` function that computes a result matrix
 * through a series of operations on input matrices A and B using basic C loops.
 * This version is intended as a baseline for performance comparison against
 * optimized and BLAS-based versions.
 */
#include "utils.h"


/**
 * @brief Computes a matrix result using nested loops.
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the input matrix A (N x N).
 * @param B A pointer to the input matrix B (N x N).
 * @return A pointer to the resulting matrix. The caller is responsible for freeing this memory.
 *
 * @note The implementation is non-obvious and appears to perform a custom sequence of matrix operations.
 * The logic is broken down as follows:
 * 1. A first pass computes two intermediate results in `aux1` and `aux2`.
 *    - `aux1` seems to involve a multiplication of an upper triangular part of A with B.
 *    - `aux2` seems to compute `A_lower^T * A_lower`.
 * 2. A second pass updates `aux2` by adding the product of `aux1` and the transpose of B.
 * The overall operation is approximately `aux2 = (A_lower^T * A_lower) + (A_upper * B) * B^T`.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	int i, j, k;
	// Allocate memory for two auxiliary matrices.
	double *aux1 = (double*) calloc(N * N, sizeof(double));
	double *aux2 = (double*) calloc(N * N, sizeof(double));

	/**
	 * @brief First pass: concurrent computation of two intermediate matrices.
	 * This block iterates through the matrices to perform two distinct calculations
	 * within the same loop structure, populating `aux1` and `aux2`.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
				for (k = 0; k < N; k++) {
					/**
					 * @brief This conditional block computes `aux1 = A_upper * B`.
					 * It multiplies the upper triangular part of A (including the diagonal)
					 * with matrix B.
					 */
					if (i <= k) {
						aux1[N * i + j] += A[N * i + k] * B[N * k + j];
					}
					/**
					 * @brief This block computes `aux2 = A_lower^T * A_lower`.
					 * It uses elements from the lower triangular part of A to compute
					 * a symmetric result in `aux2`. The condition `k <= i && k <= j`
					 * effectively transposes the columns of lower-triangular A into rows
					 * for the multiplication.
					 */
					if (k <= i && k <= j) {  
						aux2[N * i + j] += A[N * k + i] * A[N * k + j];
					}
				}
		}
	}

	/**
	 * @brief Second pass: Update `aux2` with the product of `aux1` and `B^T`.
	 * This operation calculates `aux1 * B^T` and adds the result to `aux2`.
	 * The indexing `B[N * j + k]` accesses B in a column-major fashion, which is
	 * equivalent to using the transpose of B (`B^T`) in a row-major calculation.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				aux2[N * i + j] += aux1[N * i + k] * B[N * j + k];
			}
		}
	}

	// Free the memory used for the first intermediate matrix.
	free(aux1);
	// Return the final result stored in `aux2`.
	return aux2;
}