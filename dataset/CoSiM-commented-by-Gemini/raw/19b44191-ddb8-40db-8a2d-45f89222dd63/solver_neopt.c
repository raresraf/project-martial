/**
 * @file solver_neopt.c
 * @brief Naive, non-optimized implementation of a matrix equation solver.
 * @details This file provides a basic, loop-based implementation for computing
 * the matrix equation: C = A^T * A + A * (B * B^T). This version is intended
 * for correctness testing and as a baseline for performance comparison against
 * optimized versions. The implementation uses triple-nested loops, leading to
 * a time complexity of O(N^3).
 */
#include "utils.h"
#include <stdlib.h>

/**
 * @brief Computes the matrix expression C = A^T * A + A * (B * B^T) using naive loops.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N, row-major).
 * @param B A pointer to the second input matrix (N x N, row-major).
 * @return A pointer to the resulting N x N matrix (C). The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double* B) {
	double *rez, *C;
	int i, j, k;

	// Allocate and zero-initialize memory for the result and an intermediate matrix.
	C = calloc(N * N, sizeof(double));
	rez = calloc(N * N, sizeof(double));

	
	/**
	 * Block Logic: Compute rez = B * B^T.
	 * The access pattern `B[j * N + k]` accesses B transposed.
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				rez[i * N + j] += B[i * N + k] * B[j * N + k];
			}
		}
	}

	
	/**
	 * Block Logic: Compute C = A * rez, which is A * (B * B^T).
	 * This calculates the second term of the main equation.
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				C[i * N + j] += A[i * N + k] * rez[k * N + j];
			}
		}
	}

	
	// Zero-out the intermediate matrix `rez` for reuse.
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			rez[i * N + j] = 0;
		}
	}

	
	/**
	 * Block Logic: Compute rez = A^T * A and add it to C.
	 * The calculation `rez[i][j] += A[k][i] * A[k][j]` computes a term of A^T * A.
	 * The result is immediately added to C. The loop `k <= j` suggests an
	 * assumption about the matrix structure (e.g., lower triangular A) or may be a bug if A is dense.
	 * Time Complexity: O(N^3)
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k <= j; ++k) {
				rez[i * N + j] += A[k * N + i] * A[k * N + j];
			}
			C[i * N + j] += rez[i * N + j];
		}
	}
	
	// Free the intermediate matrix.
	free(rez);
	return C;
}
