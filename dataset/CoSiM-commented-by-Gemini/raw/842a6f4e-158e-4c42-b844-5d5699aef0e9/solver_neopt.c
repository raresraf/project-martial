/**
 * @raw/842a6f4e-158e-4c42-b844-5d5699aef0e9/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver.
 * Algorithm: Standard nested loops for computing matrix operations involving an upper triangular matrix.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ for intermediate result allocations.
 */
#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	int i, j, k;

	/**
	 * Functional Utility: Allocates memory for the final result and intermediate matrices.
	 * Initializes all matrices to zero, ensuring correct accumulation during multiplication.
	 */
	double *C = (double *)calloc(N * N, sizeof(double));
	if (C == NULL) return NULL;

	double *result1 = (double *)calloc(N * N, sizeof(double));
	if (result1 == NULL) return NULL;

	double *result2 = (double *)calloc(N * N, sizeof(double));
	if (result2 == NULL) return NULL;

	/**
	 * Block Logic: Computes intermediate matrix `result1` = A * B.
	 * Invariant: Matrix A is assumed to be upper triangular, hence elements where i > k are skipped.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				if (i <= k) {
					result1[i * N + j] += A[i * N + k] * B[k * N + j];
				} else {
					continue;
				}
			}
		}
	}

	/**
	 * Block Logic: Computes `result2` = `result1` * B.
	 * Invariant: Accumulates the product of the intermediate matrix and B into `result2`.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				result2[i * N + j] += result1[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Computes A^T * A and stores it into C.
	 * Invariant: Matrix A is upper triangular, so the dot product of column i and column j 
	 * only needs to iterate up to the diagonal element of column i (k <= i).
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				if (k <= i) {
					C[i * N + j] += A[k * N + i] * A[k * N + j];
				} else {
					break;
				}
			}
		}
	}

	/**
	 * Block Logic: Finalizes the result matrix by adding `result2` to C.
	 * Invariant: Element-wise addition combining partial computations.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += result2[i * N + j];
		}
	}

	free(result1);
	free(result2);

	return C;
}
