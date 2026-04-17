/**
 * @file solver_neopt.c
 * @brief Unoptimized implementation of matrix operations.
 *
 * Provides a straightforward, naive implementation of the mathematical formula
 * C = A * B * B^T + A^T * A. Used as a baseline for performance comparisons.
 */

#include <string.h>
#include "utils.h"


/**
 * @brief Solves the matrix equation C = A * B * B^T + A^T * A without optimization.
 *
 * @param N Matrix dimension (N x N).
 * @param A Pointer to the first input matrix (assumed upper triangular).
 * @param B Pointer to the second input matrix.
 * @return Pointer to the resulting matrix C, or NULL on allocation failure.
 */
double* my_solver(int N, double *A, double* B) {
	double *C = (double*) calloc(N * N, sizeof(double));
	if (!C)
		return NULL;

	double *aux = (double*) calloc(N * N, sizeof(double));
	if (!aux)
		return NULL;
	int i, j, k;
	/**
	 * @brief Computes aux = A * B.
	 * Pre-condition: A is an upper triangular matrix (zero below diagonal).
	 * Invariant: aux stores partial products for row i.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * @brief Iterate over columns of B.
		 * Pre-condition: Valid row index i.
		 * Invariant: Cell aux[i, j] accumulates the dot product.
		 */
		for (j = 0; j < N; j++) {
			/**
			 * @brief Compute dot product utilizing A's upper triangular property.
			 * Pre-condition: Elements A[i, k] for k < i are zero.
			 * Invariant: Accumulates multiplication over valid non-zero elements.
			 */
			for (k = i; k < N; k++) {
				aux[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * @brief Computes C = aux * B^T (equivalent to aux * transpose(B)).
	 * Pre-condition: aux contains the result of A * B.
	 * Invariant: C stores partial products for row i.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * @brief Iterate over columns of the transposed B matrix.
		 * Pre-condition: Valid row index i.
		 * Invariant: Cell C[i, j] accumulates the dot product.
		 */
		for (j = 0; j < N; j++) {
			/**
			 * @brief Compute dot product between aux row and B row (representing B^T column).
			 * Pre-condition: aux and B are fully initialized.
			 * Invariant: Accumulates element-wise products.
			 */
			for (k = 0; k < N; k++) {
				C[i * N + j] += aux[i * N + k] * B[j * N + k];
			}
		}
	}

	memset(aux, 0, N * N * sizeof(double));
	/**
	 * @brief Computes aux = A^T * A.
	 * Pre-condition: aux buffer is zeroed out.
	 * Invariant: aux stores partial products for row i of A^T.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * @brief Iterate over columns of A.
		 * Pre-condition: Valid row index i.
		 * Invariant: Cell aux[i, j] accumulates the dot product.
		 */
		for (j = 0; j < N; j++) {
			/**
			 * @brief Compute dot product utilizing A's upper triangular structure.
			 * Pre-condition: A^T row i has non-zeros only up to index i.
			 * Invariant: Accumulates valid non-zero multiplications.
			 */
			for (k = 0; k < i + 1; k++) {
				aux[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * @brief Final accumulation C = C + aux.
	 * Pre-condition: C contains A * B * B^T, aux contains A^T * A.
	 * Invariant: Resulting matrix C is formed row by row.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * @brief Element-wise addition.
		 * Pre-condition: Valid row index i.
		 * Invariant: Elements up to j are summed.
		 */
		for (j = 0; j < N; j++) {
			C[i * N + j] += aux[i * N + j];
		}
	}

	free(aux);

	return C;
}
