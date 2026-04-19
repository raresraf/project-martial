/**
 * @raw/8890c616-fbd7-4a4e-98ec-ad5ad4a09f55/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Standard nested loops traversing the matrices.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ for storing the final and intermediate computation matrices.
 */

#include <stdlib.h>
#include "utils.h"

/**
 * Functional Utility: Allocates memory for the destination and intermediate matrices.
 */
void allocate(int N, double **C, double **AB, double **ABB_t,
			  double **A_tA)
{
	*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);

	*AB = calloc(N * N, sizeof(**AB));
	if (NULL == *AB)
		exit(EXIT_FAILURE);

	*ABB_t = calloc(N * N, sizeof(**ABB_t));
	if (NULL == *ABB_t)
		exit(EXIT_FAILURE);

	*A_tA = calloc(N * N, sizeof(**A_tA));
	if (NULL == *A_tA)
		exit(EXIT_FAILURE);
}

double* my_solver(int N, double *A, double* B)
{
	double *C, *AB, *ABB_t, *A_tA;

	int i, j, k;

	allocate(N, &C, &AB, &ABB_t, &A_tA);

	/**
	 * Block Logic: Computes the intermediate matrix product AB = A * B.
	 * Invariant: Exploits the upper triangular nature of matrix A (k >= i).
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Computes ABB_t = AB * B^T.
	 * Invariant: Accesses matrix B in transposed layout using the standard index arithmetic.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				ABB_t[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Computes the product A_tA = A^T * A.
	 * Invariant: Adjusts loop boundaries based on the upper triangular geometry of matrix A to avoid explicit multiplications by zero.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			if (i < j) {
				for (k = 0; k <= i; k++) {
					A_tA[i * N + j] += A[k * N + i] * A[k * N + j];
				}
			} else {
				for (k = 0; k <= j; k++) {
					A_tA[i * N + j] += A[k * N + i] * A[k * N + j];
				}
			}
		}
	}

	/**
	 * Block Logic: Consolidates the terms computing C = ABB_t + A_tA.
	 * Invariant: Performs an element-wise matrix addition to finalize the formula evaluation.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = ABB_t[i * N + j] + A_tA[i * N + j];
		}
	}

	free(AB);
	free(ABB_t);
	free(A_tA);

	return C;
}
