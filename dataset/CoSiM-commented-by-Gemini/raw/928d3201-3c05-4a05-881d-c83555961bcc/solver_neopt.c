/**
 * @raw/928d3201-3c05-4a05-881d-c83555961bcc/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing C = A * (B * B^T) + A^T * A.
 * Algorithm: Standard nested loops generating explicit transposition maps and partial multiplications.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ representing intermediate target buffers.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	int i, j, k;

	/**
	 * Functional Utility: Allocates memory for computing B * B^T.
	 */
	double *BBt = (double*) calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Computes BBt = B * B^T.
	 * Invariant: Evaluates the dot product assuming matrix B's second operand is transposed using index `[j * N + k]`.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k < N; ++k) {
				BBt[i * N + j] += B[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Functional Utility: Allocates memory for computing A * (B * B^T).
	 */
	double *ABBt = (double*) calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Computes ABBt = A * BBt.
	 * Invariant: Exploits the upper triangular nature of matrix A by beginning the k loop from i.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = i; k < N; ++k) {
				ABBt[i * N + j] += A[i * N + k] * BBt[k * N + j];
			}
		}
	}

	free(BBt);

	/**
	 * Functional Utility: Allocates memory for computing A^T * A.
	 */
	double *AAt = (double*) calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Computes AAt = A^T * A.
	 * Invariant: Matrix A's upper triangular properties are preserved via dynamic minimum range bounds.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			for (k = 0; k <= ((i < j) ? i : j); ++k) {
				AAt[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Functional Utility: Allocates memory for the final aggregate matrix C.
	 */
	double *res = (double*) calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Consolidates C = ABBt + AAt.
	 * Invariant: Element-wise addition merging two separate mathematical terms sequentially.
	 */
	for (i = 0; i < N * N; ++i) {
		res[i] = ABBt[i] + AAt[i];
	}

	free(ABBt);
	free(AAt);

	return res;
}
