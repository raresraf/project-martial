/**
 * @raw/8dfa30bd-f2c8-43eb-9acb-391166828a75/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Standard nested loops traversing the matrices.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ for intermediate accumulation matrices.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	int i, j, k;
	double *AB, *ABBt, *AtA, *C;

	/**
	 * Functional Utility: Allocates memory for intermediate computation matrices and the final result.
	 */
	AB = calloc(N * N, sizeof(double));
	if (AB == NULL)
		exit(-1);
	ABBt = calloc(N * N, sizeof(double));
	if (ABBt == NULL)
		exit(-1);	
	AtA = calloc(N * N, sizeof(double));
	if (AtA == NULL)
		exit(-1);	
	C = calloc(N * N, sizeof(double));
	if (C == NULL)
		exit(-1);

	/**
	 * Block Logic: Computes intermediate matrix AB = A * B.
	 * Invariant: Evaluates the dot product assuming matrix A is upper triangular (k >= i).
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Computes ABBt = AB * B^T.
	 * Invariant: Accesses B simulating transposed layout utilizing `[j * N + k]` logic.
	 */
	for (i = 0; i < N; i++) { 
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Computes AtA = A^T * A.
	 * Invariant: Tracks structural properties to prematurely exit loop via `k == i || k == j`.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				AtA[i * N + j] += A[k * N + i] * A[k * N + j];
				if (k == i || k == j)
					break;
			}
		}
	}

	/**
	 * Block Logic: Accumulates the previously computed terms determining C = ABBt + AtA.
	 * Invariant: Executes an element-wise matrix addition to construct the target output.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
