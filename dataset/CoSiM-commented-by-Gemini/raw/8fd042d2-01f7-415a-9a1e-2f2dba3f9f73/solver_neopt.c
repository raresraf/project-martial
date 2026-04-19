/**
 * @raw/8fd042d2-01f7-415a-9a1e-2f2dba3f9f73/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing C = A^T * A + (A * B) * B^T.
 * Algorithm: Evaluates the formula through standard nested loops traversing matrices element-by-element.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ for intermediate arrays storing partial multiplication results.
 */
#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	
	/**
	 * Functional Utility: Initializes intermediate and output memory buffers for consecutive stage accumulations.
	 */
	double *temp2 = (double *)calloc(N * N, sizeof(double));
	double *temp3 = (double *)calloc(N * N, sizeof(double));
	double *temp5 = (double *)calloc(N * N, sizeof(double));
	double *res = (double *)calloc(N * N, sizeof(double));
	int i, j, k;
	
	/**
	 * Block Logic: Computes A^T * A.
	 * Invariant: Evaluates the dot product assuming matrix A exhibits upper triangular sparsity (k <= i).
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for(k = 0; k <= i; k++) {
				temp2[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}
	
	/**
	 * Block Logic: Computes partial product term A * B.
	 * Invariant: Treats matrix A structurally as upper triangular by initializing the summation index `k = i`.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				temp3[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
	
	/**
	 * Block Logic: Solves (A * B) * B^T by multiplying the previously computed intermediate matrix with B transposed.
	 * Invariant: Leverages reverse matrix indexing `[j * N + k]` to replicate transpose access.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for(k = 0; k < N; k++)
				temp5[i * N + j] += temp3[i * N + k] * B[j * N + k];
		}
	}
	
	/**
	 * Block Logic: Synthesizes final summation C = A^T * A + (A * B) * B^T.
	 * Invariant: Element-wise addition combining partial evaluations.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {	
			res[i * N + j] = temp5[i * N + j] + temp2[i * N + j];
		}
	}
	
	free(temp2);
	free(temp3);
	free(temp5);
	
	return res;
}
