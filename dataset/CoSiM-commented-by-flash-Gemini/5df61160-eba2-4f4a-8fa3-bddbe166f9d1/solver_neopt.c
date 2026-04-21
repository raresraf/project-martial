/**
 * @5df61160-eba2-4f4a-8fa3-bddbe166f9d1/solver_neopt.c
 * @brief Unoptimized reference implementation of a matrix expression solver.
 * Functional Utility: Implements Result = (A * B) * B^T + (A^T * A) using basic 
 * iterative loops without cache considerations or specialized libraries.
 * Domain: HPC Benchmarking.
 */

#include "utils.h"


/**
 * @brief Naive matrix solver kernel.
 * Algorithm: Triple-nested loop matrix multiplication.
 */
double *my_solver(int N, double *A, double *B) {
	double *RESULT = (double *) calloc(N * N, sizeof(double));
	double *TEMPORARY = (double *) calloc(N * N, sizeof(double));
	double temporary_sum;
	int i, j, k;

	/**
	 * Block Logic: Compute TEMPORARY = A * B.
	 * Invariant: k loop skips elements based on A's upper triangular sparsity (k starts at i).
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			temporary_sum = 0;
			for (k = i; k < N; k++) {
				temporary_sum += A[i * N + k] * B[k * N + j];
			}
			TEMPORARY[i * N + j] = temporary_sum;
		}
	}

	/**
	 * Block Logic: Compute RESULT += TEMPORARY * B^T.
	 * Logic: Implicitly transposes B by accessing its elements as B[j * N + k].
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			temporary_sum = 0;
			for (k = 0; k < N; k++) {
				temporary_sum += TEMPORARY[i * N + k] * B[j * N + k];
			}
			RESULT[i * N + j] += temporary_sum;
		}
	}

	/**
	 * Block Logic: Compute RESULT += A^T * A.
	 * Logic: Computes the dot product of columns from matrix A.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			temporary_sum = 0;
			for (k = 0; k < N; k++) {
				temporary_sum += A[k * N + i] * A[k * N + j];
			}
			RESULT[i * N + j] += temporary_sum;
		}
	}

	free(TEMPORARY);
	return RESULT;
}
