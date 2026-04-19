/**
 * @raw/86718b5f-b7e4-4201-b5f5-fcdee2963f89/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver.
 * Algorithm: Standard nested loops computing C = A^T * A + (A * B) * B^T
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ for intermediate result allocations.
 */
#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	int i = 0;
	int j = 0;
	int k = 0;

	/**
	 * Functional Utility: Allocates memory for storing A^T * A.
	 */
	double *result1 = malloc(N * N * sizeof(double));
	
	/**
	 * Block Logic: Computes the matrix multiplication result1 = A^T * A.
	 * Invariant: Performs a standard dot product over rows and columns of matrix A.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result1[i * N + j] = 0;
			for (k = 0; k < N; k++) {
				result1[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Functional Utility: Allocates memory for storing the intermediate product A * B.
	 */
	double *result2 = malloc(N * N * sizeof(double));
	
	/**
	 * Block Logic: Computes intermediate matrix result2 = A * B.
	 * Invariant: Matrix A is assumed to be upper triangular; only processes k >= i.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result2[i * N + j] = 0;
			for (k = i; k < N; k++) {
				result2[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Functional Utility: Allocates memory for storing (A * B) * B^T.
	 */
	double *result3 = malloc(N * N * sizeof(double));

	/**
	 * Block Logic: Computes result3 = result2 * B^T.
	 * Invariant: B is accessed with indices [j * N + k], meaning column elements of B^T.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result3[i * N + j] = 0;
			for (k = 0; k < N; k++) {
				result3[i * N + j] += result2[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Functional Utility: Allocates memory for the final matrix C.
	 */
	double *resfinal = malloc(N * N * sizeof(double));

	/**
	 * Block Logic: Computes final summation C = result3 + result1.
	 * Invariant: Performs element-wise addition of A^T * A and (A * B) * B^T.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0 ; j < N; j++) {
			resfinal[i * N + j] = result3[i * N + j] + result1[i * N + j];
		}
	}

	free(result1);
	free(result2);
	free(result3);
	printf("NEOPT SOLVER\n");
	return resfinal;

}
