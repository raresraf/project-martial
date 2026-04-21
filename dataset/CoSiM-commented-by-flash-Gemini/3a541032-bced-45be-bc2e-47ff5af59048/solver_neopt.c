
#include "utils.h"

/**
 * @file solver_neopt.c
 * @brief Unoptimized (naive) matrix solver implementation.
 * 
 * Functional Intent: Provides a reference implementation for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Algorithm: Series of triple-nested loops for matrix-matrix multiplication.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store multiple intermediate matrix buffers.
 * 
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

/**
 * my_solver - Naive implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	int i = 0;
	int j = 0;
	int k = 0;

	/**
	 * Block Logic: Compute result_A = A^T * A.
	 * Logic: Accesses columns of A (as rows of A^T) by indexing k*N+i.
	 * Invariant: result_A[i][j] stores the dot product of A's i-th and j-th columns.
	 */
	double *result_A = calloc(N * N, sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result_A[i * N + j] = 0;
			for (k = 0; k < N; k++) {
				result_A[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute result_AB = A * B.
	 * Optimization: Exploits A's upper triangularity by starting k from i.
	 */
	double *result_AB = calloc(N * N, sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result_AB[i * N + j] = 0;
			for (k = i; k < N; k++) {
				result_AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute result_ABBt = result_AB * B^T.
	 * Logic: Accesses B[j][k] which corresponds to B^T[k][j].
	 */
	double *result_ABBt = calloc(N * N, sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			result_ABBt[i * N + j] = 0;
			for (k = 0; k < N; k++) {
				result_ABBt[i * N + j] += result_AB[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Final summation of components.
	 */
	double *C = calloc(N * N, sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0 ; j < N; j++) {
			C[i * N + j] = result_ABBt[i * N + j] + result_A[i * N + j];
		}
	}

	free(result_A);
	free(result_AB);
	free(result_ABBt);
	return C;
}
