/**
 * @file solver_neopt.c
 * @brief Naive reference implementation of the matrix expression solver.
 *
 * Computes: Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Algorithm: Uses standard triple-nested loops for matrix multiplication.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store intermediate product matrices.
 *
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

#include "utils.h"

/**
 * my_solver - Naive implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	int i, j, k;

	/**
	 * Pre-condition: Allocation and zero-initialization of result and 
	 * intermediate buffers.
	 */
	double* C = (double*)calloc(N * N, sizeof(double));
	double* AB = (double*)calloc(N * N, sizeof(double)); 
	double* prod1 = (double*)calloc(N * N, sizeof(double)); 
	double* prod2 = (double*)calloc(N * N, sizeof(double)); 
	if (C == NULL || AB == NULL || prod1 == NULL || prod2 == NULL) {
        perror("calloc failed\n");
        exit(EXIT_FAILURE);
    }

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Exploits A's upper triangularity by starting k from i.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute prod1 = AB * B^T.
	 * Invariant: B^T[k][j] is accessed as B[j][k] to avoid explicit transposition.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				prod1[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Compute prod2 = A^T * A.
	 * Algorithm: Direct dot product implementation of the Gramian matrix.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				prod2[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Final summation of components.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = prod1[i * N + j] + prod2[i * N + j];
		}
	}

	free(AB);
	free(prod1);
	free(prod2);
	return C;
}
