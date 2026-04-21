/**
 * @file solver_neopt.c
 * @brief Naive reference implementation of the matrix expression solver.
 *
 * Computes: Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Algorithm: Standard nested-loop matrix multiplications.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store intermediate product matrices.
 *
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

#include "utils.h"

int min(int a, int b) {
	return (a < b) ? a : b;
}

/**
 * my_solver - Naive implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B) {

	/**
	 * Pre-condition: Allocation and zero-initialization of result and 
	 * scratch buffers.
	 */
	double *C = calloc(N * N, sizeof(double));
	if(C == NULL) {
		perror("Calloc failed!");
		exit(ENOMEM);
	}
	double *D = calloc(N * N, sizeof(double));
	if(D == NULL) {
		perror("Calloc failed!");
		exit(ENOMEM);
	}

	/**
	 * Block Logic: Compute C = A * B.
	 * Optimization: Exploits A's upper triangularity by starting k from i.
	 */
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			for(int k = i; k < N; k++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute D = C * B^T.
	 * Invariant: B^T[k][j] is accessed as B[j][k] to avoid explicit transposition.
	 */
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			for(int k = 0; k < N; k++) {
				D[i * N + j] += C[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Accumulate D += A^T * A.
	 * Algorithm: Direct dot product implementation of the Gramian matrix, 
	 * bounded by A's triangularity.
	 */
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			for(int k = 0; k <= min(i, j); k++) {
				D[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	free(C);
	return D;
}
