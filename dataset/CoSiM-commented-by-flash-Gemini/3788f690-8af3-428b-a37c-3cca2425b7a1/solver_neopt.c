
#include <string.h>
#include "utils.h"

/**
 * @file solver_neopt.c
 * @brief Unoptimized (naive) implementation of the matrix solver.
 * 
 * Functional Intent: Provides a reference implementation of the expression:
 * Result = (A * B * B^T) + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Algorithm: Triple-nested loop matrix multiplication.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store the result and intermediate buffers.
 * 
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

/**
 * my_solver - Naive implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	// Logic: Allocation for the resulting matrix C and workspace aux.
	double *C = (double*) calloc(N * N, sizeof(double));
	if (!C)
		return NULL;

	double *aux = (double*) calloc(N * N, sizeof(double));
	if (!aux)
		return NULL;

	/**
	 * Block Logic: Compute aux = A * B.
	 * Optimization: Exploits the fact that A is upper triangular (starts k from i).
	 * Invariant: aux[i][j] stores the dot product of A's i-th row and B's j-th column.
	 */
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				aux[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute C = aux * B^T.
	 * Logic: Instead of explicit transposition, access B[j][k] which corresponds 
	 * to B^T[k][j].
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				C[i * N + j] += aux[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Compute aux = A^T * A.
	 * Optimization: Handles A^T by accessing A[k][i] and A[k][j]. 
	 * Limit k to i+1 due to A being upper triangular (non-zero only when k <= i/j).
	 */
	memset(aux, 0, N * N * sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < i + 1; k++) {
				aux[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Accumulate the symmetric product into the final result.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += aux[i * N + j];
		}
	}

	free(aux);
	return C;
}
