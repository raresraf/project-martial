
#include "utils.h"

/**
 * @file solver_neopt.c
 * @brief Unoptimized (naive) implementation of the matrix expression solver.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Algorithm: Series of standard triple-nested loop matrix multiplications.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate matrices (C1, C2, C3).
 * 
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

/**
 * my_solver - Reference implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	// Logic: Pre-allocation and initialization of workspaces.
	double *C1 = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Compute C1 = A * B.
	 * Optimization: Exploits upper triangularity of A by starting k from i.
	 * Invariant: C1[i][j] stores the dot product of A's i-th row and B's j-th column.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = i; k < N; k++) {
				C1[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	double *C2 = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Compute C2 = C1 * B^T.
	 * Logic: Accesses B[j][k] (equivalent to B^T[k][j]) to compute the symmetric product.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				C2[i * N + j] += C1[i * N + k] * B[j * N + k];
			}
		}
	}

	double *C3 = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Compute C3 = A^T * A.
	 * Optimization: Exploits upper triangularity by limiting k up to i. 
	 * Logic: Implements transposition by accessing column indices k*N+i.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k <= i; k++) {
				C3[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Final accumulation pass.
	 * Invariant: Sums the general product term (C2) and the symmetric term (C3) 
	 * into the final result buffer.
	 */
	for (int i = 0; i < N * N; i++)
		C2[i] += C3[i];

	free(C1);
	free(C3);

	return C2;
}
