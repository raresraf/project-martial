
#include "utils.h"

/**
 * @file solver_neopt.c
 * @brief Naive matrix solver implementation with manual property management.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Algorithm: Series of standard triple-nested loop matrix multiplications.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate matrix 'mat' and result 'C'.
 * 
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

/**
 * my_solver - Reference implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	double *C = malloc(N * N * sizeof(double));
	double *mat = malloc(N * N * sizeof(double));
	
	/**
	 * Block Logic: Compute mat = A * B.
	 * Optimization: Exploits upper triangularity of A (k >= i check).
	 * Invariant: mat[i][j] stores the dot product of A[i] and B[][j].
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			mat[i * N + j] = 0.0;
			for (int k = 0; k < N; k++) {
				if(k >= i)
					mat[i * N + j] += A[i * N + k] * B[k * N + j];

			}
		}
	}

	/**
	 * Block Logic: Compute C = mat * B^T.
	 * Logic: Accesses B[j][k] (equivalent to B^T[k][j]) to compute the product.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C[i * N + j] = 0.0;
			for (int k = 0; k < N; k++) {
					C[i * N + j] += mat[i * N + k] * B[j * N + k];

			}
		}
	}

	/**
	 * Block Logic: Accumulate C += A^T * A.
	 * Logic: Computes the symmetric product of triangular A by accessing 
	 * column indices (k*N+i). The conditional k <= i (or k >= j) reflects 
	 * the sparse structure of the triangular multiplication.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				if(k <= i || k >= j)
					C[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	free(mat);
	return C;
}
