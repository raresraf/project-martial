
#include "utils.h"

/**
 * @file solver_neopt.c
 * @brief Unoptimized matrix solver implementation with manual property management.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Algorithm: Series of standard triple-nested loop matrix multiplications.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate workspaces (first_mul, second_mul, third_mul).
 * 
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

/**
 * my_solver - Reference implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	
	// Logic: Allocation of intermediate results for P1 and P2 components.
	double *first_mul = calloc (N * N, sizeof(double));
	if (!first_mul)
		return NULL;

	double *second_mul = calloc (N * N, sizeof(double));
	if (!second_mul)
		return NULL;

	double *third_mul = calloc (N * N, sizeof(double));
	if (!third_mul)
		return NULL;

	double *res = malloc (N * N * sizeof(double));
	if (!res)
		return NULL;

	int i, j, k;

	/**
	 * Block Logic: Compute first_mul = A * B.
	 * Optimization: Exploits A's upper triangularity by starting k from i.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				first_mul[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute second_mul = first_mul * B^T.
	 * Logic: Accesses B[j][k] (equivalent to B^T[k][j]) to compute the 
	 * symmetric product of (A*B) and B^T.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				second_mul[i * N + j] += first_mul[i * N + k] * B[j * N + k]; 
			}
		}
	}

	/**
	 * Block Logic: Compute third_mul = A^T * A.
	 * Optimization: Exploits upper triangularity of A by limiting k up to i. 
	 * Logic: Implements transposition by accessing A[k][i] instead of A[i][k].
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k <= i; k++) {
				third_mul[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Final summation of components into res.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			res[i * N + j] = second_mul[i * N + j] + third_mul[i * N + j];
		}
	}

	free(first_mul);
	free(second_mul);
	free(third_mul);
	return res;
}
