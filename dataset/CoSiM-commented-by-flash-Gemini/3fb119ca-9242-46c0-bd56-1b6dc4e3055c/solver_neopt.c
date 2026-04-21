
#include "utils.h"

/**
 * @file solver_neopt.c
 * @brief Naive matrix solver implementation with manual triangularity exploitation.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Algorithm: Series of triple-nested loops for sequential matrix multiplications.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for intermediate workspaces (AB, ABBt, AtA).
 * 
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

#define min(x, y) (((x) < (y)) ? (x) : (y))

/**
 * my_solver - Reference implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	double * AB, *ABBt, * AtA, * result;
	int i, j, k;
	
	// Logic: Pre-allocation and initialization of workspaces.
	AB = calloc(N * N, sizeof(double));
	ABBt = calloc(N * N, sizeof(double));
	AtA = calloc(N * N, sizeof(double));
	result = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Exploits A's upper triangularity by starting k from i.
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute ABBt = AB * B^T.
	 * Logic: Handles transposition by accessing B[j][k] (equivalent to B^T[k][j]).
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = 0; k < N; k++) {
				ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	/**
	 * Block Logic: Compute AtA = A^T * A.
	 * Optimization: Exploits upper triangularity of A by limiting k up to min(i, j). 
	 * Logic: Implements A^T by accessing A[k][i] instead of A[i][k].
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for(k = 0; k <= min(i, j); k++) {
				AtA[i * N +j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Final summation into result buffer.
	 */
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			result[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);
	return result;
}
