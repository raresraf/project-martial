/**
 * @file solver_neopt.c
 * @brief Unoptimized (naive) reference implementation of the matrix expression solver.
 *
 * Computes: Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Algorithm: Standard triple-nested loop matrix multiplications.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ to store multiple intermediate matrices.
 *
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

#include "utils.h"

/**
 * my_solver - Naive implementation using standard C nested loops.
 */
double* my_solver(int N, double *A, double* B) {
	int i = 0;
	int j = 0;
	int k = 0;

	/**
	 * Pre-condition: Explicit transposition of input matrices to simplify
	 * subsequent row-major products.
	 */
	double *At = malloc(N * N * sizeof(double));
	double *Bt = malloc(N * N * sizeof(double));

	for ( i = 0; i < N; i++) {
		for ( j = 0; j < N; j++) {
			int index1 = i * N + j;
			int index2 = j * N + i;

			At[index2] = A[index1];
			Bt[index2] = B[index1];
		}
	}

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Exploits A's upper triangularity by starting k from i.
	 */
	double *AB = malloc(N * N * sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
				AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Compute AtA = A^T * A and ABBt = AB * B^T.
	 * Logic: Computes two matrix products in a single loop structure for 
	 * workspace efficiency.
	 */
	double *AtA = malloc(N * N * sizeof(double));
	double *ABBt = malloc(N * N * sizeof(double));

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				AtA[i * N + j] += At[i * N + k] * A[k * N + j];
				ABBt[i * N + j] += AB[i * N + k] * Bt[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Final summation into result matrix.
	 */
	double *res = malloc(N * N * sizeof(double));

	for (i = 0; i < N; i++) {
		for (j = 0 ; j < N; j++) {
			res[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(At);
	free(Bt);
	free(AtA);
	free(ABBt);
	printf("NEOPT SOLVER\n");
	return res;
}
