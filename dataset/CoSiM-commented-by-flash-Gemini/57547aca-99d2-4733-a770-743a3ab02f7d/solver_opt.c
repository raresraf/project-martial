/**
 * @file solver_opt.c
 * @brief Manually optimized Implementation of the matrix expression solver.
 *
 * Computes: Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Performance Optimization:
 * 1. Register Caching: Frequently used indices and loop variables are marked 
 *    for register allocation to minimize indexing overhead.
 * 2. Loop Reordering: Structuring inner loops to maximize spatial locality 
 *    and CPU cache hit rates.
 * 3. Pointer Arithmetic: Implicitly used through row-base address caching.
 *
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

#include "utils.h"

/**
 * my_solver - Optimized implementation using loop reordering and register hints.
 */
double* my_solver(int N, double *A, double *B) {
	
	register int i = 0;
	register int j = 0;
	register int k = 0;

	/**
	 * Pre-condition: Preparing transposed copies to ensure sequential 
	 * memory access in the innermost loops.
	 */
	double *At = malloc(N * N * sizeof(double));
	double *Bt = malloc(N * N * sizeof(double));

	for ( i = 0; i < N; i++) {
		for ( j = 0; j < N; j++) {
			register int index1 = i * N + j;
			register int index2 = j * N + i;

			At[index2] = A[index1];
			Bt[index2] = B[index1];
		}
	}

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Caches the row base index to avoid redundant multiplications.
	 */
	double *AB = malloc(N * N * sizeof(double));
	for (i = 0; i < N; i++) {
		register int index = i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0;
			for (k = i; k < N; k++) {
				sum += A[index + k] * B[k * N + j];
			}
			AB[index + j] = sum;
		}
	}

	/**
	 * Block Logic: Compute components AtA and ABBt.
	 * Optimization: Uses loop reordering (i-k-j) to improve spatial locality 
	 * for the destination matrices.
	 */
	double *AtA = malloc(N * N * sizeof(double));
	double *ABBt = malloc(N * N * sizeof(double));

	for (i = 0; i < N; i++) {
		for (k = 0; k < N; k++) {
			for (j = 0; j < N; j++) {
				AtA[i * N + j] += At[i * N + k] * A[k * N + j];
				ABBt[i * N + j] += AB[i * N + k] * Bt[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Result aggregation.
	 */
	double *res = malloc(N * N * sizeof(double));

	for (i = 0; i < N; i++) {
		register int index = i * N;
		for (j = 0 ; j < N; j++) {
			res[index + j] = ABBt[index + j] + AtA[index + j];
		}
	}

	free(At);
	free(Bt);
	free(AtA);
	free(ABBt);
	printf("OPT SOLVER\n");
	return res;	
}
