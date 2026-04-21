
#include "utils.h"

/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver utilizing loop reordering and pointer caching.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs several manual C tuning techniques:
 * 1. Loop Reordering (i-k-j): Enhances spatial locality for the output and right-hand 
 *    matrices, allowing for more efficient cache line usage.
 * 2. Register Pointer Caching: Caches the base address of the active row in registers 
 *    to minimize indexing arithmetic in the innermost loop.
 * 3. Property Exploitation: Limits iteration ranges based on A's triangularity 
 *    (e.g., k starting from i or j starting from k).
 * 
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

/**
 * my_solver - Optimized matrix solver with cache-aware loop structures.
 */
double* my_solver(int N, double *A, double* B) {
	double *At, *Bt, *sum, *sum2, *sum3, *res;
	int i = 0, j = 0, k = 0;
	
	// Logic: Pre-allocation of workspaces for component results and transpositions.
	At = (double *)calloc(N * N, sizeof(double));
	Bt = (double *)calloc(N * N, sizeof(double));
	sum = (double *)calloc(N * N, sizeof(double));
	sum2 = (double *)calloc(N * N, sizeof(double));
	sum3 = (double *)calloc(N * N, sizeof(double));
	res = (double *)calloc(N * N, sizeof(double));
	
	/**
	 * Block Logic: Initial transpositions.
	 * Optimization: Uses a shared index counter and exploits A's triangularity 
	 * to perform a sparse copy for At.
	 */
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			register int idx = i * N + j;
			if (i >= j){
				At[idx] = A[j * N + i];
			}
			Bt[idx] = B[j * N + i];
		}
	}

	/**
	 * Block Logic: Compute sum = A * B.
	 * Optimization: i-k-j Loop Order.
	 * Logic: By placing j in the innermost loop, the code accesses B[k] and 
	 * sum[i] linearly, maximizing CPU cache line utilization.
	 */
	for (i = 0; i < N; i++){
		register double *orig_pa = &A[i * N];
		for (k = i; k < N; k++){
			register double valA = orig_pa[k];
			register double *orig_pb = &B[k * N];
			register double *dest_row = &sum[i * N];
			for (j = 0; j < N; j++){
				dest_row[j] += valA * orig_pb[j];
			}
		}
	}

	/**
	 * Block Logic: Compute sum2 = sum * Bt.
	 * Optimization: i-k-j traversal for general matrix multiplication.
	 */
	for (i = 0; i < N; i++){
		register double *orig_pa = &sum[i * N];
		for (k = 0; k < N; k++){
			register double valSum = orig_pa[k];
			register double *orig_pb = &Bt[k * N];
			register double *dest_row = &sum2[i * N];
			for (j = 0; j < N; j++){
				dest_row[j] += valSum * orig_pb[j];
			}
		}
	}

	/**
	 * Block Logic: Compute sum3 = At * A.
	 * Optimization: Combined triangular/i-k-j pattern.
	 * Logic: Exploits At being lower triangular and A being upper triangular.
	 */
	for (i = 0; i < N; i++){
		register double *orig_pa = &At[i * N];
		for (k = 0; k < N; k++){
			register double valAt = orig_pa[k];
			register double *orig_pb = &A[k * N];
			register double *dest_row = &sum3[i * N];
			// Optimization: Starts j from k as A is upper triangular.
			for (j = k; j < N; j++){
				dest_row[j] += valAt * orig_pb[j];
			}
		}
	}

	/**
	 * Block Logic: Final summation pass.
	 */
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			register int idx = i * N + j;
			res[idx] = sum2[idx] + sum3[idx];
		}
	}
	
	free(At);
	free(Bt);
	free(sum);
	free(sum2);
	free(sum3);
	return res;
}
