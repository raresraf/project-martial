/**
 * @file solver_opt.c
 * @brief Manually optimized Implementation of the matrix expression solver.
 *
 * Computes: Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Performance Optimization:
 * 1. Register Caching: Frequently used indices and row pointers are explicitly 
 *    cached in registers.
 * 2. Loop Reordering (i-k-j): Maximizes spatial locality for the output matrix 
 *    by processing rows sequentially.
 * 3. Pointer Arithmetic: Uses direct offset increments to minimize address 
 *    calculation overhead.
 *
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

#include "utils.h"

/**
 * my_solver - Optimized implementation using loop reordering and register hints.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	int i, j, k;

	/**
	 * Pre-condition: Allocation of result and scratchpad buffers.
	 */
	double *C = (double*) calloc(N * N, sizeof(double));
	if (C == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}
    double *D = (double*) calloc(N * N, sizeof(double));
	if (D == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}

	/**
	 * Block Logic: Compute C = A * B.
	 * Optimization: Reorders loops to i-k-j for cache-friendly sequential 
	 * writes to the destination row.
	 */
    for (i = 0; i < N; i++) {
		register double *p_c = C + i * N;
		register double *p_a = A + i * N;
		for (k = i; k < N; k++) {
			register double *p_b = B + k * N;
            for (j = 0; j < N; j++) {
				*(p_c + j) += *(p_a + k) * *(p_b + j);
            }
        }
    }
	
	/**
	 * Block Logic: Compute D = C * B^T.
	 * Algorithm: Optimized dot product implementation using register accumulators.
	 */
    for (i = 0; i < N; i++) {
		register double *p_c = C + i * N;
		register double *p_d = D + i * N;
        for (j = 0; j < N; j++) {
			register double rez = 0.0;
			register double *p_b = B + j * N;
            for (k = 0; k < N; k++) {
				rez += *(p_c + k) * *(p_b + k);
            }
			*(p_d + j) = rez;
        }
    }
	
	/**
	 * Block Logic: Accumulate D += A^T * A.
	 * Optimization: Exploits symmetry and upper triangular structure to 
	 * reduce total operations.
	 */
	for (k = 0; k < N; k++) {
		register double *p_a = A + k * N;
		for (i = k; i < N; i++) {
			register double *p_d = D + i * N;
            for (j = k; j < N; j++) {
				*(p_d + j) += *(p_a + i) * *(p_a + j);
            }
		}
	}

	free(C);
	return D;
}
