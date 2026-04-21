
#include "utils.h"

/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs multiple manual C tuning techniques:
 * 1. Register Caching: Frequently used indices and accumulators are explicitly 
 *    marked for register allocation.
 * 2. Index Pre-calculation: Caches common subexpressions (e.g., row base addresses) 
 *    to minimize redundant multiplications.
 * 3. Loop Fusion: Merges the calculation of (A*B)B^T and (A^T*A) into a single 
 *    pass to improve temporal locality for the output matrix C.
 * 4. Triangular Property: Specifically optimizes A*B by starting the inner 
 *    loop at k=i.
 * 
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

/**
 * my_solver - Optimized matrix solver with loop fusion and register hints.
 */
double* my_solver(int N, double *A, double* B) {
	register int i = 0;
	register int j = 0;
	register int k = 0;
	register int size = N * N * sizeof(double);

	/**
	 * Block Logic: Compute result_AB = A * B.
	 * Optimization: Row-major row caching.
	 * Invariant: indx stores the pre-calculated base offset for row i. 
	 * Limit k to [i, N) exploiting A's upper triangular nature.
	 */
	double *result_AB = malloc(size);
	for (i = 0; i < N; i++) {
		register int indx = i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			for (k = i; k < N; k++) {
				sum += A[indx + k] * B[k * N + j];
			}
			result_AB[indx + j] = sum;
		}
	}

	/**
	 * Block Logic: Compute Result = (result_AB * B^T) + (A^T * A).
	 * Optimization: Fused Calculation.
	 * Logic: Computes both matrix products simultaneously within the inner loop. 
	 * This maximizes cache reuse for row indx1 of result_AB and A.
	 */
	double *C = malloc(size);
	for (i = 0; i < N; i++) {
		register int indx1 = i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			register double sumA = 0.0;
			register int jn = j * N;
			for (k = 0; k < N; k++) {
				register int kn = k * N;
				// Synchronization: Simultaneous dot products for components P1 and P2.
				sum += result_AB[indx1 + k] * B[jn + k]; // (result_AB * B^T)
				sumA += A[kn + i] * A[kn + j];           // (A^T * A)
			}
			C[indx1 + j] = sum + sumA;
		}
	}

	printf("OPT SOLVER\n");
	free(result_AB);
	return C;	
}
