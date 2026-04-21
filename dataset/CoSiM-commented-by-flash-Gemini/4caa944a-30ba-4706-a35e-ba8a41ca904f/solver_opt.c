
#include "utils.h"

/**
 * @file solver_opt.c
 * @brief Manually optimized Implementation of the matrix expression solver.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs several low-level C tuning strategies:
 * 1. Register Caching: Frequently accessed row base pointers (orig_pa) and 
 *    accumulators (sum) are cached in CPU registers to minimize indexing arithmetic.
 * 2. Pointer Arithmetic: Replaces array indexing with direct increments 
 *    (pa++, pb+=N) to reduce instruction latency and pipeline stalls.
 * 3. Cache-friendly Traversal: Structures inner loops to exploit row-major 
 *    storage, converting strided access into sequential access where possible.
 * 4. Triangular Property: Specifically optimizes products involving A by 
 *    starting loops at the diagonal (k=i) or limiting them (k<=i).
 * 
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

/**
 * my_solver - Optimized implementation using pointer-based dot product loops.
 */
double* my_solver(int N, double *A, double *B) {
	printf("OPT SOLVER\n");

	register int i, j, k;
	double *C, *AtA, *AB, *ABBt;

	// Logic: Pre-allocation of result and workspace buffers.
	C = calloc(N * N, sizeof(double));
	DIE(C == NULL, "calloc C");

	AtA = calloc(N * N, sizeof(double));
	DIE(AtA == NULL, "calloc AtA");

	AB = calloc(N * N, sizeof(double));
	DIE(AB == NULL, "calloc AB");

	ABBt = calloc(N * N, sizeof(double));
	DIE(ABBt == NULL, "calloc ABBt");

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Pointer-based dot product.
	 * Logic: Exploys A's upper triangularity by starting k from i. 
	 * Uses linear increments for row A (pa++) and strided increments for column B (pb+=N).
	 */
	for (i = 0; i < N; i++) {
		register double *orig_pa = A + i * N;
        for (j = 0; j < N; j++) {
			register double *pa = orig_pa + i;
			register double *pb = B + i * N + j;
			register double sum = 0.0;
            for (k = i; k < N; k++) {
                sum += *pa * *pb;
				pa++;
				pb += N;
            }
			*(AB + i * N + j) = sum;
        }
    }

	/**
	 * Block Logic: Compute ABBt = AB * B^T.
	 * Optimization: Row-major row product.
	 * Logic: Multiplies rows of AB by rows of B (acting as columns of B^T). 
	 * This results in dual sequential access patterns, maximizing CPU cache efficiency.
	 */
	for (i = 0; i < N; i++) {
		double *orig_pab = AB + i * N;
        for (j = 0; j < N; j++) {
			register double *pab = orig_pab;
			register double *pb = B + j * N;
			register double sum = 0.0;
            for (k = 0; k < N; k++) {
				sum += *pab * *pb;
				pab++;
				pb++;
            }
			*(ABBt + i * N + j) = sum;
        }
    }

	/**
	 * Block Logic: Compute AtA = A^T * A and Final aggregation.
	 * Optimization: Dual strided pointer access.
	 * Logic: Accesses columns of A (as rows of A^T) and rows of A using 
	 * strided increments, limited by the upper triangularity k <= min(i, j). 
	 * Folds the final addition of terms directly into the output write.
	 */
	for (i = 0; i < N; i++) {
		register double *orig_pat = A + i;
        for (j = 0; j < N; j++) {
			register double *pat = orig_pat;
			register double *pa = A + j;
			register double sum = 0.0;
            for (k = 0; k <= i && k <= j; k++) {
                sum += *pat * *pa;
				pat += N;
				pa += N;
            }
			*(AtA + i * N + j) = sum;
			*(C + i * N + j) = *(ABBt + i * N + j) + *(AtA + i * N + j);
        }
    }

	free(AB);
    free(AtA);
    free(ABBt);

	return C;
}
