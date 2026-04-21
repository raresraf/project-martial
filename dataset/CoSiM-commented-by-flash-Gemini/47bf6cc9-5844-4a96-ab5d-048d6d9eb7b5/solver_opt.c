
#include "utils.h"

/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation using pointer arithmetic and register caching.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs several low-level C tuning strategies:
 * 1. Register Caching: Frequently used row base pointers (orig_pa) and 
 *    accumulators (sum) are cached in CPU registers to minimize indexing arithmetic.
 * 2. Pointer Arithmetic: Replaces array indexing with direct increments 
 *    (pa++, pb+=N) to reduce instruction latency and pipeline stalls.
 * 3. Cache-friendly traversal: Structures inner loops to exploit row-major 
 *    memory layout, converting strided access into sequential access where possible.
 * 4. Sparse Property Optimization: Specifically optimizes the Gramian component 
 *    (A^T * A) by breaking the inner loop upon encountering zero elements.
 * 
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

/**
 * my_solver - Optimized implementation using pointer-based dot product loops.
 */
double* my_solver(int N, double *A, double* B) {
	double *AB;
	double *C;
	
	// Logic: Pre-allocation of result and workspace matrices.
	AB = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));
	
	// Guard: Ensure memory provisioning succeeded.
	if (AB == NULL || C == NULL) {
        perror("malloc failed\n");
        exit(EXIT_FAILURE);
    }
	
	register int i, j, k;
	register double *orig_pa, *pa, *pb, sum;

    /**
     * Block Logic: Compute AB = A * B.
     * Optimization: Pointer-based dot product.
     * Logic: Exploys A's upper triangularity by starting k from i. 
     * Uses linear increments for row A (pa++) and strided increments for column B (pb+=N).
     */
    for (i = 0; i < N; i++) {
		orig_pa = &A[i * N];
		for (j = 0; j < N; j++) {
			pa = orig_pa + i;
			pb = &B[i * N + j];
			sum = 0;
			for (k = i; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb+= N;
			}
			AB[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Compute C = AB * B^T.
	 * Optimization: Row-major sequential dot product.
	 * Logic: Multiplies rows of AB by rows of B (acting as columns of B^T). 
	 * This results in dual sequential access patterns, maximizing CPU cache efficiency.
	 */
	for (i = 0; i < N; i++) {
		orig_pa = &AB[i * N];
		for (j = 0; j < N; j++) {
			pa = orig_pa;
			pb = &B[j * N];
			sum = 0;
			for (k = 0; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb++;
			}
			C[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Accumulate C += A^T * A.
	 * Optimization: Sparse/Triangular property exploitation.
	 * Logic: Accesses columns of A (representing rows of A^T) and rows of A using 
	 * strided increments. Terminate early once a zero element is hit, reflecting 
	 * A's triangular structure.
	 */
	for (i = 0; i < N; i++) {
		orig_pa = &A[i];
		for (j = 0; j < N; j++) {
			pa = orig_pa;
			pb = &A[j];
			sum = 0;
			for (k = 0; k < N; k++) {
			    // Optimization: Early-exit for sparse upper-triangular traversal.
			    if (*pa == 0 || *pb == 0) {
			        break;
				}
				sum += *pa * *pb;
		        pa += N;
				pb += N;
			}
			C[i * N + j] += sum;
		}
	}

	free(AB);
	return C;
}
