
#include "utils.h"

/**
 * @file solver_opt.c
 * @brief Manually optimized implementation of the matrix expression solver.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs several low-level C tuning strategies:
 * 1. Register Caching: Frequently accessed row base pointers (tmp) and 
 *    accumulators (tmp_res) are cached in CPU registers to minimize indexing arithmetic.
 * 2. Pointer Arithmetic: Replaces array indexing with direct increments 
 *    (point_i++, point_j+=N) to reduce instruction count and pipeline stalls.
 * 3. Cache-friendly traversal: Structures inner loops to exploit row-major 
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
	register double *C = malloc(N * N * sizeof(double));
	register double *mat = malloc(N * N * sizeof(double));
	register double *tmp, *point_i, *point_j, *p_C;
	
	/**
	 * Block Logic: Compute mat = A * B.
	 * Optimization: Pointer-based dot product.
	 * Logic: Exploys A's upper triangularity by starting k from i. 
	 * Uses linear increments for row A (point_i++) and strided increments for column B (point_j+=N).
	 */
	for (register int i = 0; i < N; i++) {
		tmp = &A[i * N];
		p_C = &mat[i * N];
		for (register int j = 0; j < N; j++) {
			point_i = tmp + i;
			point_j = &B[j];
			point_j += i * N;
			register double tmp_res = 0.0;
			for (register int k = i; k < N; k++) {
					tmp_res += *point_i * *point_j;
					point_i++;
					point_j += N;

			}
			*p_C = tmp_res;
			p_C++;
		}
	}

	/**
	 * Block Logic: Compute C = mat * B^T.
	 * Optimization: Row-major row product.
	 * Logic: Multiplies rows of 'mat' by rows of B (acting as columns of B^T). 
	 * This results in dual sequential access, maximizing CPU cache efficiency.
	 */
	for (register int i = 0; i < N; i++) {
		tmp = &mat[i * N];
		p_C = &C[i * N];
		for (register int j = 0; j < N; j++) {
			point_i = tmp;
			point_j = &B[j * N];
			register double tmp = 0.0;
			for (register int k = 0; k < N; k++) {
					tmp += *point_i * *point_j;
					point_i++;
					point_j++;
			}
			*p_C = tmp;
			p_C++;
		}
	}
	
	/**
	 * Block Logic: Accumulate C += A^T * A.
	 * Optimization: Triangular symmetric product.
	 * Logic: Accesses columns of A (representing rows of A^T) and rows of A, 
	 * limited by the upper triangularity k <= i.
	 */
	for (register int i = 0; i < N; i++) {
		p_C = &C[i * N];
		for (register int j = 0; j < N; j++) {
			register double tmp_res = 0.0;
			for (register int k = 0; k <= i; k++) {
				register double *point = &A[k * N];
				tmp_res += *(point + i) * *(point + j);
			}
			*p_C += tmp_res;
			p_C++;
		}
	}
	free(mat);
	return C;
}
