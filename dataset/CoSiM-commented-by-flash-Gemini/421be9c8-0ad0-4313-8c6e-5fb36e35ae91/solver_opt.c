
#include "utils.h"

/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A^T * A) + (A * B) * B^T
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs several manual C tuning strategies:
 * 1. Register Pointer Caching: Frequently used row and column base addresses 
 *    are cached in registers to minimize indexing arithmetic in inner loops.
 * 2. Pointer Arithmetic: Replaces array indexing with direct address 
 *    increments (p2++, p3+=N) to reduce instruction count.
 * 3. Cache-friendly traversal: Structures loops to maximize linear access 
 *    patterns where possible.
 * 
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

/**
 * my_solver - Optimized implementation using pointer-based iteration.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	double *C, *AB;
	int i, j, k;

	C = (double *)calloc(sizeof(double), N * N);
	AB = (double *)calloc(sizeof(double), N * N);

	/**
	 * Block Logic: Compute C = A^T * A.
	 * Optimization: Pointer-based transposition dot product.
	 * Logic: Accesses columns of A via strided pointer increments (p2+=N, p3+=N).
	 * Invariant: p2 and p3 always point to the start of the current columns 
	 * for i and j respectively.
	 */
	for (i = 0; i < N; ++i) {
		register double *p1 = &A[i];
		for (j = 0; j < N; ++j) {
			register double *p2 = p1;
			register double *p3 = &A[j];
			register double sum = 0;
			for (k = 0; (k <= i && i < j) || (k <= j && i >= j); ++k) {
				sum += *p2 * *p3;
				p2 += N;
				p3 += N;
			}
			C[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Triangular pointer offset.
	 * Logic: Uses linear increments for row A (p2++) and strided increments 
	 * for column B (p3+=N). Exploits triangularity by starting p2 at A[i][i].
	 */
	for (i = 0; i < N; ++i) {
		register double *p1 = &A[i * N + i];
		for (j = 0; j < N; ++j) {
			register double *p2 = p1;
			register double *p3 = &B[i * N + j];
			register double sum = 0;
			for (k = i; k < N; ++k) {
				sum += *p2 * *p3;
				p2++;
				p3 += N;
			}
			AB[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Final accumulation C = C + AB * B^T.
	 * Optimization: Linear sequential access.
	 * Logic: Accesses row AB and row B (as column B^T) using simple linear 
	 * increments, achieving optimal cache throughput for the summation.
	 */
	for (i = 0; i < N; ++i) {
		register double *p1 = &AB[i * N];
		for (j = 0; j < N; ++j) {
			register double *p2 = p1;
			register double *p3 = &B[j * N];
			register double sum = 0;
			for (k = 0; k < N; ++k) {
				sum += *p2 * *p3;
				p2++;
				p3++;
			}
			C[i * N + j] += sum;
		}
	}

	free(AB);
	return C;
}
