
#include "utils.h"

/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation using pointer arithmetic and register caching.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs several low-level C optimization strategies:
 * 1. Register Caching: Frequently accessed row base pointers (pa_first) and 
 *    accumulators (suma) are cached in CPU registers to minimize memory overhead.
 * 2. Pointer Arithmetic: Replaces array indexing with direct increments 
 *    (pa++, pb+=N) to reduce instruction count and pipeline stalls.
 * 3. Cache-friendly Traversal: Structures inner loops to exploit row-major 
 *    storage, converting strided access into sequential access where possible.
 * 4. Triangular Property: Specifically optimizes products involving A by 
 *    starting or ending loops at the diagonal (k=i).
 * 
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

/**
 * my_solver - Optimized implementation using pointer-based dot product loops.
 */
double* my_solver(int N, double *A, double* B) {
	// Logic: Pre-allocation of workspaces for P1 and P2 components.
	double *first_mul = calloc (N * N, sizeof(double));
	if (!first_mul)
		return NULL;

	double *second_mul = calloc (N * N, sizeof(double));
	if (!second_mul)
		return NULL;

	double *third_mul = calloc (N * N, sizeof(double));
	if (!third_mul)
		return NULL;

	double *result = malloc (N * N * sizeof(double));

	register int i, j, k;

	/**
	 * Block Logic: Compute first_mul = A * B.
	 * Optimization: Pointer-based dot product.
	 * Logic: Exploys A's upper triangularity by starting k from i. 
	 * Uses linear increments for row A (pa++) and strided increments for column B (pb+=N).
	 */
	for (i = 0; i < N; i++) {
		double *pa_first = &A[i * (N + 1)];
		double *res = &first_mul[i * N];
		for (j = 0; j < N; j++) {
			register double *pa = pa_first;
			register double *pb = &B[i * N + j];
			register double suma = 0;
			for (k = i; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			*res = suma;
			res++;
		}
	}

	/**
	 * Block Logic: Compute second_mul = first_mul * B^T.
	 * Optimization: Row-major row product.
	 * Logic: Multiplies rows of first_mul by rows of B (acting as columns of B^T). 
	 * This results in dual sequential access, maximizing cache performance.
	 */
	for (i = 0; i < N; i++) {
		double *pa_first = &first_mul[i * N];
		double *res = &second_mul[i * N];
		for (j = 0; j < N; j++) {
			register double *pa = pa_first;
			register double *pb = &B[j * N];
			register double suma = 0;
			for (k = 0; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb++;
			}
			*res = suma;
			res++;
		}
	}

	/**
	 * Block Logic: Compute third_mul = A^T * A.
	 * Optimization: Triangular symmetric product.
	 * Logic: Accesses columns of A (as rows of A^T) and rows of A using 
	 * strided increments, limited by the upper triangularity k <= i.
	 */
	for (i = 0; i < N; i++) {
		double *pa_first = &A[i];
		double *res = &third_mul[i * N];
		for (j = 0; j < N; j++) {
			register double *pa = pa_first;
			register double *pb = &A[j];
			register double suma = 0;
			for (k = 0; k <= i; k++) {
				suma += *pa * *pb;
				pa += N;
				pb += N;
			}
			*res = suma;
			res++;
		}
	}

	/**
	 * Block Logic: Final accumulation Result = second_mul + third_mul.
	 */
	for (i = 0; i < N; i++) {
		register double *res = &result[i * N];
		register double *pa = &second_mul[i * N];
		register double *pb = &third_mul[i * N];

		for (j = 0; j < N; j++) {
			*res = *pa + *pb;
			res++;
			pa++;
			pb++;
		}
	}

	free(first_mul);
	free(second_mul);
	free(third_mul);
	return result;
}
