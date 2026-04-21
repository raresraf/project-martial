
#include "utils.h"

/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = A * (B * B^T) + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs several manual C tuning strategies:
 * 1. Symmetry Exploitation: Calculates only the upper half of symmetric 
 *    products (B*B^T and A^T*A) and mirrors the result, halving the inner 
 *    loop workload.
 * 2. Pointer Arithmetic: Uses direct pointer increments (*pa++, *pb++) to 
 *    minimize address calculation latency in the innermost loops.
 * 3. Register Caching: Frequently used accumulators and row pointers are 
 *    explicitly marked for CPU register allocation.
 * 4. Cache-friendly traversal: Structures the (A * BBT) product to traverse 
 *    the destination matrix sequentially.
 * 
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

/**
 * my_solver - Optimized implementation utilizing symmetry and pointer-based loops.
 */
double* my_solver(int N, double *A, double* B) {
	int i, j, k;

	// Logic: Workspace allocation for the symmetric product component P1.
	double *C1 = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Compute C1 = B * B^T.
	 * Optimization: Symmetry exploitation.
	 * Logic: Computes only the upper triangular part (j >= i) and mirrors 
	 * to the lower part, reducing operations by 50%.
	 */
	for (i = 0; i < N; i++) {
		double *orig_pa = &B[i * N];

		for (j = i; j < N; j++) {
			double *pa = orig_pa;
			double *pb = &B[j * N];
			register double sum = 0.0;

			// Invariant: Standard dot product using linear pointer increments.
			for (k = 0; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb++;
			}
			C1[i * N + j] = sum;
			C1[j * N + i] = sum;
		}
	}

	double *C2 = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Compute C2 = A * C1.
	 * Optimization: Triangular traversal.
	 * Logic: Uses row-major traversal for destination C2. Exploits A's 
	 * upper triangularity by starting k from i.
	 */
	for (i = 0; i < N; i++) {
		double *pa = &A[i * N + i];
		double *pb = &C1[i * N];

		for (k = i; k < N; k++) {
			for (j = 0; j < N; j++) {
				// Synchronization: Scalar-vector accumulation into destination row.
				C2[i * N + j] += *pa * *pb;
				pb++;
			}
			pa++;
		}
	}

	double *C3 = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Compute C3 = A^T * A.
	 * Optimization: Symmetry and Triangularity.
	 * Logic: Accesses columns of A as rows of A^T via strided pointer increments. 
	 * Exploits the fact that the result is symmetric and A is triangular.
	 */
	for (i = 0; i < N; i++) {
		double *orig_pa = &A[i];

		for (j = i; j < N; j++) {
			double *pa = orig_pa; 
			double *pb = &A[j];
			register double sum = 0.0;

			for (k = 0; k <= i; k++) {
				sum += *pa * *pb;
				pa += N;
				pb += N;
			}
			C3[i * N + j] = sum;
			C3[j * N + i] = sum;
		}
	}

	/**
	 * Block Logic: Final result aggregation.
	 */
	for (i = 0; i < N * N; i++)
		C2[i] += C3[i];

	free(C1);
	free(C3);

	return C2;	
}
