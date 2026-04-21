
#include "utils.h"

/**
 * @file solver_opt.c
 * @brief Manually optimized matrix expression solver implementation.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs several manual C tuning strategies:
 * 1. Register Caching: Frequently used row base pointers (orig_pa) and 
 *    accumulators (suma) are cached in CPU registers to minimize indexing arithmetic.
 * 2. Pointer Arithmetic: Replaces array indexing with direct increments 
 *    (pa++, pb+=N) to reduce instruction latency and pipeline stalls.
 * 3. Cache-friendly traversal: Structures inner loops to exploit row-major 
 *    storage, converting strided access into sequential access where possible.
 * 4. Triangular Property: Specifically optimizes products involving A by 
 *    starting loops at the diagonal (k=i) or limiting them (k<=j).
 * 
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

/**
 * my_solver - Optimized implementation using pointer-based dot product loops.
 */
double* my_solver(int N, double *A, double *B)
{
	double *AtA, *C, *ABBt, *AB;

	// Logic: Pre-allocation of workspaces for component results.
	C = malloc(N * N * sizeof(*C));
	if (NULL == C)
		exit(1);

	AtA = malloc(N * N * sizeof(*AtA));
	if (NULL == AtA)
		exit(1);

	AB = malloc(N * N * sizeof(*AB));
	if (NULL == AB)
		exit(1);

	ABBt = malloc(N * N * sizeof(*ABBt));
	if (NULL == ABBt)
		exit(1);
	
	// Pre-condition: Zero-initialize all components before computation.
	for (register int i = 0; i < N; i++)
		for (register int j = 0; j < N; j++){
			AB[i * N + j] = 0;
			ABBt[i * N + j] = 0;
			AtA[i * N + j] = 0;
			C[i * N + j] = 0;
		}

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Pointer-based dot product.
	 * Logic: Exploys A's upper triangularity by starting k from i. 
	 * Uses linear increments for row A (pa++) and strided increments for column B (pb+=N).
	 */
	for (register int i = 0; i < N; i++) {
		register double *orig_pa = &A[i * N + i];
		for (register int j = 0; j < N; j++){
			register double suma = 0.0;
			register double *pa = orig_pa;
    		register double *pb = &B[i * N + j];
			for (int k = i; k < N; k++){
				suma += *pa* *pb;
				pa++;
				pb +=N;
			}
			AB[i * N + j] = suma;
		}
	}
	
	/**
	 * Block Logic: Compute ABBt = AB * B^T.
	 * Optimization: Row-major row product.
	 * Logic: Multiplies rows of AB by rows of B (acting as columns of B^T). 
	 * This results in dual sequential access patterns, maximizing CPU cache efficiency.
	 */
	for (register int i = 0; i < N; i++) {
		register double *orig_pa = &AB[i * N];
		for (register int j = 0; j < N; j++) {
			register double suma = 0.0;
			register double *pa = orig_pa;
    		register double *pb = &B[j * N];
			for (register int k = 0; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb+=1;
			}
			ABBt[i * N + j] = suma;
		}
	}

	/**
	 * Block Logic: Compute AtA = A^T * A.
	 * Optimization: Triangular symmetric product.
	 * Logic: Accesses columns of A (representing rows of A^T) and rows of A using 
	 * strided increments, limited by the upper triangularity k <= j.
	 */
	for (register int i = 0; i < N; i++) {
		register double *orig_pa = &A[i];
		for (register int j = 0; j < N; j++){
			register double suma = 0.0;
			register double *pa = orig_pa;
    		register double *pb = &A[j];
			for (register int k = 0; k <= j; k++){
				suma += *pa* *pb;
				pa+=N;
				pb+=N;
			}
			AtA[i * N + j] = suma;
		}
	}
	
	/**
	 * Block Logic: Final accumulation pass.
	 * Logic: Sums the two matrix products using optimized pointer-based traversal.
	 */
	for (register int i = 0; i < N; i++) {
		register double *pa = &ABBt[i * N];
		register double *pb = &AtA[i * N];
		for (register int j = 0; j < N; j++){
			C[i * N + j] = *pa + *pb;
			pa++;
			pb++;
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);
	return C;
}
