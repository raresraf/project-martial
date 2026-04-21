/**
 * @76dca07c-c626-46b0-8168-5bec8ae598f2/solver_opt.c
 * @brief Manually optimized matrix expression solver.
 *
 * Functional Utility: Computes Result = (A * B * B^T) + (A^T * A) using pointer 
 * arithmetic, register accumulation, and loop invariant code motion to minimize 
 * address calculation overhead and cache misses.
 *
 * Domain: HPC Performance Tuning.
 */

#include "utils.h"


/**
 * @brief Optimized matrix solver kernel.
 * Optimization: Replaces indirect indexing with direct pointer increments and 
 * promotes local sum variables and loop invariants to hardware registers to 
 * reduce memory traffic and enhance computational density.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	// Synchronization: Promotes induction variables to registers.
	register int i, j, k;
	register double  *pa, *pb, *orig_pa;
	register double suma;
	// Pre-condition: Allocates heap workspace for intermediate calculations.
	double *M1 = (double *)calloc(N * N, sizeof(double)); 
	double *M2 = (double *)calloc(N * N, sizeof(double)); 
	double *M3 = (double *)calloc(N * N, sizeof(double)); 
	if (!M1 || !M2 || !M3)
		return NULL;

	/**
	 * Block Logic: Compute M1 = A * B.
	 * Optimization: Pointer-based traversal for matrix A (pa) and matrix B (pb).
	 * Invariant: k starts at i, leveraging A's upper triangular properties.
	 * Logic: Scalar `suma` reduces store frequency to the M1 buffer.
	 */
	for (i = 0; i < N; i++) {
		orig_pa = &(A[i * N + i]);
		for (j = 0; j < N; j++) {
			pa = orig_pa;
			pb = &(B[i * N + j]);
			suma = 0.0;
			for(k = i; k < N; k++) {
				suma += *(pa++) * *pb;
				pb += N;
			}
			M1[i * N + j] = suma;
		}
	}

	/**
	 * Block Logic: Compute M2 = M1 * B^T.
	 * Optimization: Employs a linear memory sweep for contiguous data access patterns.
	 */
	for (i = 0; i < N; i++) {
		orig_pa = &(M1[i * N]);
		for (j = 0; j < N; j++) {
			pa = orig_pa;
			pb = &(B[j * N]);
			suma = 0.0;
			for(k = 0; k < N; k++) {
			 	suma += *pa * *pb;
				pa++;
				pb++;
			}
			M2[i * N + j] = suma;
		}
	}

	/**
	 * Block Logic: Compute M3 = A^T * A.
	 * Optimization: Scalar register accumulation for column-column products.
	 * Invariant: Bound P = min(i, j) exploits triangular sparsity.
	 */
	register int end;
	for (i = 0; i < N; i++) {
		orig_pa = &(A[i]);
		for (j = 0; j < N; j++) {
			if (i < j)
				end = i;
			else
				end = j;
			pa = orig_pa;
			pb = &(A[j]);
			suma = 0.0;
			for(k = 0; k <= end; k++) {
			 	suma += *pa * *pb;
				pa += N;
				pb += N;
			}
			M3[i * N + j] = suma;
		}

	}

	/**
	 * Block Logic: Final summation pass.
	 * Optimization: Reuses pointers to performing linear-time accumulation.
	 */
	for (i = 0; i < N; i++) {
		pa = &(M2[i]);
		pb = &(M3[i]);
		for (j = 0; j < N; j++) {
			*pa += *pb;
			pa += N;
			pb += N;
		}
	}

	/**
	 * Cleanup: Reclaims temporary matrix storage.
	 */
	free(M1);
	free(M3);
	return M2;
}
