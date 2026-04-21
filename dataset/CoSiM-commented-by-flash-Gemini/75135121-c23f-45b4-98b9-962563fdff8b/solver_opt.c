/**
 * @75135121-c23f-45b4-98b9-962563fdff8b/solver_opt.c
 * @brief Manually optimized matrix expression solver.
 *
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using pointer 
 * arithmetic, register accumulation, and loop invariant code motion to minimize 
 * address calculation overhead and cache misses.
 *
 * Domain: HPC Performance Tuning.
 */

#include "utils.h"


/**
 * @brief Utility to compute the transpose of a square matrix.
 */
double* compute_transpose(int N, double* M)
{
	double* res = (double*)malloc(N * N * sizeof(double));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res[j * N + i] = M[i * N + j];
		}
	}
	return res;
}

/**
 * @brief Optimized matrix solver kernel.
 * Optimization: Replaces indirect indexing with direct pointer increments and 
 * promotes local sum variables to hardware registers to reduce memory traffic.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	// Pre-condition: Buffers are allocated for intermediate and final results.
	double* res_AxB = (double*)malloc(N * N * sizeof(double));
	double* res_ABBt = (double*)malloc(N * N * sizeof(double));
	double* res_AtA = (double*)malloc(N * N * sizeof(double));
	double* res = (double*)malloc(N * N * sizeof(double));
	double* B_t = compute_transpose(N, B);
	double* A_t = compute_transpose(N, A);
	register int i, j, k;
	double* pa = 0;
	double* pb = 0;
	
	/**
	 * Block Logic: Compute (A * B).
	 * Optimization: Pointer-based traversal for matrix A (pa) and matrix B (pb).
	 * Invariant: Scalar `suma` in register avoids redundant stores to `res_AxB`.
	 * Logic: Leverages A's upper triangularity by starting k at i.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			pa = &A[i * N + i];
			pb = &B[i * N + j];
			register double suma = 0;
			for (k = i; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			
			res_AxB[i * N + j] = suma;
		}
	}

	/**
	 * Block Logic: Compute (A * B) * B^T.
	 * Optimization: Pointer-based linear sweep for `res_AxB` and column-major sweep for `B_t`.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			pa = &res_AxB[i * N];
			pb = &B_t[j];
			register double suma = 0;
			for (k = 0; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			
			res_ABBt[i * N + j] = suma;
		}
	}
	
	free(res_AxB);
	free(B_t);

	/**
	 * Block Logic: Compute (A^T * A) and consolidate results.
	 * Optimization: Pointer arithmetic for Gramian matrix calculation.
	 * Invariant: Bound P = min(i, j) exploits triangular sparsity.
	 */
	int P = 0;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			pa = &A_t[i * N];
			pb = &A[j];
			if (i < j)
				P = i;
			else
				P = j;
			register double suma = 0;
			for (k = 0; k <= P; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			res[i * N + j] = 0;
			res_AtA[i * N + j] = suma;
			res[i * N + j] += res_ABBt[i * N + j] + res_AtA[i * N + j];
		}
	}
	free(A_t);
	
	free(res_ABBt);
	free(res_AtA);
	return res;	
}
