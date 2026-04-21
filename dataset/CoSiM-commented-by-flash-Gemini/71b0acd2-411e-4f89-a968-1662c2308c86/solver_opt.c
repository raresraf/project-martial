/**
 * @71b0acd2-411e-4f89-a968-1662c2308c86/solver_opt.c
 * @brief Manually optimized matrix expression solver.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using pointer 
 * arithmetic, register accumulation, and loop invariant code motion to 
 * improve cache utilization and execution speed.
 * Domain: HPC Performance Tuning.
 */

#include "utils.h"


/**
 * @brief Optimized matrix solver kernel.
 * Optimization: Replaces indirect array access with direct pointer incrementing 
 * and promotes high-frequency sum variables to registers.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	
	double *AB = (double *) calloc(N * N, sizeof(double));
	double *ABBt = (double *) calloc(N * N, sizeof(double));
	double *AtA = (double *) calloc(N * N, sizeof(double));
	double *RES = (double *) calloc(N * N, sizeof(double));
	register int i, j, k;

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Pointer-based traversal for matrix A (pa) and matrix B (pb). 
	 * Invariant: k starts at i, leveraging A's upper triangular properties.
	 */
	for (i = 0; i < N; i++) {
		double *pa_orig = A + i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			double *pa = pa_orig + i;
			double *pb = B + i * N + j;
			for (k = i; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb += N;
			}
			AB[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Compute ABBt = AB * B^T.
	 * Optimization: Linear memory sweep of AB (pab) and sequential sweep of B (pb).
	 */
	for (i = 0; i < N; i++) {
		double *pab_orig = AB + i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			double *pab = pab_orig;
			double *pb = B + j * N;
			for (k = 0; k < N; k++) {
				sum += *pab * *pb;
				pab++;
				pb++;
			}
			ABBt[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Compute AtA = A^T * A.
	 * Optimization: Accumulates column-wise products into a local register.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			double *pa1 = A + i;
			double *pa2 = A + j;
			for (k = 0; k < N; k++) {
				sum += *pa1 * *pa2;
				pa1 += N;
				pa2 += N;
			}
			AtA[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Result consolidation pass.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			RES[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);
	return RES;
}
