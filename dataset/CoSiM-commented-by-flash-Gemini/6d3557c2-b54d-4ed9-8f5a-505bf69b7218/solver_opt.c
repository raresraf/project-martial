/**
 * @6d3557c2-b54d-4ed9-8f5a-505bf69b7218/solver_opt.c
 * @brief Manually optimized matrix expression solver.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using pointer 
 * arithmetic, register-based accumulation, and loop invariant code motion to 
 * maximize compute-to-memory ratio.
 * Domain: HPC Performance Tuning.
 */

#include "utils.h"


/**
 * @brief Optimized matrix solver kernel.
 * Optimization: Minimizes address recalculation by using pointer increments 
 * and promoting loop-local sum variables to registers to reduce cache pressure.
 */
double* my_solver(int N, double *A, double* B) {
	
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
	 * Block Logic: Consolidate partial products into the result matrix.
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
