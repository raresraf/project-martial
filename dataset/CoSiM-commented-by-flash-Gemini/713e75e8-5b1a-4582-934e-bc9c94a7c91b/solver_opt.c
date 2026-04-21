/**
 * @713e75e8-5b1a-4582-934e-bc9c94a7c91b/solver_opt.c
 * @brief Manually optimized matrix expression solver.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using pointer 
 * arithmetic, register accumulation, and specialized traversal for triangular 
 * and Gramian matrices.
 * Domain: HPC Performance Tuning.
 */

#include "utils.h"


/**
 * @brief Optimized matrix solver kernel.
 * Optimization: Minimizes address calculation overhead by using linear pointer 
 * incrementing and promoting loop-local variables to hardware registers.
 */
double* my_solver(int N, double *A, double* B) {
	
	double *AB = calloc(sizeof(double), N * N);
	double *ABBt = calloc(sizeof(double), N * N);
	double *AtA = calloc(sizeof(double), N * N);
	double *RES = calloc(sizeof(double), N * N);

	register int i, j, k;

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Employs `pa` and `pb` pointers for efficient cache access.
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
	 * Optimization: Pointer-based linear sweep for `AB` and stride-N sweep for `B`.
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
	 * Optimization: Scalar register `sum` for dot products of A's columns.
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
	 * Block Logic: Consolidate term results.
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
