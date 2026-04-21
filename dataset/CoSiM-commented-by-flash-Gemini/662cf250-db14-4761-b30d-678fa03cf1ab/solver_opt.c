/**
 * @662cf250-db14-4761-b30d-678fa03cf1ab/solver_opt.c
 * @brief Manually optimized matrix expression solver.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using pointer 
 * arithmetic, register accumulation, and loop invariant code motion to 
 * enhance computational density.
 * Domain: HPC Performance Tuning.
 */

#include "utils.h"
#include <string.h>
#include <stdlib.h>


/**
 * @brief Optimized matrix solver kernel.
 * Optimization: Replaces array subscripts with pointer-based traversal and 
 * promotes intermediate sum variables to hardware registers.
 */
double* my_solver(int N, double *A, double* B) {
	
	double *AB = (double*)malloc(N * N * sizeof(double));
	double *ABBt = (double*)malloc(N * N * sizeof(double));
	double *AtA = (double*)malloc(N * N * sizeof(double));
	double *Res = (double*)malloc(N * N * sizeof(double));
	register int i, j, k;

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Pointer-based linear sweep for A (pa) and column-major sweep for B (pb).
	 * Invariant: Scalar register `sum` minimizes memory store operations.
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
	 * Optimization: Sequential sweep for AB and stride-N sweep for B.
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
	 * Optimization: Accumulates products of elements from columns of A.
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
	 * Block Logic: Result aggregation pass.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			Res[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);

	return Res;
}
