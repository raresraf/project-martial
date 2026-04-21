/**
 * @66423df7-06af-4ccd-ac5c-54f5e8b29793/solver_opt.c
 * @brief Manually optimized matrix expression solver.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using pointer 
 * arithmetic, register-based accumulation, and loop invariant code motion.
 * Domain: HPC Performance Tuning.
 */

#include "utils.h"


/**
 * @brief Optimized matrix solver kernel.
 * Optimization: Replaces indirect indexing with pointer-based traversal to 
 * minimize address calculation overhead. Utilizes registers for local sum 
 * storage to reduce cache write-back frequency.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	
	double *AB = (double *)calloc(N * N, sizeof(double));
	double *ABBt = (double *)calloc(N * N, sizeof(double));
	double *AtA = (double *)calloc(N * N, sizeof(double));
	double *RES = (double *)calloc(N * N, sizeof(double));
	register int i, j, k;

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Employs `pa` and `pb` pointers for linear matrix sweep. 
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
	 * Optimization: Uses a local register `sum` for dot product accumulation.
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
	 * Logic: Column-wise dot product of matrix A.
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
	 * Block Logic: Summation of the two primary compute terms.
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
