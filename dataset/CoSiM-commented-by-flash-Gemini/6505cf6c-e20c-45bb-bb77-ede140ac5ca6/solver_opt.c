/**
 * @6505cf6c-e20c-45bb-bb77-ede140ac5ca6/solver_opt.c
 * @brief Manually optimized matrix expression solver.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using pointer 
 * arithmetic, register-based accumulation, and loop invariant code motion.
 * Domain: HPC Performance Tuning.
 */

#include "utils.h"


/**
 * @brief Optimized matrix solver kernel.
 * Optimization: Minimizes subscript calculation overhead by using linear pointer 
 * incrementing and promoting loop-local variables to registers.
 */
double* my_solver(int N, double *A, double* B) {
	double *AB;
	double *ABB_t;
	double *AtA;
	double *C;
	register int i, j, k;

	AB = calloc(N * N, sizeof(double));
	ABB_t = calloc(N * N, sizeof(double));
	AtA = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Pointer-based traversal for matrix A (pa) and matrix B (pb). 
	 * Employs a local register `sum` to reduce memory traffic.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			double *pa = A + i * N + i;
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
	 * Block Logic: Compute ABB_t = AB * B^T.
	 * Optimization: Sequential sweep for AB and stride-N sweep for B.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			register double sum = 0.0;
			double *pab = AB + i * N;
			double *pb = B + j * N;
			for (k = 0; k < N; k++) {
				sum += *pab * *pb;
				pab++;
				pb++;
			}
			ABB_t[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Compute AtA = A^T * A.
	 * Optimization: Local register accumulation for column-column products.
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
	 * Block Logic: Aggregation pass.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = ABB_t[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABB_t);
	free(AtA);
	return C;
}
