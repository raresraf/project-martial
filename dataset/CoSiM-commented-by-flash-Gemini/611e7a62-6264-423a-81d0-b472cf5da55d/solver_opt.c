/**
 * @611e7a62-6264-423a-81d0-b472cf5da55d/solver_opt.c
 * @brief Manually optimized implementation of a matrix expression solver.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using pointer 
 * arithmetic, register accumulation, and loop transformation for cache efficiency.
 * Domain: HPC Performance Tuning.
 */

#include "utils.h"


/**
 * @brief Optimized matrix solver kernel.
 * Optimization: Minimizes address calculation overhead by utilizing linear pointer 
 * incrementing and promoting accumulators to hardware registers.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double *C;
	double *D;
	double *E;
	register int i, j, k;
	register int size = N*N;
	
	C = calloc(size, sizeof(double));
	D = calloc(size, sizeof(double));
	E = calloc(size, sizeof(double));

	/**
	 * Block Logic: Compute C = A * B.
	 * Optimization: Pointer-based traversal of matrix A (row-major) and matrix B (column-major).
	 * Invariant: Scalar `suma` reduces memory store operations per k-iteration.
	 */
	for(i = 0; i < N; i++){
		double *orig_pa = &A[i * N];
		for (j = 0; j < N; j++) {
			register double *pa = orig_pa;
			register double *pb = &B[j];
			register double suma = 0;
			int var = 0;
			if (i >= j) {
				var = i;
			}
			pa += var;
			pb += var * N;
			for (k = var; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			C[i * N + j] = suma;
		}
	}

	/**
	 * Block Logic: Compute D = C * B^T.
	 * Optimization: Linear memory sweep of matrix C and transposed-access to B.
	 */
	for(i = 0; i < N; i++) {
		double *orig_pc = &C[i * N];
		for(j = 0; j < N; j++) {
			register double *pc = orig_pc;
			register double *pb1 = &B[j * N];
			register double sum = 0;
			for(k = 0; k < N; k++) {
				sum += *pc * *pb1;
				pc++;
				pb1++;
			}
			D[i * N + j] = sum;
		}
	}

	/**
	 * Block Logic: Compute E = D + (A^T * A).
	 * Optimization: Pointer arithmetic for Gramian matrix calculation.
	 */
	for(i = 0; i < N; i++){
		for (j = 0; j < N; j++) {
		register double *pa1 = &A[i];
		register double *pa2 = &A[j];
		register double sum3 = 0;	
		for (k = 0; k <= j; k++) {
			sum3 += *pa1 * *pa2;
			pa1 += N;
			pa2 += N;
		}
		E[i * N + j] = D[i * N + j] + sum3;
		}
	}

	free(C);
	free(D);

 	return E;	
}
