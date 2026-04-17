/*
 * Module: @raw/7d87b802-1fe0-4a7d-911d-4c5f132c4e4b/solver_blas.c
 * Purpose: Matrix solver utilizing BLAS libraries for matrix operations.
 */
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double c[N][N], d[N][N];
	
    // Pre-condition: A and B are valid N*N BLAS matrices. Invariant: c is initialized from B.
	for(int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			c[i][j] = B[i * N + j];
		}
	}

	double *e = malloc(N * N * sizeof(double));

	double *orig_pc = &c[0][0];
	double *orig_pd = &d[0][0];

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, orig_pc, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                N, N, N, 1.0, orig_pc, N, B, N, 0.0, orig_pd, N);

	// Pre-condition: e allocated, A is valid. Invariant: e is initialized from A.
	for(int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			e[i * N + j] = A[i * N + j];
		}
	}
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
                N, N, 1.0, A, N, e, N);

	
	cblas_daxpy(N * N, 1.0, orig_pd, 1, e, 1);	
	
	return e;
}