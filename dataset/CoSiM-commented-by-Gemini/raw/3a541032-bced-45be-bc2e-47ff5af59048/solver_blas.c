/*
 * Module: @raw/3a541032-bced-45be-bc2e-47ff5af59048/solver_blas.c
 * High-level purpose: BLAS solver.
 */
#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
	
	int i = 0;
	int j = 0;

	double *auxB = (double *)calloc(N * N, sizeof(double));
	for(i = 0; i < N * N; i++)
                auxB[i] = B[i];
	
	double *C = (double *)calloc(N * N, sizeof(double));
        for(i = 0; i < N * N; i++)
                C[i] = B[i];
	
	double *D = (double *)calloc(N * N, sizeof(double));
        for(i = 0; i < N * N; i++)
                D[i] = A[i];
	
	cblas_dtrmm(CblasRowMajor, CblasLeft,
                CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, auxB, N);
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, auxB, N, B, N, 0, C, N);
	
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				N, N, N, 1, A, N, A, N, 0, D, N);

	/* Pre-condition: Matrices C and D computed. Invariant: N size constraints check. */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] += D[i * N + j];
		}
	}

	free(auxB);
	free(D);	
	return C;
}
