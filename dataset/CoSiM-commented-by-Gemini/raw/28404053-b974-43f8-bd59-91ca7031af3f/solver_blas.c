/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver.
 * Relies on highly tuned vendor libraries for maximum SIMD/cache utilization.
 * Performs C = AtA + ABBt efficiently.
 */

#include "utils.h"
#include "cblas.h"

/**
 * @brief Computes C = At * A + A * B * Bt.
 * Allocates memory dynamically and executes matrix operations.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {
	
	register int i = 0;
	register int j = 0;

	double *fst = (double *)malloc(N * N * sizeof(double)); 
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				N, N, N, 1, A, N, A, N, 0, fst, N);
	
	double *tmp = (double *)malloc(N * N * sizeof(double));

	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for(i = 0; i < N ; i++) {
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; j++)
		{
			tmp[i * N + j] = B[i * N + j];
		}
	}
   	 cblas_dtrmm(CblasRowMajor, CblasLeft,
                CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, tmp, N);

	double *snd = (double *)malloc(N * N * sizeof(double));
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, tmp, N, B, N, 0, snd, N);

	/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
	for (i = 0; i < N; ++i) {
		/* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
		for (j = 0; j < N; ++j) {
			snd[i * N  + j] += fst[i * N  + j];
		}
	}

	free(tmp);
	free(fst);	
	return snd; 
}
