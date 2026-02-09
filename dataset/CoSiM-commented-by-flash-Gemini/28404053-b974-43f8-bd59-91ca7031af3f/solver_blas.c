/**
 * @file solver_blas.c
 * @brief Semantic documentation for solver_blas.c. 
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */

#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
	
	register int i = 0;
	register int j = 0;

	double *fst = (double *)malloc(N * N * sizeof(double)); 
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				N, N, N, 1, A, N, A, N, 0, fst, N);
	
	double *tmp = (double *)malloc(N * N * sizeof(double));

	for(i = 0; i < N ; i++) {
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

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			snd[i * N  + j] += fst[i * N  + j];
		}
	}

	free(tmp);
	free(fst);	
	return snd; 
}
