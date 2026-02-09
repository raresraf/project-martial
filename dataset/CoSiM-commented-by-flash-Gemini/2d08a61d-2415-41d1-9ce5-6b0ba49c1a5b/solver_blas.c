/**
 * @file solver_blas.c
 * @brief Semantic documentation for solver_blas.c. 
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */

#include "utils.h"
#include <cblas.h>


double *my_solver(int N, double *A, double *B)
{
	int i;
	double *C = (double *)calloc(N * N, sizeof(double));
	double *A_B = (double *)calloc(N * N, sizeof(double));
	double *unitMat = (double *)calloc(N * N, sizeof(double));

	for (i = 0; i < N; i++)
	{
		unitMat[i * N + i] = 1;
	}

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, unitMat, N, A, N, 0, C, N);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, unitMat, N, B, N, 0, A_B, N);

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, A_B, N);
	

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, C, N);
	

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, A_B, N, B, N, 1, C, N);
	

	free(A_B);
	free(unitMat);

	printf("BLAS SOLVER\n");
	return C;
}