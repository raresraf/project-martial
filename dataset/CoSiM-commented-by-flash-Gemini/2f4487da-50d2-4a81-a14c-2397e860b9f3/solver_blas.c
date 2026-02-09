/**
 * @file solver_blas.c
 * @brief Semantic documentation for solver_blas.c.
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */

#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *C = (double *)malloc(N * N * sizeof(double));
    double *D = (double *)malloc(N * N * sizeof(double));

    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N,
    	1.0, B, N, B, N, 0.0, C, N);
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
    	CblasNonUnit, N, N, 1.0, A, N, C, N);

    
    cblas_dcopy(N * N, A, 1, D, 1);
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
    	CblasNonUnit, N, N, 1.0, A, N, D, N);

    
    cblas_daxpy(N * N, 1.0, C, 1, D, 1);

    free(C);
	return D;
}
