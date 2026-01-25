
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *C, *aux;
	C = (double *)malloc(N * N * sizeof(double));
	aux = (double *)malloc(N * N * sizeof(double));

	cblas_dcopy(N * N, B, 1, aux, 1); 
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, aux, N); 
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, aux, N, B, N, 0, C, N); 
	cblas_dcopy(N * N, A, 1, aux, 1); 
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, aux, N); 
	cblas_daxpy(N * N, 1, aux, 1, C, 1); 

	free(aux);

	return C;
}
