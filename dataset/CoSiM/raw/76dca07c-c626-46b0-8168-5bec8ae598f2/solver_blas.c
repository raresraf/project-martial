
#include "utils.h"
#include<cblas.h>
#include<string.h>

double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *M2 = (double *) malloc(N * N * sizeof(double));
	double *M1 = (double *) calloc(N * N, sizeof(double));
	memcpy(M2, A, N * N * sizeof(double));

	
 	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, B, N, B, N, 1.0, M1, N);
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, M2, N);
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, M1, N, 1.0, M2, N);

	free(M1);
	return M2;



}
