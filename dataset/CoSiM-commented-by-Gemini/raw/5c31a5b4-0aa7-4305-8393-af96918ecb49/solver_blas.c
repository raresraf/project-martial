
#include "utils.h"
#include "cblas.h"
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *X = (double*) calloc(N * N, sizeof(double));
	if (X == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}
    memcpy(X, B, N*N*sizeof(double));
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, B, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, A, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, B, N, X, N, 1.0, A, N);
	
	double *Z = (double*) calloc(N * N, sizeof(double));
	if (Z == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}
	memcpy(Z, A, N*N*sizeof(double));
	free(X);
	return Z;
}
