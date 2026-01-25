
#include "utils.h"
#include "cblas.h"
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	double *AB = (double *)calloc(N * N, sizeof(double));
	double *AtA = (double *) calloc(N * N, sizeof(double));

	memcpy(AB, B, N * N * sizeof(double));
	
	cblas_dtrmm (CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit , N, N, 1.0, A, N, AB, N);
	memcpy(AtA, A, N * N * sizeof(double));
	
	cblas_dtrmm (CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit , N, N, 1.0, A, N, AtA, N);
    cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, AtA, N);
	
	free(AB);
	return AtA;
}
