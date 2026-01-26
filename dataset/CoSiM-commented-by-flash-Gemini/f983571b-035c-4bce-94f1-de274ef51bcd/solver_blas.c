
#include "utils.h"
#include <string.h>
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	
	double *A_tA = malloc(N * N * sizeof(double));
	double *AB = malloc(N * N * sizeof(double));
	double *C = malloc(N * N * sizeof(double));

	
	memcpy(A_tA, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasTrans,
				CblasNonUnit,
				N,
				N,
				1.0,
				A,
				N,
				A_tA,
				N);

	
	
	memcpy(AB, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasNoTrans,
				CblasNonUnit,
				N,
				N,
				1.0,
				A,
				N,
				AB,
				N);

	
	memcpy(C, A_tA, N * N * sizeof(double));

	
	cblas_dgemm(CblasRowMajor,
				CblasNoTrans,
				CblasTrans,
				N,
				N,
				N,
				1.0,
				AB,
				N,
				B,
				N,
				1.0,
				C, N);
	
	
	free(A_tA);
	free(AB);

	return C;
}
