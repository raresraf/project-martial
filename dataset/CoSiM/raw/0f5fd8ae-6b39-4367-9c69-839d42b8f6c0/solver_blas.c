
#include <string.h>

#include "utils.h"
#include "cblas.h"




double* AB(int N, double *A, double *B) {
	double *AB;
	
	AB = calloc(N * N, sizeof(double));
	if (!AB) {
		exit(EXIT_FAILURE);
	}

	
	memcpy(AB, B, N * N * sizeof(double));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N, 1.0, A, N,
		AB, N);

	return AB;
}


double* A_tA(int N, double *A_t, double *A) {
	double *A_tA;
	
	A_tA = calloc(N * N, sizeof(double));
	if (!A_tA) {
		exit(EXIT_FAILURE);
	}

	
	memcpy(A_tA, A, N * N * sizeof(double));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N, 1.0, A, N,
		A_tA, N);

	return A_tA;
}


double* ABB_t(int N, double *AB, double *B, double *A_tA) {
	double *sum;
	
	sum = calloc(N * N, sizeof(double));
	if (!sum) {
		exit(EXIT_FAILURE);
	}

	
	memcpy(sum, A_tA, N * N * sizeof(double));

	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N, 1.0, AB,
		N, B, N, 1.0, sum, N);

	return sum;
}

double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *ptr_AB = AB(N, A, B);
	double *ptr_A_tA = A_tA(N, A, A);
	double *ptr_ABB_t = ABB_t(N, ptr_AB, B, ptr_A_tA);

	free(ptr_AB);
	free(ptr_A_tA);

	return ptr_ABB_t;
}
