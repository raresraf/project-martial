
#include "utils.h"
#include <string.h>
#include <cblas.h>


double *sum(int N, double *X, double *Y) {
	double *identity = calloc(N * N, sizeof(double));\
	double *result = malloc(N * N * sizeof(double));
	memcpy(result, Y, N * N * sizeof(double));
	for(int i = 0; i < N; i++) {
		identity[i * N + i] = 1;
	}
	int alpha = 1.0, beta = 1.0;
	cblas_dgemm(CblasRowMajor, CblasNoTrans,
                 CblasNoTrans, N, N,
                 N, alpha, X,
                N, identity, N,
                 beta, result, N);
	free(identity);
	return result;
}


double *mul_B_BT(int N, double *B) {
	double *mul_result = malloc(N * N * sizeof(double));
	double alpha = 1.0, beta = 0.0;
	cblas_dgemm(CblasRowMajor, CblasNoTrans,
				CblasTrans, N, N, N, alpha,
				B, N, B, N, beta, mul_result, N);
	return mul_result;
}


double *mul_A_MAT(int N, double *A, double *MAT) {
	double *result = malloc(N * N * sizeof(double));
	memcpy(result, MAT, N * N * sizeof(double));
	double alpha = 1.0;
	cblas_dtrmm(CblasRowMajor, CblasLeft,
                 CblasUpper, CblasNoTrans,
                 CblasNonUnit, N, N,
                 alpha, A, N,
                 result, N);
	return result;
}


double *mul_AT_A(int N, double *A) {
	double alpha = 1.0;
	double *A_CPY = malloc(N * N * sizeof(double));
	memcpy(A_CPY, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft,
                 CblasUpper, CblasTrans,
                 CblasNonUnit, N, N,
                 alpha, A, N,
                 A_CPY, N);
	return A_CPY;
}

double* my_solver(int N, double *A, double *B) {
	double *B_X_BT = mul_B_BT(N, B);
	double *A_X_B_X_BT = mul_A_MAT(N, A, B_X_BT);
	double *AT_X_A = mul_AT_A(N, A);
	double *result = sum(N, A_X_B_X_BT, AT_X_A);
	free(B_X_BT);
	free(A_X_B_X_BT);
	free(AT_X_A);
	return result;
}
