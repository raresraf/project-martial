
#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {

	double *AB = calloc(N * N, sizeof(double));
	cblas_dcopy(N * N, B, 1, AB, 1);
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
	N, N, 1.0, A, N, AB, N);

	double *AtA = calloc(N * N, sizeof(double));
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, N, 1.0, A, N,
	A, N, 1.0, AtA, N);

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N,
	B, N, 1.0, AtA, N);

	free(AB);
	return AtA;
}
