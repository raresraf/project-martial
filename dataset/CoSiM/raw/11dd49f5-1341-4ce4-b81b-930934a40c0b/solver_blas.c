
#include "utils.h"
#include "cblas.h"
#include <string.h>



double* my_solver(int N, double *A, double *B) {
	register double *ab = (double *) malloc(N * N * sizeof(double));
	register double *ata = (double *) malloc(N * N * sizeof(double));
	register double *abbt = (double *) malloc(N * N * sizeof(double));
	register double *result = (double *) malloc(N * N * sizeof(double));

	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			ata[i * N + j] = A[i * N + j];
		}
	}
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
				CblasNonUnit, N, N, 1, A, N, ata, N);

	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			ab[i * N + j] = B[i * N + j];		
		}
	}
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
				CblasNonUnit, N, N, 1, A, N, ab, N);
	
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, ab, N, B, N, 0, abbt, N);


	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			result[i * N + j] = abbt[i * N + j] + ata[i * N + j];
		}
	}


	free(ab);
	free(abbt);
	free(ata);

	return result;
}
