
#include "utils.h"
#include <cblas.h>

void copy(int N, double **copyM, double *M) {
	for(int i = 0; i < N*N; i++) {
		(*copyM)[i] = M[i];
	}
}


double* my_solver(int N, double *A, double *B) {
	double *AB = malloc((N * N) * sizeof(double));
	copy(N, &AB, B);
	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasNoTrans,
				CblasNonUnit,
				N, N, 1.0,
				A, N,
				AB, N);

	double *AtA = malloc((N * N) * sizeof(double));
	copy(N, &AtA, A);
	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasTrans,
				CblasNonUnit,
				N, N, 1.0,
				A, N,
				AtA, N);

	double *res = malloc((N * N) * sizeof(double));
	copy(N, &res, AtA);
	cblas_dgemm(CblasRowMajor,
				CblasNoTrans,
				CblasTrans,
				N, N, N, 
				1.0, 
				AB, N,
				B, N,
				1.0,
				res, N
				);

	free(AB);
	free(AtA);
	return res;
}
