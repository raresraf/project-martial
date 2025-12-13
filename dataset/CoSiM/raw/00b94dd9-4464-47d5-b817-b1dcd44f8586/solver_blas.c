
#include "utils.h"
#include "cblas.h"
#include <string.h>


static double *get_transpose(double *M, int N)
{
	double *tr = calloc(N * N, sizeof(double));
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			tr[i * N + j] = M[j * N + i];
		}
	}
	return tr;
}


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *first_mul = calloc(N * N, sizeof(double));
	double *first_mul_aux = calloc(N * N, sizeof(double));
	double *At = get_transpose(A, N);
	double *Bt = get_transpose(B, N);
	
	memcpy(first_mul, A, N * N * sizeof(double));

	
	cblas_dtrmm( CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N, 1.0, B, N, first_mul, N);

	
	 cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
	 	N, N, N, 1, first_mul, N, Bt, N, 0, first_mul_aux, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N, 1.0, A, N, At, N);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			first_mul_aux[i * N + j] += At[i * N + j];
		}
	}

	
	free(first_mul);
	free(At);
	free(Bt);
	return first_mul_aux;
}
