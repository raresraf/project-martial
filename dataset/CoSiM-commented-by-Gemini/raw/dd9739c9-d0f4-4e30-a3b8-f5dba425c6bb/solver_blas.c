
#include "utils.h"
#include <string.h>
#include "cblas.h"



double* my_solver(int N, double *A, double *B) {
	double* C = calloc(N * N, sizeof(double));
	double* X = calloc(N * N, sizeof(double));

	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasNoTrans,
		N, N, N,
		1.0, A, N,
		B, N, 0, 
		X, N
	);


	
	double* Y = calloc(N * N, sizeof(double));
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		112,
		N, N, N,
		1.0, X, N,
		B, N, 0, 
		Y, N
	);


	
	double* Z = calloc(N * N, sizeof(double));
	cblas_dgemm(
		CblasRowMajor,
		112,
		CblasNoTrans,
		N, N, N,
		1.0, A, N,
		A, N, 0, 
		Z, N
	);

	
	double* I = calloc(N * N, sizeof(double));
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			if(j == i) {
				I[i * N + j] = 1;
			}
		}
	}


	
	memcpy(C, Z, N * N * sizeof(double));
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasNoTrans,
		N, N, N,
		1.0, I, N,
		Y, N, 1.0, 
		C, N
	);
	free(X);
	free(Y);
	free(I);
	free(Z);
	return C;
}
