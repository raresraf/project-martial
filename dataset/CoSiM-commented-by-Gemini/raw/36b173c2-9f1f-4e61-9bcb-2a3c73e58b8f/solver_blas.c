
#include "utils.h"
#include "cblas.h"
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *C = (double*) calloc(N * N, sizeof(double));
	if (C == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	double *res_A = (double*) calloc(N * N, sizeof(double));
	if (res_A == NULL) {
		printf("Failed calloc!");
		return NULL;
	}

	
	memcpy(C, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, 141, 121, 111, 131, N, N, 1, A, N, C, N);
	
	memcpy(res_A, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, 141, 121, 112, 131, N, N, 1, A, N, res_A, N); 
	
	cblas_dgemm(CblasRowMajor, 111, 112, N, N, N, 1, C, N, B, N, 1, res_A, N);

	free(C);
	return res_A;
}
