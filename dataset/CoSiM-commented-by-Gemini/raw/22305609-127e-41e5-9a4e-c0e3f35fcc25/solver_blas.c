
#include "utils.h"
#include <cblas.h>
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	register size_t i, j;

	double *C = calloc(sizeof(double), N * N);
	double *tmp = calloc(sizeof(double), N * N);
	double *tmp2 = calloc(sizeof(double), N * N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, A, N, B, N, 0, C, N);
	
	memcpy(tmp2, C, N * N * sizeof(double));
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, tmp2, N, B, N, 0, C, N);
	
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, N, 1, A, N, A, N, 0, tmp, N);

	for (i = 0; i < N; ++i) {
		register double *pc = C + N * i;
		register double *ptmp = tmp + N * i;
		for (j = 0; j < N; ++j) {
			*pc += *ptmp;
			pc++, ptmp++;
		}
	}
	free(tmp);
	free(tmp2);

	return C;
}
