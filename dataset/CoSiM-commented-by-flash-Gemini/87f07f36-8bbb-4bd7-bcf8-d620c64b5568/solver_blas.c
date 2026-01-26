
#include "utils.h"
#include <cblas.h>


double* my_solver(int N, double *A, double *B) {
	double *C = calloc(N * N, sizeof(double));
	double *temp = calloc(N * N, sizeof(double));
	int i, j;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = B[i * N + j];
			temp[i * N + j] = A[i * N + j];
		}
	}

	
	cblas_dtrmm(101, 141, 121, 111, 131, N, N, 1.0, A, N, C, N);

	
	cblas_dtrmm(101, 141, 121, 112, 131, N, N, 1.0, A, N, temp, N);

	
	cblas_dgemm(101, 111, 112, N, N, N, 1.0, C, N, B, N, 1.0, temp, N);
	
	free(C);

	return temp;
}
