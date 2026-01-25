
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	int i, j;

	double* C = calloc(N * N, sizeof(double));
	double* prod1 = calloc(N * N, sizeof(double));
	double* result = calloc(N * N, sizeof(double));
	if (C == NULL || prod1 == NULL || result == NULL) {
        perror("calloc failed\n");
        exit(EXIT_FAILURE);
    }

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = B[i * N + j];
			result[i * N + j] = A[i * N + j];
		}
	}

	
	
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
				CblasNonUnit, N, N, 1.0, A, N, C, N);

	
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
				CblasNonUnit, N, N, 1.0, A, N, result, N);

	
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1.0, C, N, B, N, 1.0, result, N);

	
	free(prod1);
	free(C);
	return result;
}