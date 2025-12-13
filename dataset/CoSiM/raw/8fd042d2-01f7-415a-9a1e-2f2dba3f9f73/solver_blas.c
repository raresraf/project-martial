
#include "utils.h"


#include "cblas.h"
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *temp1 = (double *)calloc(N * N, sizeof(double));
	double *temp2 = (double *)calloc(N * N, sizeof(double));
	double *temp3 = (double *)calloc(N * N, sizeof(double));
	double *res = (double *)calloc(N * N, sizeof(double));
	int i, j;
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			int index = i * N + j;
			temp1[index] = A[index];
			temp2[index] = B[index];
		}
	}
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
				N, N, 1, A, N, temp2, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
				N, N, 1, A, N, temp1, N);
				
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, temp2, N, B, N, 0, temp3, N);
	
	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			res[i * N + j] = temp3[i * N + j] + temp1[i * N + j];
		}
	}
	free(temp1);
	free(temp2);
	free(temp3);
	return res;
}
