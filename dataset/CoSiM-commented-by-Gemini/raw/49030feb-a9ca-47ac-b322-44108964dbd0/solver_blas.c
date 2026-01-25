
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *C = malloc(N * N * sizeof(double));
	double *mat = malloc(N * N * sizeof(double));
	
	for(int i = 0; i < N; i++){
		double *idx = &B[i * N];
		double *id = &mat[i * N];
		for(int j = 0; j < N; j++) {
			*id = *idx;
			id++;
			idx++;
		}
	}

	double *point_C = &C[0];
	for(int i = 0; i < N * N; i++){
		*point_C = 0;
		point_C++;
	}

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, 
				CblasNonUnit, N, N, 1.0, A, N, mat, N);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 
				1.0, mat, N, B, N, 1.0, C, N);
	
	for(int i = 0; i < N; i++){
		double *idx = &A[i * N];
		double *id = &mat[i * N];
		for(int j = 0; j < N; j++) {
			*id = *idx;
			id++;
			idx++;
		}
	}

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, 
				CblasNonUnit, N, N, 1.0, mat, N, mat, N);
	cblas_daxpy(N * N, 1.0, mat, 1, C, 1);
	free(mat);
	return C;
}
