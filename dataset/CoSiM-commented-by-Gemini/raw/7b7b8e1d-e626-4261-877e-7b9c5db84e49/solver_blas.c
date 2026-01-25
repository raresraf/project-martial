
#include "utils.h"
#include "cblas.h"



double* my_solver(int N, double *A, double *B) {

	double *AB = calloc(N * N, sizeof(double));

	for(int i = 0; i < N * N; i++)
		AB[i] = B[i];

	
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, AB, N);

	

	double *ABB = calloc(N * N, sizeof(double));
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, AB, N, B, N, 1, ABB, N);

	
	
	double *AA = calloc(N * N, sizeof(double));
	
	for(int i = 0; i < N * N; i++)
		AA[i] = A[i];
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, AA, N);

	

	double *res = calloc(N * N, sizeof(double));

	for (int i = 0; i < N * N; i++){
		res[i] = ABB[i] + AA[i];
	}

	
	
	free(AB);
	free(ABB);
	free(AA);

	return res;
	}
