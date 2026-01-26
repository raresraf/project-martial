
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *AAt;
	double *BBt;
	register int i, j;
	double alpha = 1.0;

	enum CBLAS_ORDER layout;
	enum CBLAS_TRANSPOSE transa;
	enum CBLAS_TRANSPOSE transat;
	enum CBLAS_SIDE side;
	enum CBLAS_UPLO upper;

	upper = CblasUpper;
	side = CblasLeft;
	layout = CblasRowMajor;
	transa = CblasNoTrans;
	transat = CblasTrans;

	
	AAt = calloc(N * N, sizeof(*AAt));
	BBt = calloc(N * N, sizeof(*BBt));

	
	for(i = 0; i < N; i++) {
		double *aa_ptr = AAt + i * N;
		double *a_ptr = A + i * N;
		for(j = 0; j < N; j++) {
			*aa_ptr = *a_ptr;
			aa_ptr++;
			a_ptr++; 
		}
	}

	
	for(i = 0; i < N; i++) {
		double *bb_ptr = BBt + i * N;
		double *b_ptr = B + i * N;
		for(j = 0; j < N; j++) {
			*bb_ptr = *b_ptr;
			bb_ptr++;
			b_ptr++; 
		}
	}
	
	
	cblas_dtrmm(layout, side, upper, transat, CblasNonUnit, N, N, alpha, A, N, AAt, N);

	
	cblas_dtrmm(layout, side, upper, transa, CblasNonUnit, N, N, alpha, A, N, BBt, N);
	
	
	cblas_dgemm(layout, transa, transat, N, N, N, alpha, BBt, N, B, N, alpha, AAt, N);
	
	free(BBt);

	return AAt;
}
