
#include "utils.h"
#include "cblas.h"
#include <string.h>



double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	
	
	double *rezPartial1 = (double *) calloc(N * N, sizeof(double));
	double *rezPartial2 = (double *) calloc(N * N, sizeof(double));
	double *rez = (double *) calloc(N * N, sizeof(double));
	enum CBLAS_ORDER order;
    enum CBLAS_TRANSPOSE transA, transB;
    enum CBLAS_SIDE side;
    enum CBLAS_UPLO uplo;
    enum CBLAS_DIAG diag;
    int lda, ldb, ldc;
    double alpha, beta;

    memcpy(rezPartial1, A, N*N*sizeof(double));
	order = CblasRowMajor;
	side = CblasLeft;
	uplo = CblasUpper;
	transA = CblasTrans;
	diag = CblasNonUnit;
	lda = N;
	alpha = 1;

	
	
	cblas_dtrmm(order, side, uplo, transA, diag, N, N, alpha, A, lda, rezPartial1, lda);

	memcpy(rezPartial2, B, N*N*sizeof(double));
	order = CblasRowMajor;
	side = CblasLeft;
	uplo = CblasUpper;
	transA = CblasNoTrans;
	diag = CblasNonUnit;
	lda = N;
	alpha = 1;

	
	
	cblas_dtrmm(order, side, uplo, transA, diag, N, N, alpha, A, lda, rezPartial2, lda);

	memcpy(rez, rezPartial1, N*N*sizeof(double));
	order = CblasRowMajor;
	transA = CblasNoTrans;
	transB = CblasTrans;
	lda = N;
	ldb = N;
	ldc = N;
	alpha = 1;
	beta = 1;
	
	
	
	cblas_dgemm(order, transA, transB, N, N, N, alpha, rezPartial2, lda, 
				B, ldb, beta, rez, ldc);	

	free(rezPartial1);
	free(rezPartial2);
	return rez;
}