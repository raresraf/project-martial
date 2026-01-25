
#include "utils.h"
#include "cblas.h"




double* my_solver(int N, double *A, double *B) {
	int i, j;
	double *prod_Bt_B = NULL;
	double *result = NULL;
	double *identity_matrix = NULL;
	double alpha = 1.0;
	double beta = 1.0;

	
	result = (double *)calloc(N * N, sizeof(double));
	if (result == NULL) {
		fprintf(stderr, "Error calloc tranpose.\n");
		exit(0);
	}

	for (i = 0; i < N; i++) {
		for (j = i; j < N; j++) {
			result[i * N + j] = A[i * N + j];
		}
	}

	cblas_dtrmm(CblasRowMajor, 
				CblasLeft, 
				CblasUpper, 
				CblasTrans, 
				CblasNonUnit, 
				N, N, 
				alpha,
				A, N, result, N); 

	
	prod_Bt_B = (double *)calloc(N * N, sizeof(double));
	if (prod_Bt_B== NULL) {
		fprintf(stderr, "Error calloc tranpose.\n");
		exit(0);
	}

	cblas_dgemm(CblasRowMajor, 
				CblasNoTrans, 
				CblasTrans, 
				N, N, N, 
				alpha, B, N, 
				B, N,
				beta, prod_Bt_B, N);

	
	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasNoTrans,
				CblasNonUnit,
				N, N,
				alpha,
				A, N, prod_Bt_B, N);


	
	identity_matrix = (double *)calloc(N * N, sizeof(double));
	if (identity_matrix == NULL) {
		fprintf(stderr, "Error calloc tranpose.\n");
		exit(0);
	}

	for (i = 0; i < N; i++) {
		identity_matrix[i * N + i] = 1.0;
	}

	cblas_dgemm(CblasRowMajor,
				CblasNoTrans, 
				CblasNoTrans, 
				N, N, N,
				alpha, prod_Bt_B, N,
				identity_matrix, N, 
				beta, result, N);

	free(identity_matrix);
	free(prod_Bt_B);
	return result;
}
