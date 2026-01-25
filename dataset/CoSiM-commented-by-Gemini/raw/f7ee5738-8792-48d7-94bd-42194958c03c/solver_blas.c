

#include "cblas.h"
#include "utils.h"



double* make_copy(int N, double *mat) {
	int i, j;

	double *copy_mat = (double *) calloc(N * N, sizeof(double));
	if (copy_mat == NULL) {
		return NULL;
	}

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			copy_mat[i * N + j] = mat[i * N + j];
		}
	}

	return copy_mat;
}



double* my_solver(int N, double *A, double *B) {

	

	
	double *M = make_copy(N, B);
	if (M == NULL) 
		return NULL;


	
	
	
	cblas_dtrmm(
		CblasRowMajor,		
		CblasLeft, 			
		CblasUpper,			
		CblasNoTrans,		
		CblasNonUnit,		
		N,
		N,
		1.0,				
		A,					
		N,
		M,					
		N 
	);



	double *P = make_copy(N, A);
	if (P == NULL) 
		return NULL;

	
	
	cblas_dtrmm(
		CblasRowMajor,		
		CblasLeft, 			
		CblasUpper,			
		CblasTrans,			
		CblasNonUnit,		
		N,
		N,
		1.0,				
		A,					
		N,
		P,					
		N
	);



	
	
	cblas_dgemm(
		CblasRowMajor,		
		CblasNoTrans,		
		CblasTrans,			
		N,
		N,
		N,
		1.0,				
		M,					
		N,
		B,					
		N,
		1.0,				
		P,					
		N
	); 


	free(M);

	return P;
}
