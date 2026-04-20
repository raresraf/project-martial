/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */


#include "cblas.h"
#include "utils.h"



double* make_copy(int N, double *mat) {
	int i, j;

	double *copy_mat = (double *) calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (copy_mat == NULL) {
		return NULL;
	}

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; ++i) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; ++j) {
			copy_mat[i * N + j] = mat[i * N + j];
		}
	}

	return copy_mat;
}



/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {

	

	
	double *M = make_copy(N, B);
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
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
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
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
