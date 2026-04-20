/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include "utils.h"
#include "cblas.h"




/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {
	int i, j;
	double *prod_Bt_B = NULL; /* Inline: Non-obvious pointer arithmetic/dereference for optimized memory access */
	double *result = NULL;
	double *identity_matrix = NULL;
	double alpha = 1.0;
	double beta = 1.0;

	
	result = (double *)calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (result == NULL) {
		fprintf(stderr, "Error calloc tranpose.\n");
		exit(0);
	}

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
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
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
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
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (identity_matrix == NULL) {
		fprintf(stderr, "Error calloc tranpose.\n");
		exit(0);
	}

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
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
