/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include "utils.h"
#include "string.h"

#include "cblas.h"


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *C, *D;


	C = calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (C == NULL)
		exit(EXIT_FAILURE);

	D = calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Conditional state branch.
	 * Invariant: The conditional branch maintains control flow invariants.
	 */
	if (D == NULL)
		exit(EXIT_FAILURE);

	
	memcpy(D, B, N * N * sizeof(double));
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		D, N 
	);

	
	
	
	memcpy(C, A, N * N * sizeof(double));
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans, 
		CblasNonUnit,
		N, N,
		1.0, A, N,
		C, N 
	);


	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans, 
		CblasTrans,  
		N, N, N,
		1.0, D, N,
		B, N,
		1.0, C, N 
	);

	free(D);

	return C;
}
