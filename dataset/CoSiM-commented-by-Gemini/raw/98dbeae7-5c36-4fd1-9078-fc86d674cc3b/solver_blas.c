/**
 * @raw/98dbeae7-5c36-4fd1-9078-fc86d674cc3b/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Hardware-accelerated computation of C = A^T * A + A * (B * B^T).
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ representing distinct output and intermediate buffers.
 */

#include "utils.h"
#include <string.h>
#include <cblas.h>

double* my_solver(int N, double *A, double *B) {
	/**
	 * Functional Utility: Initializes dynamically allocated matrix buffers and caches.
	 */
	double *C = (double*)malloc(N * N * sizeof(double));
	double *D = (double*)malloc(N * N * sizeof(double));
	double *E = (double*)malloc(N * N * sizeof(double));
	
	/**
	 * Functional Utility: Transfers A's memory block onto C targeting the consecutive A^T * A.
	 */
	memcpy(C, A, N * N * sizeof(double));
	
	/**
	 * Functional Utility: Multiplies C by upper triangular A^T in-place yielding A^T * A.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
		N, N, 1.0, A, N, C, N);
	
	memcpy(D, B, N * N * sizeof(double));
	
	/**
	 * Functional Utility: General matrix multiplication mapping D * B^T solving B * B^T and depositing inside E.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N,
		1.0, D, N, B, N, 0.0, E, N);
	
	/**
	 * Functional Utility: Solves final structure combining previous segments evaluating C += A * E.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 
		1.0, A, N, E, N, 1.0, C, N);

	free(D);
	free(E);
	return C;
}
