/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include "utils.h"
#include <cblas.h>
#include <string.h>


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {
	double *AB, *C, *AtA;

	AB = malloc(N * N * sizeof(double));
	memcpy(AB, B, N * N * sizeof(double));
	C = calloc(N * N, sizeof(double));
	AtA = malloc(N * N * sizeof(double));
	memcpy(AtA, A, N * N * sizeof(double));

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit, 
		N, N,
		1.0,
		A, N, AB, N);
	
	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit, 
		N, N,
		1.0,
		A, N, AtA, N);
	
	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1.0,
		AB, N,
		B, N, 1.0,
		AtA, N);

	memcpy(C, AtA, N * N * sizeof(double));

	free(AB);
	free(AtA);
	return C;
}
