/**
 * @file solver_blas.c
 * @brief A matrix equation solver using a BLAS (Basic Linear Algebra Subprograms) implementation.
 * @details This file contains a function, my_solver, that computes the result of the
 * matrix expression C = (A * B) * B' + (A' * A), where A and B are input matrices,
 * A' and B' are their transposes, and A is assumed to be upper triangular. The
 * computation is optimized by using high-performance BLAS library functions.
 *
 * @algorithm The computation is decomposed into three BLAS level-3 operations:
 * 1. C = A' * A (achieved via dtrmm, as C is initialized with A's content)
 * 2. AB = A * B (dtrmm)
 * 3. C = 1.0 * AB * B' + 1.0 * C (dgemm)
 *
 * @time_complexity O(N^3), dominated by the three matrix-matrix multiplications.
 * @space_complexity O(N^2) to store the intermediate matrix AB and the result matrix C.
 */
#include "utils.h"
#include "cblas.h"

/**
 * @brief Solves the matrix equation C = (A * B) * B' + (A' * A) using BLAS routines.
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the matrix A, stored in row-major order. Assumed to be upper triangular.
 * @param B A pointer to the matrix B, stored in row-major order.
 * @return A pointer to the resulting matrix C. The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double *B) {
	int i, n;
	double *C;
	double *AB;
	n = N * N;
	
	// Allocate memory for the result matrix C.
	C = calloc(N * N, sizeof(double));
	if (C == NULL)
		return NULL;
	
	// Allocate memory for the intermediate matrix AB = A * B.
	AB = calloc(N * N, sizeof(double));
	if (AB == NULL)
		return NULL;
	
	// Copy input matrices A and B into C and AB respectively to preserve the originals,
	// as BLAS operations can modify input arrays.
	for (i = 0; i < n; i++) {
		C[i] = A[i];
		AB[i] = B[i];
	}
	
	// Operation 1: C = A' * C (where C is initially A). This computes C = A' * A.
	// cblas_dtrmm performs a triangular matrix-matrix multiplication.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
	CblasNonUnit, N, N, 1.0, A, N, C, N);
	
	// Operation 2: AB = A * AB (where AB is initially B). This computes AB = A * B.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, 
	CblasNonUnit, N, N, 1.0, A, N, AB, N);
	
	// Operation 3: C = 1.0 * (AB) * B' + 1.0 * C.
	// Since AB = A * B and C = A' * A, this computes the final result:
	// C = (A * B) * B' + (A' * A).
	// cblas_dgemm performs a general matrix-matrix multiplication.
	cblas_dgemm(CblasRowMajor, CblasNoTrans,
                 CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, C, N);
	
	// Free the memory used for the intermediate matrix.
	free(AB);
	
	return C;
}