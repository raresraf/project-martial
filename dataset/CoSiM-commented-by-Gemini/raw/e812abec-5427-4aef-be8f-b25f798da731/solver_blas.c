/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include <cblas.h>
#include <string.h>
#include "utils.h"


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *result1 = calloc(N * N, sizeof(double));
	memcpy(result1, B, N * N * sizeof(double));
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, result1, N);
	
	double *result2 = calloc(N * N, sizeof(double));
	memcpy(result2, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, result2, N);
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, result1, N, B, N, 1.0, result2, N);
	free(result1);
	return result2;
}
