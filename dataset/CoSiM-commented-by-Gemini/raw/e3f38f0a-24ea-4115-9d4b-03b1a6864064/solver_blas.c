/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include "utils.h"
#include "stdio.h"
#include "cblas.h"
#include <string.h>
#include <stdlib.h>

/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *AXB = (double *) malloc(N * N * sizeof(double));
	double *A_transposeXA = (double *) malloc(N * N * sizeof(double));

	memcpy(AXB, B, N * N * sizeof(double));
	memcpy(A_transposeXA, A, N * N * sizeof(double));

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
		AXB, 
		N
	);


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
		A_transposeXA,
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
		AXB,
		N,
		B,
		N,
		1.0,
		A_transposeXA,
		N
	);

	free(AXB);
	return A_transposeXA;
}
