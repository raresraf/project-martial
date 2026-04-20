/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include "utils.h"
#include <string.h>
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
	
	
	double *A_tA = malloc(N * N * sizeof(double));
	double *AB = malloc(N * N * sizeof(double));
	double *C = malloc(N * N * sizeof(double));

	
	memcpy(A_tA, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasTrans,
				CblasNonUnit,
				N,
				N,
				1.0,
				A,
				N,
				A_tA,
				N);

	
	
	memcpy(AB, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasNoTrans,
				CblasNonUnit,
				N,
				N,
				1.0,
				A,
				N,
				AB,
				N);

	
	memcpy(C, A_tA, N * N * sizeof(double));

	
	cblas_dgemm(CblasRowMajor,
				CblasNoTrans,
				CblasTrans,
				N,
				N,
				N,
				1.0,
				AB,
				N,
				B,
				N,
				1.0,
				C, N);
	
	
	free(A_tA);
	free(AB);

	return C;
}
