/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include "utils.h"
#include "cblas.h"
#include <string.h>


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {
	double *Acopy = malloc(N * N * sizeof(double));
	double *Bcopy = malloc(N * N * sizeof(double));

	memcpy(Acopy, A, N * N * sizeof(double));
	memcpy(Bcopy, B, N * N * sizeof(double));

	
    	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
                CblasNonUnit, N, N, 1, A, N, Bcopy, N);

	
    	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
                CblasNonUnit, N, N, 1, A, N, Acopy, N);

	
    	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, N, N, 1, Bcopy, N, B, N, 1, Acopy, N);

	free(Bcopy);

	return Acopy;
}