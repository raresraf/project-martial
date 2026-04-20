/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include "utils.h"
#include "cblas.h"
#include "string.h"


/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {
	double *AxB = malloc(N * N * sizeof(double));
	double *A_copy = malloc(N * N * sizeof(double));
	double *C = malloc(N * N * sizeof(double));
	int i, j;

	
	memcpy(AxB, B, N * N * sizeof(double));
	memcpy(A_copy, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
			 CblasNonUnit, N, N, 1, A, N, AxB, N);
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
			 CblasNonUnit, N, N, 1, A, N, A_copy, N);
	cblas_dgemm(CblasRowMajor, CblasNoTrans,
			 CblasTrans, N, N, N, 1, AxB, N, B, N, 0, C, N);
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for (i = 0; i < N; i++)
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for (j = 0; j < N; j++)
			*(A_copy + i * N + j) += *(C + i * N + j); 
	
	free(AxB);
	free(C);
	return A_copy;

}
