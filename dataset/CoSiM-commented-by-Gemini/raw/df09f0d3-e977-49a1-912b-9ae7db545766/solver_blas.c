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
	printf("BLAS SOLVER\n");

	double *C1 = (double*) calloc(N * N, sizeof(double));
	double *C2 = (double*) calloc(N * N, sizeof(double));
	int i;
	
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(i = 0; i < N * N; i++)
		C1[i] = B[i];

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, 
				CblasNoTrans, CblasNonUnit, N, N, 1,
				A, N, C1, N);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, N, N, 1, C1, N, B, N, 0, C2, N);

	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(i = 0; i < N * N; i++)
		C1[i] = A[i];

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper,
				CblasTrans, CblasNonUnit, N, N, 1,
				A, N, C1, N);

	cblas_daxpy(N * N, 1, C2, 1, C1, 1);

	free(C2);

	return C1;
}
