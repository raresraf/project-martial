/**
 * @raw/8fd042d2-01f7-415a-9a1e-2f2dba3f9f73/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Hardware-accelerated computation of C = A^T * A + (A * B) * B^T.
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ representing distinct output and intermediate buffers.
 */
#include "utils.h"

#include "cblas.h"
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	/**
	 * Functional Utility: Initializes dynamically allocated matrix buffers and caches.
	 */
	double *temp1 = (double *)calloc(N * N, sizeof(double));
	double *temp2 = (double *)calloc(N * N, sizeof(double));
	double *temp3 = (double *)calloc(N * N, sizeof(double));
	double *res = (double *)calloc(N * N, sizeof(double));
	int i, j;
	
	/**
	 * Block Logic: Clones initial matrix values directly to safeguard against overwriting variables.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			int index = i * N + j;
			temp1[index] = A[index];
			temp2[index] = B[index];
		}
	}
	
	/**
	 * Functional Utility: In-place calculation for `temp2 = A * B` assuming A is upper triangular.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
				N, N, 1, A, N, temp2, N);

	/**
	 * Functional Utility: In-place calculation for `temp1 = A^T * A` assuming A is upper triangular.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
				N, N, 1, A, N, temp1, N);
				
	/**
	 * Functional Utility: Executes calculation evaluating `temp3 = temp2 * B^T` (which is (A * B) * B^T).
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, temp2, N, B, N, 0, temp3, N);
	
	/**
	 * Block Logic: Evaluates final matrix summation executing res = temp3 + temp1.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			res[i * N + j] = temp3[i * N + j] + temp1[i * N + j];
		}
	}
	free(temp1);
	free(temp2);
	free(temp3);
	return res;
}
