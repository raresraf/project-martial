/**
 * @raw/86718b5f-b7e4-4201-b5f5-fcdee2963f89/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Hardware-accelerated computation of C = A^T * A + (A * B) * B^T.
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ representing distinct output and intermediate buffers.
 */
#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
	
	register int i = 0;
	register int j = 0;

	/**
	 * Functional Utility: Creates a localized copy of matrix B to serve as the initial operand state for `cblas_dtrmm`.
	 */
	double *temp = (double *)malloc(N * N * sizeof(double));
	for(i = 0; i < N * N; i++)
                temp[i] = B[i];
	 
	/**
	 * Functional Utility: Multiplies upper triangular matrix A with B, updating `temp` in-place.
	 * Effectively computes temp = A * B.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft,
                CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, temp, N);
	
	/**
	 * Functional Utility: Allocates memory and computes res1 = temp * B^T = (A * B) * B^T.
	 * Evaluates general matrix multiplication utilizing transposed matrix B.
	 */
	double *res1 = (double *)malloc(N * N * sizeof(double));
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, temp, N, B, N, 0, res1, N);
	
	/**
	 * Functional Utility: Allocates memory and computes res2 = A^T * A.
	 * Leverages generalized matrix multiplication on matrix A and its transposition.
	 */
	double *res2 = (double *)malloc(N * N * sizeof(double));
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				N, N, N, 1, A, N, A, N, 0, res2, N);

	/**
	 * Block Logic: Consolidates the partitioned matrix multiplications into the final state.
	 * Invariant: Executes an element-wise addition, overwriting `res1`.
	 */
	for (i = 0; i < N; ++i) {
		register int in = i * N;
		for (j = 0; j < N; ++j) {
			res1[in + j] += res2[in + j];
		}
	}

	free(temp);
	free(res2);	
	return res1;
}
