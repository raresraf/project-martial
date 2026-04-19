/**
 * @raw/87f07f36-8bbb-4bd7-bcf8-d620c64b5568/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Delegates scalar multiplications to hardware-accelerated BLAS components evaluating C = A^T * A + (A * B) * B^T.
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ for auxiliary buffers `temp` and `C`.
 */
#include "utils.h"
#include <cblas.h>

double* my_solver(int N, double *A, double *B) {
	double *C = calloc(N * N, sizeof(double));
	double *temp = calloc(N * N, sizeof(double));
	int i, j;

	/**
	 * Block Logic: Prepares deep copies of the input matrices to be mutated in-place by BLAS routines.
	 * Invariant: Replicates elements of A and B prior to execution.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = B[i * N + j];
			temp[i * N + j] = A[i * N + j];
		}
	}

	/**
	 * Functional Utility: Left multiplication of an upper triangular matrix (A) onto a general matrix (C = B).
	 * C = A * B.
	 */
	cblas_dtrmm(101, 141, 121, 111, 131, N, N, 1.0, A, N, C, N);

	/**
	 * Functional Utility: Left multiplication of a transposed upper triangular matrix (A^T) onto a general matrix (temp = A).
	 * temp = A^T * A.
	 */
	cblas_dtrmm(101, 141, 121, 112, 131, N, N, 1.0, A, N, temp, N);

	/**
	 * Functional Utility: General matrix multiplication calculating temp = C * B^T + temp.
	 * Resolves temp = (A * B) * B^T + A^T * A.
	 */
	cblas_dgemm(101, 111, 112, N, N, N, 1.0, C, N, B, N, 1.0, temp, N);
	
	free(C);

	return temp;
}
