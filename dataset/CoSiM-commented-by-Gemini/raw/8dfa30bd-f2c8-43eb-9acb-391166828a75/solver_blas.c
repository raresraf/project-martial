/**
 * @raw/8dfa30bd-f2c8-43eb-9acb-391166828a75/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Hardware-accelerated computation of C = (A * B) * B^T + A^T * A utilizing dtrmm and dgemm combinations.
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ representing distinct output and intermediate buffers.
 */

#include "utils.h"

#include <cblas.h>

double* my_solver(int N, double *A, double *B) {
	double *AB, *AtA, *C, *B2;

	/**
	 * Functional Utility: Initializes dynamically allocated matrix buffers and caches for intermediate outputs.
	 */
	AB = calloc(N * N, sizeof(double));
	if (AB == NULL)
		exit(-1);
	AtA = calloc(N * N, sizeof(double));
	if (AtA == NULL)
		exit(-1);
	C = calloc(N * N, sizeof(double));
	if (C == NULL)
		exit(-1);
	B2 = calloc(N * N, sizeof(double));
	if (B2 == NULL)
		exit(-1);

	/**
	 * Functional Utility: Clones matrix B to reserve its unaltered state for `B^T` operations.
	 */
	cblas_dcopy(N * N, B, 1, B2, 1);
	
	/**
	 * Functional Utility: Performs an in-place matrix-matrix multiplication generating B = A * B.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, B, N);

	/**
	 * Functional Utility: Clones matrix A into AtA enabling its subsequent transformation.
	 */
	cblas_dcopy(N * N, A, 1, AtA, 1);
	
	/**
	 * Functional Utility: Modifies AtA sequentially computing AtA = A^T * A utilizing triangular nature.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, AtA, N);

	/**
	 * Functional Utility: Migrates AtA inside the target output matrix C.
	 */
	cblas_dcopy(N * N, AtA, 1, C, 1);
	
	/**
	 * Functional Utility: Appends to C the previously computed B instances (now containing A * B),
	 * calculating C += B * B2^T (representing A * B * B^T).
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B2, N, 1, C, N);

    free(AB);
    free(AtA);
	free(B2);

	return C;
}
