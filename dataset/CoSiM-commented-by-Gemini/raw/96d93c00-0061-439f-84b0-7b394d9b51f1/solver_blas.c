/**
 * @raw/96d93c00-0061-439f-84b0-7b394d9b51f1/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Evaluates RES = (A * B) * B^T + A^T * A.
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ tracking internal partial product evaluations.
 */

#include "utils.h"
#include "cblas.h"
#include <string.h>

double* my_solver(int N, double *A, double *B) {
	double *A_star_B;
	double *OP1;
	double *OP2;
	double *RES;
	int numMatrixElems = N * N;

	/**
	 * Functional Utility: Initializes necessary dynamically allocated matrix partitions.
	 */
	A_star_B = calloc(numMatrixElems, sizeof(*A_star_B));
	OP1 = calloc(numMatrixElems, sizeof(*OP1));
	OP2 = calloc(numMatrixElems, sizeof(*OP2));
	RES = calloc(numMatrixElems, sizeof(*RES));

	/**
	 * Functional Utility: Clones initial matrix values to isolate transformations safely without interfering with base state.
	 */
	memcpy(A_star_B, B, numMatrixElems * sizeof(*B));

	/**
	 * Functional Utility: In-place calculation evaluating A_star_B = A * B evaluating upper triangular parameters.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, A_star_B, N);

	/**
	 * Functional Utility: In-place calculation forming A = A^T * A utilizing its initial upper triangular geometry.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, A, N);

	/**
	 * Functional Utility: Combines the previously constructed entities into A = (A * B) * B^T + A. 
	 * Consequently computes (A * B) * B^T + A^T * A internally merging the values correctly into `A`.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, A_star_B, N, B, N, 1.0, A, N);

	/**
	 * Functional Utility: Maps evaluated sum back into the result pointer framework preparing the output properly.
	 */
	memcpy(RES, A, numMatrixElems * sizeof(*A));
	free(A_star_B);
	free(OP1);
	free(OP2);
	return RES;
}
