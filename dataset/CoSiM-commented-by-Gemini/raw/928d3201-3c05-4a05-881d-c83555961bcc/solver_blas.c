/**
 * @raw/928d3201-3c05-4a05-881d-c83555961bcc/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Hardware-accelerated computation of C = A * (B * B^T) + A^T * A utilizing contiguous cblas functions.
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
	double *AB = (double*) calloc(N * N, sizeof(double));

	/**
	 * Functional Utility: Duplicates base elements to serve structurally.
	 */
	cblas_dcopy(N * N, B, 1, AB, 1);

	/**
	 * Functional Utility: Constructs AB = A * B evaluating upper triangular parameters.
	 */
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

	/**
	 * Functional Utility: Prepares final accumulator vector output.
	 */
	double *result = (double*) calloc(N * N, sizeof(double));

	/**
	 * Functional Utility: Allocates base properties.
	 */
	cblas_dcopy(N * N, A, 1, result, 1);

	/**
	 * Functional Utility: Evaluates A^T * A assuming upper-triangular origin parameters.
	 */
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
		result,
		N);
		
	/**
	 * Functional Utility: Summarizes total sequence mapping equivalent combinations representing (A * B) * B^T + A^T * A.
	 */
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
		result,
		N);

	free(AB);
	return result;
}
