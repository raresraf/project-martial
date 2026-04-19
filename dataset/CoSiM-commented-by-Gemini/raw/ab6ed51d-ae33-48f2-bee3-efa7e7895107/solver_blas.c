/**
 * @raw/ab6ed51d-ae33-48f2-bee3-efa7e7895107/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Hardware-accelerated computation of C = (A * B) * B^T + A^T * A.
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ representing distinct output and intermediate buffers.
 */

#include "utils.h"
#include <cblas.h>
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	/**
	 * Functional Utility: Initializes dynamically allocated matrix buffers tracking memory addresses appropriately.
	 */
	double *C = malloc(sizeof(double) * N * N);
	if (!C) 
		return NULL;

	double *AB = malloc(sizeof(double) * N * N);
	if (!AB) 
		return NULL;

	/**
	 * Functional Utility: Duplicates matrix B inside array parameter AB to generate the formulation components utilizing standard Level 3 operations.
	 */
	memmove(AB, B, N * N * sizeof(double));

	/**
	 * Functional Utility: Calculates the fundamental step determining AB = A * B leveraging native dtrmm mappings targeting upper triangular arrays.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, AB, N);

	/**
	 * Functional Utility: Isolates memory generating A to define C subsequently executing an in-place conversion transforming C = A^T * A.
	 */
	memmove(C, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);

	/**
	 * Functional Utility: Executes computation performing overall accumulation defining combination mapping directly into C.
	 * Executes C = C + AB * B^T where AB = A * B currently.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, C, N);
	
	free(AB);
	return C;
}
