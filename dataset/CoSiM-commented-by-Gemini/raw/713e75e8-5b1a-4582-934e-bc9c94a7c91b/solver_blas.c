
/**
 * @file solver_blas.c
 * @brief A matrix solver implementation using BLAS routines.
 * @note This version uses a different sequence of BLAS calls, including the Level-1 routine `daxpy`,
 *       to compute the same mathematical expression as other solver versions.
 */
#include "utils.h"
#include <cblas.h>
#include <string.h>


/**
 * @brief Performs a sequence of matrix operations using BLAS.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N), treated as upper triangular.
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix C. The caller is responsible for freeing this memory.
 *
 * @note The function computes the following expression: C = A * (B * B^T) + A^T * A
 *       This is achieved through a series of BLAS calls:
 *       1. `cblas_dtrmm` to compute A^T * A.
 *       2. `cblas_dgemm` to compute B * B^T.
 *       3. `cblas_dtrmm` to compute A * (B * B^T).
 *       4. `cblas_daxpy` to perform the final matrix addition.
 */
double* my_solver(int N, double *A, double *B) {
	
	// Allocate memory for the final result and an intermediate copy of A.
	double *C = calloc(sizeof(double), N * N);
	double *a_copy = calloc(sizeof(double), N * N);

	memcpy(a_copy, A, N * N * sizeof(double));

	
	// Step 1: Compute a_copy = A^T * A.
	// cblas_dtrmm is used here with CblasTrans to perform the operation on a_copy in place.
	// The operation is a_copy = 1.0 * a_copy_trans * a_copy.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, a_copy, N, a_copy, N);

	
	// Step 2: Compute C = B * B^T.
	// cblas_dgemm: General Matrix-Matrix Multiply.
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, B, N, B, N, 0.0, C, N);

	
	// Step 3: Compute C = A * C, which is now A * (B * B^T).
	// A is treated as an upper triangular matrix.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);

	
	// Step 4: Add the result from Step 1. C = C + a_copy.
	// cblas_daxpy is a Level-1 BLAS routine for vector addition (Y = a*X + Y).
	// Here, it's used to add the two matrices, treating them as single large vectors.
	cblas_daxpy(N * N, 1.0, a_copy, 1, C, 1);

	// Free the memory used for the intermediate copy.
	free(a_copy);
	
    return C;
}
