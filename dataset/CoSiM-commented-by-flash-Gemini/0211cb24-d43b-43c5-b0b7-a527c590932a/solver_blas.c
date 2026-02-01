/**
 * @file solver_blas.c
 * @brief Implements a matrix solver using BLAS (Basic Linear Algebra Subprograms) routines.
 * This file provides a high-performance solution for a specific matrix computation task,
 * leveraging optimized CBLAS functions for operations on double-precision floating-point matrices.
 * It allocates intermediate matrices and performs a sequence of copy, triangular matrix multiplication,
 * and general matrix multiplication operations.
 * Algorithm: The solver performs the following sequence of operations:
 *   1. Initializes `res1` with a copy of matrix `B`.
 *   2. Computes `res1 = A * res1` (triangular matrix multiplication with `A` being upper triangular, no transpose).
 *   3. Initializes `res2` with a copy of matrix `A`.
 *   4. Computes `res2 = A^T * res2` (triangular matrix multiplication with `A` being upper triangular, transpose).
 *   5. Computes `res2 = 1 * res1 * B^T + 1 * res2` (general matrix multiplication, accumulating results).
 * Optimization: Utilizes highly optimized, hardware-accelerated BLAS/CBLAS libraries, ensuring
 * peak performance for dense linear algebra operations.
 * Time Complexity: Dominated by matrix multiplications, typically $O(N^3)$ for $N \times N$ matrices,
 * but with significant constant factor improvements due to BLAS optimizations.
 * Space Complexity: $O(N^2)$ for storing the input and intermediate result matrices.
 */

#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	// Block Logic: Allocate memory for the first intermediate result matrix 'res1'.
	// Precondition: 'N * N' represents the total number of double-precision elements required.
	// Invariant: 'res1' is initialized with zeros upon successful allocation.
	double *res1 = calloc(N * N, sizeof(double));
	// Conditional Logic: Handle memory allocation failure.
	if (res1 == NULL)
		exit(EXIT_FAILURE);

	// Block Logic: Allocate memory for the second intermediate result matrix 'res2'.
	// Precondition: 'N * N' represents the total number of double-precision elements required.
	// Invariant: 'res2' is initialized with zeros upon successful allocation.
	double *res2 = calloc(N * N, sizeof(double));
	// Conditional Logic: Handle memory allocation failure.
	if (res2 == NULL)
		exit(EXIT_FAILURE);

	// Functional Utility: Copy matrix B into res1.
	// This serves as the initial state for subsequent operations on res1.
	cblas_dcopy(N * N, B, 1, res1, 1);

	// Functional Utility: Perform triangular matrix-matrix multiplication: res1 = A * res1.
	// Interprets A as an upper triangular matrix, no transpose, with non-unit diagonal.
	// This operation effectively applies a transformation to the 'res1' matrix.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper,
		CblasNoTrans, CblasNonUnit, N, N, 1, A, N, res1, N);

	// Functional Utility: Copy matrix A into res2.
	// This prepares res2 for a subsequent triangular matrix multiplication involving A.
	cblas_dcopy(N * N, A, 1, res2, 1);

	// Functional Utility: Perform triangular matrix-matrix multiplication: res2 = A^T * res2.
	// Interprets A as an upper triangular matrix, transposed, with non-unit diagonal.
	// This operation applies another transformation to the 'res2' matrix.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper,
		CblasTrans, CblasNonUnit, N, N, 1, A, N, res2, N);

	// Functional Utility: Perform general matrix-matrix multiplication with accumulation: res2 = 1 * res1 * B^T + 1 * res2.
	// This is the final and most computationally intensive step, combining previously computed intermediate results.
	// Note that B is used as the matrix 'B' in the GEMM operation, not 'res2'.
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, res1, N, B, N, 1, res2, N);

	// Block Logic: Free the memory allocated for the intermediate result matrix 'res1'.
	// Invariant: 'res1' is no longer needed after all operations are complete.
	free(res1);
	// Functional Utility: Return the final result matrix 'res2'.
	return res2;
}
