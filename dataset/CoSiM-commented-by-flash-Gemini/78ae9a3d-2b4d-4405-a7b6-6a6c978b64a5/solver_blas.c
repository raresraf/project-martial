/**
 * @file solver_blas.c
 * @brief Implements a matrix solver using BLAS (Basic Linear Algebra Subprograms) routines.
 *        This file likely computes (B^T * B) * A + A^T * A, assuming A and B are square matrices.
 * Algorithm: Utilizes optimized `cblas_dgemm` for general matrix-matrix multiplication and
 *            `cblas_dtrmm` for triangular matrix multiplication (though here used with `CblasNoTrans` for full matrix).
 * Time Complexity: O(N^3) due to matrix multiplications, where N is the dimension of the square matrices.
 * Space Complexity: O(N^2) for the auxiliary matrix `aux`.
 */

#include "utils.h"
#include "cblas.h"

/**
 * @brief Solves a matrix problem involving multiplication and addition using BLAS.
 *
 * This function calculates `(B^T * B) * A + A^T * A` (simplified representation)
 * where A and B are N x N matrices. It allocates an auxiliary matrix to store intermediate
 * and final results.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A Pointer to the N x N matrix A.
 * @param B Pointer to the N x N matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the result.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	// aux: Auxiliary matrix to store intermediate and final results, initialized to zeros.
	double *aux = (double*) calloc(N * N, sizeof(double));

	// Block Logic: Compute B^T * B and store the result in 'aux'.
	// cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 0, aux, N)
	// Performs: aux = 1 * B * B^T + 0 * aux => aux = B * B^T
	// Note: The problem description or typical usage might imply B^T * B.
	// Re-evaluating the actual parameters:
	// op(A) is B (no transpose), op(B) is B (transpose).
	// So it computes B * B^T, not B^T * B.
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 0, aux, N);

	// Block Logic: Compute (B^T * B) * A and store in 'aux'.
	// cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, aux, N)
	// Performs: aux = A * aux (where aux is currently B*B^T)
	// This would be A * (B * B^T).
	// Note: This operation effectively performs matrix multiplication, but its `dtrmm` name suggests triangular.
	// Given CblasNoTrans for A and CblasNonUnit for triangular part, it behaves like general matrix multiply.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, aux, N);

	// Block Logic: Compute A^T * A and add it to the existing content of 'aux'.
	// cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, N, 1, A, N, A, N, 1, aux, N)
	// Performs: aux = 1 * A^T * A + 1 * aux
	// So, the final result is A * (B * B^T) + A^T * A.
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, N, 1, A, N, A, N, 1, aux, N);
	return aux;
}
