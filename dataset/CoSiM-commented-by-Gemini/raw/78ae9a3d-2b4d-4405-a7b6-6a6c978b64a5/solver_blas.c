#include "utils.h"
#include "cblas.h"

/**
 * @file solver_blas.c
 * @brief BLAS-based matrix equation solver.
 * @details This function implements a sequence of matrix operations using the CBLAS interface.
 * The operations appear to be part of a larger matrix equation solver, likely involving
 * multiplication and addition of matrices A, B, and their transposes.
 */

/**
 * @brief Solves a matrix equation using a series of BLAS operations.
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the matrix A (N x N).
 * @param B A pointer to the matrix B (N x N).
 * @return A pointer to the resulting matrix. The caller is responsible for freeing this memory.
 *
 * @note The sequence of operations is:
 * 1. aux = B * B^T (General Matrix-Matrix multiplication)
 * 2. aux = A * aux (Triangular Matrix-Matrix multiplication, assuming A is upper triangular)
 * 3. aux = A^T * A + aux (General Matrix-Matrix multiplication and addition)
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	// Allocate memory for the auxiliary matrix to store intermediate and final results.
	double *aux = (double*) calloc(N * N, sizeof(double));

	/**
	 * @brief Operation 1: aux = B * B^T
	 * @details cblas_dgemm performs a general matrix-matrix multiplication.
	 * Here, it computes B multiplied by its transpose.
	 * CblasRowMajor: Data is stored in row-major order.
	 * CblasNoTrans: Matrix B is used as is.
	 * CblasTrans: The transpose of the second matrix B is used.
	 * N, N, N: Dimensions of the matrices (M, N, K).
	 * 1: Alpha scalar multiplier for the product.
	 * B, N: Pointer to matrix B and its leading dimension.
	 * B, N: Pointer to the second matrix (also B) and its leading dimension.
	 * 0: Beta scalar multiplier for the initial matrix aux (effectively setting aux = 0 before adding the product).
	 * aux, N: Pointer to the output matrix and its leading dimension.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 0, aux, N);

	/**
	 * @brief Operation 2: aux = A * aux
	 * @details cblas_dtrmm performs a triangular matrix-matrix multiplication.
	 * It multiplies the matrix `aux` by the triangular matrix `A`.
	 * CblasRowMajor: Data is stored in row-major order.
	 * CblasLeft: The triangular matrix A is on the left side of the multiplication.
	 * CblasUpper: Matrix A is an upper triangular matrix.
	 * CblasNoTrans: Matrix A is not transposed.
	 * CblasNonUnit: The diagonal elements of A are not assumed to be 1.
	 * N, N: Dimensions of the operation.
	 * 1: Alpha scalar multiplier.
	 * A, N: Pointer to matrix A and its leading dimension.
	 * aux, N: Pointer to the input/output matrix `aux` and its leading dimension.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, aux, N);

	/**
	 * @brief Operation 3: aux = A^T * A + aux
	 * @details cblas_dgemm is used again to compute A^T * A and add the result to `aux`.
	 * CblasTrans: Transpose of the first matrix A is used.
	 * CblasNoTrans: The second matrix A is used as is.
	 * 1: Alpha scalar for the product A^T * A.
	 * 1: Beta scalar for the matrix `aux`, so the final result is (A^T * A) + aux.
	 */
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, N, 1, A, N, A, N, 1, aux, N);

	// Return the final result matrix.
	return aux;
}