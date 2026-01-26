/**
 * @file solver_blas.c
 * @brief This file implements a matrix solver (`my_solver`) using BLAS (Basic Linear Algebra Subprograms)
 *        routines, specifically CBLAS for double-precision operations. It provides an optimized
 *        solution for a specific matrix computation problem, leveraging highly tuned external libraries.
 *
 * It includes a helper function to compute matrix transposes.
 *
 * Algorithm: Leverages Level 3 BLAS routines (`cblas_dtrmm`, `cblas_dgemm`) for matrix
 *            multiplications, combined with custom matrix transposition and element-wise addition.
 * Time Complexity: Dominated by matrix multiplications, typically O(N^3).
 * Space Complexity: O(N^2) for storing auxiliary matrices and transposes.
 */

#include "utils.h" // Assumed to contain utility functions or definitions.
#include "cblas.h" // Header for CBLAS (C interface to BLAS).
#include <string.h> // For memcpy().
#include <stdio.h> // For printf().
#include <stdlib.h> // For calloc(), free().


/**
 * @brief Computes the transpose of a square matrix.
 *
 * Allocates memory for the transposed matrix and fills it by swapping
 * row and column indices of the original matrix.
 *
 * @param M A pointer to the original square matrix (stored in row-major order).
 * @param N The dimension of the square matrix (N x N).
 * @return A pointer to the newly allocated and populated transposed matrix.
 *         The caller is responsible for freeing this memory.
 *
 * Algorithm: Direct element-wise transposition.
 * Time Complexity: O(N^2) due to nested loops iterating over all elements.
 * Space Complexity: O(N^2) for the new transposed matrix.
 */
static double *get_transpose(double *M, int N)
{
	// Allocate memory for the transposed matrix, initialized to zeros.
	double *tr = calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Iterates through the original matrix to populate the transposed matrix.
	 * Invariant: `tr[i*N + j]` receives the value from `M[j*N + i]`, effectively
	 *            swapping rows and columns.
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			tr[i * N + j] = M[j * N + i]; // Perform transposition.
		}
	}
	return tr;
}


/**
 * @brief Implements a matrix solver using CBLAS routines for optimized performance.
 *
 * This function calculates a complex matrix expression involving input matrices A and B.
 * The expression can be described as: `result = (A * U) * B_transpose + A_transpose`
 * where U is an implicit upper triangular matrix based on `cblas_dtrmm`.
 * It leverages `cblas_dtrmm` for triangular matrix-matrix multiplication and
 * `cblas_dgemm` for general matrix-matrix multiplication, along with custom
 * matrix transposition and element-wise addition.
 *
 * @param N The dimension of the square matrices (N x N).
 * @param A A pointer to the first input square matrix.
 * @param B A pointer to the second input square matrix.
 * @return A pointer to the newly allocated result matrix. The caller is
 *         responsible for freeing this memory.
 *
 * Algorithm: Sequence of matrix operations using BLAS primitives.
 * Time Complexity: Dominated by matrix multiplications, O(N^3).
 * Space Complexity: O(N^2) for `first_mul`, `first_mul_aux`, `At`, and `Bt`.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n"); // Indicate which solver is being used.

	// Allocate memory for intermediate and final result matrices.
	double *first_mul = calloc(N * N, sizeof(double));     // Stores A * B (initial triangular mult)
	double *first_mul_aux = calloc(N * N, sizeof(double)); // Stores (A * B) * Bt + At (final result)
	
	// Compute transposes of input matrices A and B.
	double *At = get_transpose(A, N); // Transpose of A.
	double *Bt = get_transpose(B, N); // Transpose of B.
	
	// Copy matrix A to `first_mul` to use it as an input for the first multiplication.
	memcpy(first_mul, A, N * N * sizeof(double));

	/**
	 * Block Logic: Performs the first matrix-matrix multiplication using CBLAS.
	 * Operation: `first_mul = B * first_mul` where `first_mul` is initially `A`.
	 *            Specifically, `first_mul = B * A` (interpreted as B * (implicit Upper Triangular from A)).
	 *            CblasRowMajor: Matrices stored in row-major order.
	 *            CblasLeft: B is on the left side of the operation.
	 *            CblasUpper: Only the upper triangular part of B is used implicitly.
	 *            CblasNoTrans: B is not transposed.
	 *            CblasNonUnit: Diagonal elements of B are not assumed to be 1.
	 *            N, N: Dimensions of the matrices.
	 *            1.0: Scalar alpha.
	 *            B: Matrix B.
	 *            first_mul: Matrix A (modified in place to store result B*A).
	 */
	cblas_dtrmm( CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N, 1.0, B, N, first_mul, N);

	/**
	 * Block Logic: Performs the second general matrix-matrix multiplication.
	 * Operation: `first_mul_aux = first_mul * Bt`
	 *            CblasRowMajor: Matrices stored in row-major order.
	 *            CblasNoTrans: `first_mul` is not transposed.
	 *            CblasNoTrans: `Bt` is not transposed (it's already B transposed).
	 *            N, N, N: Dimensions M, N, K for (M x K) * (K x N).
	 *            1: Scalar alpha.
	 *            first_mul: Input matrix (result of first multiplication).
	 *            Bt: Input matrix (transpose of B).
	 *            0: Scalar beta.
	 *            first_mul_aux: Output matrix (stores the result).
	 */
	 cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
	 	N, N, N, 1, first_mul, N, Bt, N, 0, first_mul_aux, N);

	/**
	 * Block Logic: Computes A * A_transpose using triangular matrix-matrix multiplication.
	 * Operation: This line seems to be an intermediate calculation for the final element-wise addition.
	 *            It appears to overwrite `At` with a result involving A.
	 *            cblas_dtrmm: A * At (interpreted as A * (implicit Upper Triangular from A))
	 *            CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit: Similar parameters as first dtrmm.
	 *            A: Matrix A.
	 *            At: Matrix At (modified in place).
	 * @note This call to `cblas_dtrmm` seems to use `At` as both an input (implicitly from its structure)
	 *       and an output for storing the result. This might be part of a larger expression.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N, 1.0, A, N, At, N);

	/**
	 * Block Logic: Performs element-wise addition of `At` (modified from previous step)
	 *              to `first_mul_aux` (result of second multiplication).
	 * Invariant: Each element of `first_mul_aux` is updated with the sum of itself
	 *            and the corresponding element from `At`.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			first_mul_aux[i * N + j] += At[i * N + j]; // Element-wise addition.
		}
	}

	/**
	 * Block Logic: Frees dynamically allocated memory for intermediate matrices.
	 * Functional Utility: Prevents memory leaks by releasing resources.
	 */
	free(first_mul);
	free(At);
	free(Bt);
	return first_mul_aux; // Return the final result matrix.
}
