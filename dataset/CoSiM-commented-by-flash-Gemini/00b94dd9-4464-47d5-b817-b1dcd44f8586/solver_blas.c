/**
 * @00b94dd9-4464-47d5-b817-b1dcd44f8586/solver_blas.c
 * @brief Implements an optimized matrix solver (`my_solver`) leveraging BLAS (Basic Linear Algebra Subprograms)
 *        routines, specifically the CBLAS interface for double-precision floating-point operations.
 *
 * This module provides a high-performance solution for a complex matrix computation problem.
 * It utilizes highly optimized external libraries to perform matrix multiplications and
 * other linear algebra operations, which are critical for scientific computing and
 * numerical analysis applications. The solver includes a helper function for matrix transposition.
 *
 * Algorithm: The solver orchestrates a sequence of Level 3 BLAS routines (`cblas_dtrmm`, `cblas_dgemm`)
 *            for matrix multiplications. This is complemented by custom implementations for
 *            matrix transposition and element-wise addition, strategically designed for efficiency.
 * Time Complexity: The dominant operations are matrix-matrix multiplications,
 *                  resulting in an overall time complexity of O(N^3), where N is the
 *                  dimension of the square matrices.
 * Space Complexity: O(N^2) for storing intermediate matrices and transposes,
 *                   proportional to the size of the input matrices.
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
	/**
	 * Functional Utility: Dynamically allocates memory for the transposed matrix.
	 * Allocation: Uses `calloc` to allocate `N * N` elements of type `double` and initializes all bytes to zero.
	 * Error Handling: In a production system, a check for `tr == NULL` should be present to handle allocation failures.
	 */
	double *tr = calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Iterates through each element of the original matrix `M` using nested loops.
	 *              For each element `M[j][i]` (logical representation), its value is placed
	 *              into the corresponding transposed position `tr[i][j]`.
	 * Invariant: `tr[i*N + j]` receives the value from `M[j*N + i]`, effectively
	 *            swapping rows and columns and storing in row-major order.
	 */
	for (int i = 0; i < N; ++i) { // Iterates over rows of the transposed matrix.
		for (int j = 0; j < N; ++j) { // Iterates over columns of the transposed matrix.
			// Inline: Assigns the element from `M` at logical position `(j, i)` to `tr` at logical position `(i, j)`.
			//         This implements the matrix transposition.
			tr[i * N + j] = M[j * N + i];
		}
	}
	return tr; // Functional Utility: Returns a pointer to the newly created transposed matrix.
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
	printf("BLAS SOLVER\n"); // Functional Utility: Prints an identifier to standard output, indicating that the BLAS-optimized solver is being utilized.

	/**
	 * Functional Utility: Dynamically allocates memory for intermediate and final result matrices.
	 * Allocation Strategy: Uses `calloc` to ensure memory is zero-initialized, which is often a safe
	 *                      default for numerical computations, and explicitly handles `N*N` double-precision elements.
	 * Error Handling: In a production-grade system, each `calloc` call would be followed by a null-pointer check
	 *                 to ensure successful memory allocation before proceeding.
	 */
	double *first_mul = calloc(N * N, sizeof(double));     /**< Stores the result of the first triangular matrix-matrix multiplication (A * U). */
	double *first_mul_aux = calloc(N * N, sizeof(double)); /**< Stores the accumulating result of subsequent matrix operations, eventually holding the final solution. */
	
	/**
	 * Block Logic: Computes the transposes of the input matrices `A` and `B`.
	 * Functional Utility: These transposed matrices (`At` and `Bt`) are necessary for subsequent
	 *                     BLAS operations that require operands in a specific orientation
	 *                     or for implementing specific parts of the matrix expression.
	 * Resource Management: The memory for `At` and `Bt` is dynamically allocated within `get_transpose`
	 *                      and must be freed by the caller of `my_solver`.
	 */
	double *At = get_transpose(A, N); /**< Stores the transpose of the input matrix `A`. */
	double *Bt = get_transpose(B, N); /**< Stores the transpose of the input matrix `B`. */
	
	/**
	 * Functional Utility: Copies the content of matrix `A` into `first_mul`.
	 * Rationale: This deep copy is crucial because `first_mul` will be used as both an
	 *            input and an in-place output buffer for `cblas_dtrmm`. Preserving the
	 *            original `A` allows it to be used later in the computation.
	 * Parameters:
	 *   `first_mul`: Destination memory block.
	 *   `A`: Source memory block.
	 *   `N * N * sizeof(double)`: Number of bytes to copy (size of an N x N matrix of doubles).
	 */
	memcpy(first_mul, A, N * N * sizeof(double));

	/**
	 * Block Logic: Executes the first crucial matrix-matrix multiplication using `cblas_dtrmm`.
	 *              This routine performs a triangular matrix multiplication of the form:
	 *              `first_mul = alpha * B * first_mul` (where `first_mul` initially held `A`).
	 * Functional Utility: Computes an intermediate product, where `B` acts on `A` (specifically its
	 *                     upper triangular part due to `CblasUpper`), storing the result back into `first_mul`.
	 * CBLAS Parameters Breakdown:
	 *   `CblasRowMajor`: Specifies that matrices are stored in row-major order.
	 *   `CblasLeft`: Indicates that the triangular matrix `B` is on the left side in the multiplication `B * A`.
	 *   `CblasUpper`: Only the upper triangular part of `B` is used in the operation.
	 *   `CblasNoTrans`: `B` is used as is, not transposed.
	 *   `CblasNonUnit`: Diagonal elements of `B` are not assumed to be 1 and are explicitly used.
	 *   `N, N`: Dimensions of the matrices, both N x N.
	 *   `1.0`: The scalar `alpha` value.
	 *   `B`: Pointer to matrix `B`.
	 *   `N`: Leading dimension of `B`.
	 *   `first_mul`: Pointer to matrix `A` (input), which is overwritten with the result (output).
	 *   `N`: Leading dimension of `first_mul`.
	 */
	cblas_dtrmm( CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N, 1.0, B, N, first_mul, N);

	/**
	 * Block Logic: Executes a general matrix-matrix multiplication using `cblas_dgemm`.
	 *              This routine performs the operation:
	 *              `C = alpha * A * B + beta * C`
	 *              In this specific context: `first_mul_aux = 1.0 * first_mul * Bt + 0.0 * first_mul_aux`.
	 * Functional Utility: Computes the product of the intermediate `first_mul` (result of the triangular multiplication)
	 *                     and the transpose of `B` (`Bt`), storing this product in `first_mul_aux`.
	 * CBLAS Parameters Breakdown:
	 *   `CblasRowMajor`: Specifies that matrices are stored in row-major order.
	 *   `CblasNoTrans`: `first_mul` (acting as matrix A in `dgemm`'s terms) is used as is, not transposed.
	 *   `CblasNoTrans`: `Bt` (acting as matrix B in `dgemm`'s terms) is used as is, not transposed (it's already `B` transposed).
	 *   `N, N, N`: Dimensions M, N, K for an (M x K) * (K x N) multiplication, all equal to N.
	 *   `1.0`: The scalar `alpha` value.
	 *   `first_mul`: Pointer to the first input matrix for `dgemm`.
	 *   `N`: Leading dimension of `first_mul`.
	 *   `Bt`: Pointer to the second input matrix for `dgemm`.
	 *   `N`: Leading dimension of `Bt`.
	 *   `0.0`: The scalar `beta` value.
	 *   `first_mul_aux`: Pointer to the output matrix, which stores the result of the multiplication.
	 *   `N`: Leading dimension of `first_mul_aux`.
	 */
	 cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
	 	N, N, N, 1.0, first_mul, N, Bt, N, 0.0, first_mul_aux, N);

	/**
	 * Block Logic: Executes a second triangular matrix-matrix multiplication using `cblas_dtrmm`.
	 *              This operation is of the form: `At = alpha * A * At` (where `At` originally held `A`'s transpose).
	 * Functional Utility: This computes an intermediate product required for the final element-wise addition.
	 *                     It multiplies matrix `A` with its own transpose (`At`), leveraging the upper
	 *                     triangular structure (implicitly from `CblasUpper`) and stores the result back into `At`.
	 * CBLAS Parameters Breakdown:
	 *   `CblasRowMajor`, `CblasLeft`, `CblasUpper`, `CblasNoTrans`, `CblasNonUnit`:
	 *     Parameters configured similarly to the first `cblas_dtrmm` call, defining row-major storage,
	 *     left-multiplication by a triangular matrix (effectively `A`), usage of the upper triangular
	 *     part of the implicit triangular matrix, no transposition, and non-unit diagonals.
	 *   `N, N`: Dimensions of the matrices.
	 *   `1.0`: The scalar `alpha` value.
	 *   `A`: Pointer to matrix `A` (input).
	 *   `N`: Leading dimension of `A`.
	 *   `At`: Pointer to matrix `At` (input/output). `At` is modified in place to store the result.
	 *   `N`: Leading dimension of `At`.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N, 1.0, A, N, At, N);

	/**
	 * Block Logic: Iterates through each element of the `first_mul_aux` and `At` matrices
	 *              to perform an element-wise addition.
	 * Functional Utility: This step completes the matrix expression by adding the final
	 *                     transformed `At` matrix to the product `(A * U) * Bt`.
	 * Pre-condition: `first_mul_aux` contains the result of the previous `cblas_dgemm` operation,
	 *                and `At` contains the result of the previous `cblas_dtrmm` operation.
	 * Invariant: After the loops, `first_mul_aux` holds the final computed matrix
	 *            (`(A * U) * Bt + A_transformed`).
	 */
	for (int i = 0; i < N; i++) { // Iterates over rows of the matrices.
		for (int j = 0; j < N; j++) { // Iterates over columns of the matrices.
			// Inline: Adds the element `At[i][j]` to `first_mul_aux[i][j]`, updating `first_mul_aux` in place.
			first_mul_aux[i * N + j] += At[i * N + j];
		}
	}

	/**
	 * Block Logic: Releases all dynamically allocated memory for intermediate matrices.
	 * Functional Utility: Crucial for preventing memory leaks and ensuring efficient resource
	 *                     management, especially in long-running applications or repeated calls.
	 * Pre-condition: `first_mul`, `At`, and `Bt` point to valid, previously allocated memory.
	 * Post-condition: The memory blocks pointed to by `first_mul`, `At`, and `Bt` are deallocated.
	 */
	free(first_mul); // Resource Management: Deallocates memory used for the `first_mul` matrix.
	free(At);       // Resource Management: Deallocates memory used for the transposed `A` matrix.
	free(Bt);       // Resource Management: Deallocates memory used for the transposed `B` matrix.
	return first_mul_aux; // Functional Utility: Returns the pointer to the final computed matrix, which the caller is responsible for freeing.
}
