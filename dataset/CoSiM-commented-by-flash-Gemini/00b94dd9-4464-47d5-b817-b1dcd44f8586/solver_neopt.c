/**
 * @00b94dd9-4464-47d5-b817-b1dcd44f8586/solver_neopt.c
 * @brief Implements a "non-optimized" matrix solver (`my_solver`) using explicit nested loops
 *        for all matrix operations. This approach serves as a baseline for comparison
 *        against optimized BLAS-based solutions (`solver_blas.c`), or for environments
 *        where high-performance linear algebra libraries are unavailable.
 *
 * This file provides a direct, pedagogical implementation of matrix arithmetic,
 * including multiplication and transposition, primarily for demonstration or
 * verification of correctness against optimized counterparts.
 *
 * Algorithm: Direct, element-wise implementation of matrix operations (multiplication,
 *            addition, transposition) using triple-nested loops for matrix products
 *            and double-nested loops for element-wise operations. Notably, partial
 *            triangular matrix multiplications are performed with specific loop bounds.
 * Time Complexity: Dominated by the triple-nested loops for matrix multiplications,
 *                  resulting in a time complexity of O(N^3) for each matrix product,
 *                  where N is the dimension of the square matrices.
 * Space Complexity: O(N^2) for storing auxiliary matrices (`second_mul`, `first_mul`,
 *                   `first_mul_helper`, `At`, `Bt`) that are created during the computation.
 */

#include "utils.h" // Assumed to contain utility functions or definitions.
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
 * @brief Implements a matrix solver using direct nested loops for matrix operations.
 *
 * This function calculates a complex matrix expression involving input matrices A and B.
 * It computes intermediate matrix products and sums using explicit triple-nested loops
 * for multiplications and double-nested loops for element-wise addition.
 *
 * @param N The dimension of the square matrices (N x N).
 * @param A A pointer to the first input square matrix.
 * @param B A pointer to the second input square matrix.
 * @return A pointer to the newly allocated result matrix. The caller is
 *         responsible for freeing this memory.
 *
 * Algorithm: Direct implementation of matrix operations.
 * Time Complexity: Dominated by matrix multiplications, O(N^3).
 * Space Complexity: O(N^2) for `second_mul`, `first_mul`, `first_mul_helper`, `At`, and `Bt`.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n"); // Functional Utility: Prints an identifier to standard output, indicating that the non-optimized solver is being utilized.

	/**
	 * Functional Utility: Dynamically allocates memory for intermediate matrices required for computation.
	 * Allocation Strategy: Uses `calloc` to allocate `N * N` elements of type `double` and initializes all bytes to zero,
	 *                      providing a clean slate for accumulation in matrix products.
	 * Error Handling: In a robust production environment, each `calloc` call should be immediately followed
	 *                 by a null-pointer check to ensure memory was successfully allocated.
	 */
	double *second_mul = calloc(N * N, sizeof(double));     /**< Stores the result of the partial matrix multiplication `At * A`. */
	double *At = get_transpose(A, N);                       /**< Stores the transpose of the input matrix `A`, obtained via `get_transpose`. */
		
	/**
	 * Block Logic: Computes a partial matrix multiplication `second_mul = At * A`.
	 *              The inner loop bounds (`k = 0; k <= i`) indicate that this operation
	 *              is effectively accumulating only the elements corresponding to a
	 *              lower triangular multiplication with respect to `At`'s rows.
	 * Algorithm: Triple-nested loop for matrix multiplication.
	 * Pre-condition: `At` contains the transpose of `A`, and `A` is the original matrix.
	 * Invariant: `second_mul[i*N + j]` accumulates the dot product of the i-th row of `At`
	 *            and the j-th column of `A`, constrained by the triangular access pattern.
	 */
	for (int i = 0; i < N; ++i) { // Iterates over rows of the resulting `second_mul` matrix.
		for (int j = 0; j < N; ++j) { // Iterates over columns of the `second_mul` matrix.
			for (int k = 0; k <= i; ++k) { // Control Flow: Loop bound (`k <= i`) indicates a triangular matrix multiplication pattern.
				// Inline: Accumulates the product of `At[i][k]` and `A[k][j]` into `second_mul[i][j]`.
				second_mul[i * N + j] += At[i * N + k] * A[k * N + j];
			}
		}
	}

	/**
	 * Functional Utility: Dynamically allocates memory for additional intermediate matrices.
	 * Allocation Strategy: Uses `calloc` to allocate and zero-initialize memory for `N * N`
	 *                      double-precision elements for each intermediate result.
	 * Error Handling: As with previous allocations, in a production environment, each `calloc`
	 *                 call should be followed by a null-pointer check.
	 */
	double *first_mul = calloc(N * N, sizeof(double));      /**< Stores the result of the partial matrix multiplication `A * B` (triangular part). */
	double *first_mul_helper = calloc(N * N, sizeof(double)); /**< Stores the result of the matrix multiplication `first_mul * Bt`. */
	double *Bt = get_transpose(B, N);                       /**< Stores the transpose of the input matrix `B`, obtained via `get_transpose`. */
	
	/**
	 * Block Logic: Computes a partial matrix multiplication `first_mul = A * B`.
	 *              The inner loop bounds (`k = i; k < N`) indicate that this operation
	 *              is effectively accumulating only the elements corresponding to an
	 *              upper triangular multiplication with respect to `B`'s columns.
	 * Algorithm: Triple-nested loop for matrix multiplication.
	 * Pre-condition: `A` and `B` are the input matrices.
	 * Invariant: `first_mul[i*N + j]` accumulates the dot product of the i-th row of `A`
	 *            and the j-th column of `B`, constrained by the upper triangular access pattern.
	 */
	for (int i = 0; i < N; ++i) { // Iterates over rows of the resulting `first_mul` matrix.
		for (int j = 0; j < N; ++j) { // Iterates over columns of the `first_mul` matrix.
			for (int k = i; k < N; ++k) { // Control Flow: Loop bound (`k = i` to `N`) implies an upper triangular matrix operation.
				// Inline: Accumulates the product of `A[i][k]` and `B[k][j]` into `first_mul[i][j]`.
				first_mul[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Computes the full matrix multiplication `first_mul_helper = first_mul * Bt`.
	 *              This involves a standard triple-nested loop where the outer loops iterate
	 *              through the resulting matrix elements, and the innermost loop computes the
	 *              dot product of a row from `first_mul` and a column from `Bt`.
	 * Algorithm: Standard matrix multiplication.
	 * Pre-condition: `first_mul` contains the partial product `A * B`, and `Bt` contains the transpose of `B`.
	 * Invariant: `first_mul_helper[i*N + j]` accumulates the dot product of the i-th row of `first_mul`
	 *            and the j-th column of `Bt`, representing an element of the product matrix.
	 */
	for (int i = 0; i < N; ++i) { // Iterates over rows of the resulting `first_mul_helper` matrix.
		for (int j = 0; j < N; ++j) { // Iterates over columns of the `first_mul_helper` matrix.
			for (int k = 0; k < N; ++k) { // Iterates over the elements for the dot product calculation.
				// Inline: Accumulates the product of `first_mul[i][k]` and `Bt[k][j]` into `first_mul_helper[i][j]`.
				first_mul_helper[i * N + j] += first_mul[i * N + k] * Bt[k * N + j];
			}
		}
	}

	/**
	 * Functional Utility: Overwrites the content of `first_mul` with the result stored in `first_mul_helper`.
	 * Rationale: This operation reuses the `first_mul` memory block to hold the intermediate product
	 *            `(A * U) * Bt` (where U is the implicit upper triangular part of B, from the earlier `first_mul` calculation)
	 *            before the final element-wise addition. This is an optimization to reduce the total
	 *            number of distinct memory allocations.
	 * Parameters:
	 *   `first_mul`: Destination memory block, which previously held `A * B` (triangular part).
	 *   `first_mul_helper`: Source memory block, containing `first_mul * Bt`.
	 *   `N * N * sizeof(double)`: Number of bytes to copy (size of an N x N matrix of doubles).
	 */
	memcpy(first_mul, first_mul_helper, N * N * sizeof(double));
	
	/**
	 * Block Logic: Performs the final element-wise addition of the `second_mul` matrix
	 *              to the `first_mul` matrix. This completes the full matrix expression.
	 * Functional Utility: Combines the two main intermediate products to yield the final
	 *                     computed result, storing it in `first_mul`.
	 * Pre-condition: `first_mul` holds the result `(A * U) * Bt`, and `second_mul` holds `At * A` (lower triangular part).
	 * Invariant: After the loops, `first_mul` contains the final solution matrix.
	 */
	for (int i = 0; i < N; ++i) { // Iterates over rows of the matrices.
		for (int j = 0; j < N; ++j) { // Iterates over columns of the matrices.
			// Inline: Adds the element `second_mul[i][j]` to `first_mul[i][j]`, updating `first_mul` in place.
			first_mul[i * N + j] += second_mul[i * N + j];
		}
	}

	/**
	 * Block Logic: Releases all dynamically allocated memory for intermediate matrices.
	 * Functional Utility: Crucial for preventing memory leaks and ensuring efficient resource
	 *                     management, especially in long-running applications or repeated calls.
	 * Pre-condition: `first_mul_helper`, `At`, `Bt`, and `second_mul` point to valid,
	 *                previously allocated memory.
	 * Post-condition: The memory blocks pointed to by `first_mul_helper`, `At`, `Bt`,
	 *                 and `second_mul` are deallocated.
	 */
	free(first_mul_helper); // Resource Management: Deallocates memory used for `first_mul_helper`.
	free(At);               // Resource Management: Deallocates memory used for the transposed `A` matrix.
	free(Bt);               // Resource Management: Deallocates memory used for the transposed `B` matrix.
	free(second_mul);       // Resource Management: Deallocates memory used for `second_mul`.

	return first_mul; // Functional Utility: Returns the pointer to the final computed matrix, which the caller is responsible for freeing.
}
