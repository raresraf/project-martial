/**
 * @00b94dd9-4464-47d5-b817-b1dcd44f8586/solver_opt.c
 * @brief Implements an "optimized" matrix solver (`my_solver`) using manual
 *        performance enhancements for explicit matrix operations, without
 *        relying on external highly-optimized libraries like BLAS.
 *
 * This file showcases techniques for improving the efficiency of matrix computations
 * at a low level. Optimizations typically include strategic use of pointer arithmetic,
 * `register` keyword hints for frequently accessed variables, and careful design of
 * loop access patterns to enhance data locality and minimize cache misses. This
 * approach aims to achieve better performance than a naive implementation (`solver_neopt.c`)
 * through direct control over memory access and CPU resource utilization.
 *
 * It includes a helper function to compute matrix transposes, also optimized for this context.
 *
 * Algorithm: Direct implementation of matrix operations (multiplication, addition, transposition)
 *            with specific manual optimizations for cache efficiency, reduced instruction count,
 *            and exploitation of memory access patterns. This includes pointer arithmetic
 *            within inner loops and partial triangular matrix multiplications.
 * Time Complexity: The computational complexity is dominated by the triple-nested loops
 *                  for matrix multiplications, resulting in an overall time complexity of O(N^3),
 *                  where N is the dimension of the square matrices.
 * Space Complexity: O(N^2) for storing auxiliary matrices (`first_mul`, `first_mul_aux`,
 *                   `second_mul`, `At`, `Bt`) created during the computation.
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
 * @brief Implements a matrix solver with manual optimizations for matrix operations.
 *
 * This function calculates a complex matrix expression involving input matrices A and B.
 * It performs various matrix multiplications and additions using explicit nested loops,
 * often incorporating pointer arithmetic and `register` hints to improve memory access
 * patterns and potentially leverage CPU registers for performance.
 *
 * @param N The dimension of the square matrices (N x N).
 * @param A A pointer to the first input square matrix.
 * @param B A pointer to the second input square matrix.
 * @return A pointer to the newly allocated result matrix. The caller is
 *         responsible for freeing this memory.
 *
 * Algorithm: Sequence of matrix operations with manual loop unrolling/optimization strategies.
 * Time Complexity: Dominated by matrix multiplications, O(N^3).
 * Space Complexity: O(N^2) for `first_mul`, `first_mul_aux`, `second_mul`, `At`, and `Bt`.
 */
double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n"); // Functional Utility: Prints an identifier to standard output, indicating that the manually optimized solver is being utilized.

	/**
	 * Functional Utility: Dynamically allocates memory for intermediate matrices required for the optimized computation.
	 * Allocation Strategy: Uses `calloc` to allocate `N * N` elements of type `double` and initializes all bytes to zero,
	 *                      providing a clean slate for accumulation in matrix products.
	 * Error Handling: In a robust production environment, each `calloc` call should be immediately followed
	 *                 by a null-pointer check to ensure memory was successfully allocated.
	 */
	double *first_mul = calloc(N * N, sizeof(double));     /**< Stores the result of the optimized partial matrix multiplication `A * B` (triangular part). */
	double *first_mul_aux = calloc(N * N, sizeof(double)); /**< Stores the accumulating result of subsequent matrix operations, eventually holding the final solution. */
	double *second_mul = calloc(N * N, sizeof(double));    /**< Stores the result of the optimized partial matrix multiplication `At * A` (triangular part). */
	
	/**
	 * Block Logic: Computes the transposes of the input matrices `B` and `A`.
	 * Functional Utility: These transposed matrices (`Bt` and `At`) are essential intermediate
	 *                     structures for the optimized matrix multiplication kernels that follow,
	 *                     allowing for specific memory access patterns or logical operations.
	 * Resource Management: The memory for `Bt` and `At` is dynamically allocated within `get_transpose`
	 *                      and must be freed by the caller of `my_solver`.
	 */
	double *Bt = get_transpose(B, N); /**< Stores the transpose of the input matrix `B`. */
	double *At = get_transpose(A, N); /**< Stores the transpose of the input matrix `A`. */

	/**
	 * Block Logic: Computes a partial matrix multiplication `first_mul = A * B`
	 *              with manual optimizations aimed at enhancing performance.
	 * Optimization Strategy: Utilizes `register` variables for frequently accessed pointers and
	 *                       accumulators, and employs pointer arithmetic within the inner loop
	 *                       to potentially improve memory access patterns and cache utilization.
	 *                       The inner loop bounds (`k = i; k < N`) indicate a focus on an
	 *                       upper triangular multiplication with respect to `B`.
	 * Algorithm: Optimized triple-nested loop for matrix multiplication.
	 * Pre-condition: `A` and `B` are the input matrices.
	 * Invariant: `first_mul[i*N + j]` stores the accumulated dot product of the i-th row of `A`
	 *            and the j-th column of `B`, considering the optimized access pattern.
	 */
	for (int i = 0; i < N; ++i) { // Outer loop: Iterates over rows of the resulting `first_mul` matrix.
		register double *aux = &A[i * N]; // Optimization: Pointer to the current row of `A`, potentially stored in a register.
		for (int j = 0; j < N; ++j) { // Middle loop: Iterates over columns of the resulting `first_mul` matrix.
			register double *collumn = &B[j]; // Optimization: Pointer to the current column of `B`, potentially stored in a register.
			register double rez = 0;         // Optimization: Accumulator for the dot product, potentially stored in a register.

			// Inner loop for dot product calculation.
			// Optimization: Uses pointer arithmetic and specific loop bounds (`k = i; k < N`)
			//               to target an upper triangular multiplication.
			// Note: The variable `line` is undeclared here, which would cause a compilation error.
			//       Assuming `line` is intended to relate to `aux` or `A`.
			for (int k = i; k < N; ++k, line++, collumn += N) {
				// Inline: Performs element multiplication and accumulation. The specific indexing
				//         `*(line + i)` and `*(collumn + i * N)` suggests a complex access pattern
				//         intended for optimization, though `line` is undefined.
				rez += *(line + i) * *(collumn + i * N);
			}
			first_mul[i * N + j] = rez; // Functional Utility: Stores the computed dot product into the `first_mul` matrix.
		}
	}


	/**
	 * Block Logic: Computes the matrix multiplication `first_mul_aux = first_mul * Bt`
	 *              with manual optimizations using pointer arithmetic and `register` variables.
	 * Optimization Strategy: Employs `register` variables for pointers and accumulators, and
	 *                       utilizes pointer arithmetic for efficient traversal of matrix elements
	 *                       during the dot product calculation.
	 * Algorithm: Optimized triple-nested loop for standard matrix multiplication.
	 * Pre-condition: `first_mul` contains the partial product from the previous step,
	 *                and `Bt` contains the transpose of `B`.
	 * Invariant: `first_mul_aux[i*N + j]` stores the accumulated dot product of the i-th row of `first_mul`
	 *            and the j-th column of `Bt`.
	 */
	for (int i = 0; i < N; ++i) { // Outer loop: Iterates over rows of the resulting `first_mul_aux` matrix.
		register double *aux = &first_mul[i * N]; // Optimization: Pointer to the current row of `first_mul`, potentially stored in a register.
		for (int j = 0; j < N; ++j) { // Middle loop: Iterates over columns of the resulting `first_mul_aux` matrix.
			register double *line = aux;          // Optimization: Pointer to traverse the current row of `first_mul`.
			register double *collumn = &Bt[j];    // Optimization: Pointer to traverse the current column of `Bt`.
			register double res = 0;              // Optimization: Accumulator for the dot product, potentially stored in a register.

			// Inner loop for dot product calculation.
			// Optimization: Uses pointer arithmetic (`line++`, `collumn += N`) for efficient element access.
			for (int k = 0; k < N; ++k, line++, collumn += N) {
				// Inline: Performs element multiplication and accumulation.
				res += *line * *collumn;
			}
			first_mul_aux[i * N + j] = res; // Functional Utility: Stores the computed dot product into the `first_mul_aux` matrix.
		}
	}

	/**
	 * Block Logic: Computes a partial matrix multiplication `second_mul = At * A`
	 *              with manual optimizations using pointer arithmetic and `register` variables.
	 * Optimization Strategy: Employs `register` variables for pointers and accumulators, and
	 *                       utilizes pointer arithmetic for efficient traversal of matrix elements.
	 *                       The inner loop bounds (`k = 0; k <= i`) suggest a multiplication
	 *                       involving the lower triangular part with respect to `A`.
	 * Algorithm: Optimized triple-nested loop for matrix multiplication with triangular access.
	 * Pre-condition: `At` contains the transpose of `A`, and `A` is the original matrix.
	 * Invariant: `second_mul[i*N + j]` stores the accumulated dot product of the i-th row of `At`
	 *            and the j-th column of `A`, constrained by the optimized triangular access pattern.
	 */
	for (int i = 0; i < N; ++i) { // Outer loop: Iterates over rows of the resulting `second_mul` matrix.
		register double *aux = &At[i * N]; // Optimization: Pointer to the current row of `At`, potentially stored in a register.
		for (int j = 0; j < N; ++j) { // Middle loop: Iterates over columns of the resulting `second_mul` matrix.
			register double *line = aux;        // Optimization: Pointer to traverse the current row of `At`.
			register double *collumn = &A[j];   // Optimization: Pointer to traverse the current column of `A`.
			register double res = 0;            // Optimization: Accumulator for the dot product, potentially stored in a register.

			// Inner loop for dot product calculation.
			// Optimization: Uses pointer arithmetic (`line++`, `collumn += N`) and specific loop bounds (`k = 0; k <= i`)
			//               to target a lower triangular multiplication.
			for (int k = 0; k <= i; ++k, line++, collumn += N) {
				// Inline: Performs element multiplication and accumulation.
				res += *line * *collumn;
			}
			second_mul[i * N + j] = res; // Functional Utility: Stores the computed dot product into the `second_mul` matrix.
		}
	}

	/**
	 * Block Logic: Performs the final element-wise addition of the `second_mul` matrix
	 *              to the `first_mul_aux` matrix. This step completes the full matrix expression.
	 * Functional Utility: Combines the two main intermediate products to yield the final
	 *                     computed result, storing it in `first_mul_aux`.
	 * Pre-condition: `first_mul_aux` holds the result `(A * U) * Bt` (from previous optimized multiplications),
	 *                and `second_mul` holds `At * A` (from previous optimized multiplication).
	 * Invariant: After the loops, `first_mul_aux` contains the final solution matrix,
	 *            which is the sum of its previous content and the corresponding element from `second_mul`.
	 */
	for (int i = 0; i < N; ++i) { // Iterates over rows of the matrices.
		for (int j = 0; j < N; ++j) { // Iterates over columns of the matrices.
			// Inline: Adds the element `second_mul[i][j]` to `first_mul_aux[i][j]`, updating `first_mul_aux` in place.
			first_mul_aux[i * N + j] += second_mul[i * N + j];
		}
	}

	/**
	 * Block Logic: Releases all dynamically allocated memory for intermediate matrices.
	 * Functional Utility: Crucial for preventing memory leaks and ensuring efficient resource
	 *                     management, especially in long-running applications or repeated calls.
	 * Pre-condition: `first_mul`, `second_mul`, `At`, and `Bt` point to valid, previously allocated memory.
	 * Post-condition: The memory blocks pointed to by `first_mul`, `second_mul`, `At`, and `Bt` are deallocated.
	 */
	free(first_mul);      // Resource Management: Deallocates memory used for `first_mul`.
	free(second_mul);     // Resource Management: Deallocates memory used for `second_mul`.
	free(At);             // Resource Management: Deallocates memory used for the transposed `A` matrix.
	free(Bt);             // Resource Management: Deallocates memory used for the transposed `B` matrix.

	return first_mul_aux; // Functional Utility: Returns the pointer to the final computed matrix, which the caller is responsible for freeing.
}
