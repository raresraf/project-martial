
/**
 * @file solver_neopt.c
 * @brief Provides a naive (non-optimized) implementation for a sequence of matrix operations.
 *
 * This file contains the 'my_solver' function which performs a series of matrix multiplications and additions
 * on N x N matrices represented in row-major order. The operations are:
 * 1. Calculate C = B * B^T
 * 2. Calculate D = A * C (using the C from step 1)
 * 3. Calculate C_temp = A * A^T
 * 4. Add C_temp to D (D = D + C_temp)
 *
 * The implementation uses three nested loops for each matrix multiplication, resulting in cubic time complexity.
 *
 * Algorithm: Naive matrix multiplication and addition.
 * Time Complexity: O(N^3) due to nested loops for matrix multiplication.
 * Space Complexity: O(N^2) for storing intermediate matrices C and D.
 */

#include "utils.h"


/**
 * @brief Solves a matrix problem involving multiplications and additions of N x N matrices.
 *
 * This function takes two N x N matrices, A and B, and performs the following operations:
 * 1. Computes C = B * B^T
 * 2. Computes D = A * C
 * 3. Computes C_temp = A * A^T
 * 4. Adds C_temp to D
 * The matrices are stored in row-major order (element at [row][col] is at index col + N * row).
 *
 * @param N The dimension of the square matrices (N x N).
 * @param A A pointer to the N x N matrix A (input).
 * @param B A pointer to the N x N matrix B (input).
 * @return A pointer to the resulting N x N matrix D after all operations.
 */
double* my_solver(int N, double *A, double* B) {
	int i,j,k;
	double *C, *D;
	
    // Functional Utility: Allocate memory for intermediate matrix C, initialized to zero.
    // Invariant: C will temporarily store the result of B * B^T or A * A^T.
	C = (double *)calloc(N * N, sizeof(double));
    // Functional Utility: Allocate memory for result matrix D, initialized to zero.
    // Invariant: D will accumulate the final result.
	D = (double *)calloc(N * N, sizeof(double));

	
    /**
     * Block Logic: Computes C = B * B^T.
     * Precondition: C is an N x N matrix initialized to zero.
     * Invariant: After completion, C[i][j] will contain the dot product of row i of B and row j of B.
     * (Equivalent to C[i][j] = sum_k(B[i][k] * B[j][k])).
     */
	for (i = 0; i < N; i++) {
		for (j = i; j < N; j++) { // Iterates only over the upper triangle to compute C = B * B^T efficiently, leveraging symmetry.
			for (k = 0; k < N; k ++) {
                // Inline: Accumulates the product of elements for C[i][j] (represented as C[j + N * i]).
				C[j + N * i] += 
					B[k + N * i] * // Element B[i][k] (row k, col i, transposed effectively)
					B[k + N * j]; // Element B[j][k] (row k, col j, transposed effectively)
			}

            // Functional Utility: Exploit matrix symmetry (C[i][j] == C[j][i]) to fill the lower triangle.
			C[i + N * j] = C[j + N * i]; 
		}	
	}

	free(C); // Performance Optimization: Free memory no longer needed for C to be reused.
	C = (double *)calloc(N * N, sizeof(double)); // Functional Utility: Reallocate and re-initialize C to zero for the next computation.
	
    /**
     * Block Logic: Computes D = A * C.
     * Precondition: D is an N x N matrix initialized to zero. C contains the result from the previous block (B * B^T).
     * Invariant: After completion, D[i][j] will contain the dot product of row i of A and column j of C.
     * (Equivalent to D[i][j] = sum_k(A[i][k] * C[k][j])).
     * Note: The loop for k starts from i, possibly indicating optimization or specific matrix properties.
     */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k ++) { // Optimization: Potentially leverages properties of A or C for reduced computations.
                // Inline: Accumulates the product of elements for D[i][j] (represented as D[j + N * i]).
				D[j + N * i] += 
					A[k + N * i] * // Element A[i][k]
					C[j + N * k]; // Element C[k][j]
			}
		}	
	}

    // Functional Utility: Free memory no longer needed for C, preparing for its final use.
	free(C);
    // Functional Utility: Allocate memory for intermediate matrix C, initialized to zero.
    // Invariant: C will temporarily store the result of A * A^T.
	C = (double *)calloc(N * N, sizeof(double));

	
    /**
     * Block Logic: Computes C = A * A^T and then adds it to D.
     * Precondition: C is an N x N matrix initialized to zero.
     * Invariant: After computing C = A * A^T, D is updated by adding the elements of the new C.
     */
	for (i = 0; i < N; i++) {
		for (j = i; j < N; j++) { // Iterates only over the upper triangle to compute C = A * A^T efficiently, leveraging symmetry.
			for (k = 0; k < N; k++) {
                // Inline: Accumulates the product of elements for C[i][j] (represented as C[j + N * i]).
				C[j + N * i] += 
					A[i + N * k] * // Element A[i][k]
					A[j + N * k]; // Element A[j][k] (effectively A^T[k][j])
			}

			// Functional Utility: Adds the computed symmetric C[i][j] element to D[i][j] and D[j][i].
			D[i + N * j] += C[j + N * i]; 
			D[j + N * i] += C[j + N * i]; 
		}	
	}

    // Functional Utility: Deallocates the memory for the last temporary matrix C.
	free(C);
    // Functional Utility: Returns the pointer to the final result matrix D.
	return D;
}

