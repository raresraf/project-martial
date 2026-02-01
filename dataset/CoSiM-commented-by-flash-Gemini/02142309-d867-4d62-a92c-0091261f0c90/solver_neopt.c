/**
 * @file solver_neopt.c
 * @brief Implements a "non-optimized" matrix solver using basic C loops for matrix operations.
 * This file provides a baseline implementation for computing the matrix expression `C = (A * B) * B^T + A^T * A`.
 * It intentionally avoids specialized linear algebra libraries or manual performance optimizations,
 * serving as a clear reference for the computational steps involved and for performance comparison
 * against highly optimized versions.
 * Algorithm: The solver performs the following sequence of operations using explicit nested loops:
 *   1. Calculates intermediate matrix `D` as `A * B`, where `A` is implicitly treated as an
 *      upper triangular matrix (elements `A[i][k]` where `i <= k` are considered).
 *   2. Calculates intermediate matrix `E` as `D * B^T`. The transpose of `B` is handled
 *      by accessing `B[j * N + k]` in the innermost loop for each element of `E`.
 *   3. Calculates the final result `C`. This involves adding `E` with `A^T * A`.
 *      The `A^T * A` part is computed within the innermost loop for `C[i][j]`, implicitly
 *      considering triangular parts of the multiplication (`A[k][i]` and `A[k][j]` from original `A`).
 * Optimization: This implementation relies solely on the compiler's default optimizations for nested loops.
 * It does not employ specific techniques like loop unrolling, cache blocking, or SIMD instructions,
 * making it suitable as a clear, unoptimized reference.
 * Time Complexity: Dominated by the explicit three-nested-loop matrix multiplications, resulting in an $O(N^3)$
 * complexity for operations on $N \times N$ matrices.
 * Space Complexity: $O(N^2)$ for storing the final result matrix `C` and intermediate matrices `D` and `E`.
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	// Block Logic: Allocate memory for the final result matrix 'C'.
	double *C = calloc(N * N, sizeof(double));
	// Block Logic: Allocate memory for intermediate matrix 'D'.
	double *D = calloc(N * N, sizeof(double));
	// Block Logic: Allocate memory for intermediate matrix 'E'.
	double *E = calloc(N * N, sizeof(double));

	int i, j, k;

	// Conditional Logic: Handle memory allocation failure for any of the matrices.
	if(C == NULL || D == NULL || E == NULL) {
		printf("Eroare la alocare");
		return NULL;
	}

	// Block Logic: Compute intermediate matrix D = A * B.
	// This multiplication implicitly uses only the upper triangular part of A.
	// Invariant: D[i * N + j] will hold the sum of A[i][k] * B[k][j] for k from i to N-1.
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			double sum = 0; // Inline: Accumulator for the dot product.
			for(k = 0; k < N; k++) {
				// Conditional Logic: Implies A is upper triangular, only considering A[i][k] where i <= k.
				if(i <= k) {
					sum += A[i * N + k] * B[k * N + j];
				}
			}
			D[i * N + j] = sum;
		}
	}

	// Block Logic: Compute intermediate matrix E = D * B^T.
	// This multiplication involves matrix D and the transpose of B.
	// Invariant: E[i * N + j] will hold the sum of D[i][k] * B[j][k] for k from 0 to N-1.
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			double sum = 0; // Inline: Accumulator for the dot product.
			for(k = 0; k < N; k++) {
				// Inline: Access B[j * N + k] effectively transposes B (B^T[k][j] = B[j][k]).
				sum += D[i * N + k] * B[j * N + k];
			}
			E[i * N + j] = sum;
		}
	}

	// Block Logic: Compute final result C = E + (A^T * A).
	// The A^T * A part is computed on-the-fly and added to E.
	// Invariant: C[i * N + j] will hold E[i * N + j] + (sum of A[k][i] * A[k][j] for k from 0 to N-1).
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			double sum = 0; // Inline: Accumulator for the A^T * A part.
			for(k = 0; k < N; k++) {
				// Conditional Logic: Implies considering elements for A^T * A. The condition `k >= i || k <= j`
				// seems to be incorrect logic for standard A^T * A. It might be a specific problem constraint.
				// For a correct A^T * A, the condition should be removed or changed to k from 0 to N-1.
				if(k >= i || k <= j) { // This condition is problematic for a general A^T * A.
					sum += A[k * N + i] * A[k * N + j];
				}
			}
			C[i * N + j] = E[i * N + j] + sum;
		}
	}

	// Block Logic: Free memory allocated for intermediate matrices.
	free(D);
	free(E);

	// Functional Utility: Return the final computed result matrix C.
	return C;
}
