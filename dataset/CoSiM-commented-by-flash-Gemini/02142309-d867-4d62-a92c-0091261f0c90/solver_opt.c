/**
 * @file solver_opt.c
 * @brief Implements a manually "optimized" matrix solver using pure C.
 * This file provides an optimized implementation for computing the matrix expression
 * `C = (A * B) * B^T + A^T * A`. It applies manual C-level optimizations such as
 * `register` keyword usage for frequently accessed variables and direct pointer arithmetic
 * for efficient array traversal, aiming for improved performance over unoptimized implementations.
 * Algorithm: The solver performs the following sequence of matrix operations:
 *   1. Computes `At` (transpose of A) and `Bt` (transpose of B), storing them in separate matrices. This is done
 *      using direct pointer manipulation for efficiency.
 *   2. Calculates intermediate matrix `D` as `A * B`, where `A` is implicitly treated as an
 *      upper triangular matrix (elements `A[i][k]` where `i <= k` are considered). This step is
 *      optimized with pointer arithmetic for element access.
 *   3. Calculates intermediate matrix `E` as `D * B^T`. The transpose of `B` is implicitly handled
 *      by accessing `B[j * N + k]` via pointer arithmetic.
 *   4. Calculates intermediate matrix `F` as `A^T * A`, where the `A^T * A` part is computed
 *      using the pre-calculated `At` matrix and implicit triangular considerations, optimized
 *      with pointer arithmetic for element access.
 *   5. Computes the final result `C = E + F` through element-wise addition, using pointer arithmetic for access.
 * Optimization: Employs manual C-level optimizations:
 *   - `register` keyword: Hints to the compiler to store loop counters and frequently used pointers in CPU registers for faster access, reducing memory access latency.
 *   - Pointer arithmetic: Direct manipulation of pointers for array element access, which can reduce address calculation overhead compared to array indexing.
 *   - Loop structure and memory access patterns: Designed to potentially enhance cache locality by accessing contiguous memory blocks, minimizing cache misses.
 * Time Complexity: Theoretically $O(N^3)$ for $N \times N$ matrices due to three-nested-loop matrix multiplications.
 * However, the manual optimizations aim to significantly reduce the constant factor, leading to improved practical performance.
 * Space Complexity: $O(N^2)$ for storing the final result matrix `C` and several intermediate matrices (`At`, `Bt`, `D`, `E`, `F`).
 */

#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	// Inline: Declare loop counters as 'register' to suggest CPU register storage for performance optimization.
	register int i;
	register int j;
	register int k;

	printf("OPT SOLVER\n");
	// Block Logic: Allocate memory for the final result matrix 'C'.
	double *C = malloc(N * N * sizeof(double));
	// Block Logic: Allocate memory for transposed matrix A ('At').
	double *At = malloc(N * N * sizeof(double));
	// Block Logic: Allocate memory for transposed matrix B ('Bt').
	double *Bt = malloc(N * N * sizeof(double));
	// Block Logic: Allocate memory for intermediate matrix 'D'.
	double *D = malloc(N * N * sizeof(double));
	// Block Logic: Allocate memory for intermediate matrix 'E'.
	double *E = malloc(N * N * sizeof(double));
	// Block Logic: Allocate memory for intermediate matrix 'F'.
	double *F = malloc(N * N * sizeof(double));

	// Conditional Logic: Handle memory allocation failure for any of the critical matrices.
	if(C == NULL || At == NULL || Bt == NULL) {
		printf("Eroare la alocare");
		return NULL;
	}

	// Block Logic: Compute transposes of matrices A and B using pointer arithmetic.
	// Optimization: Uses explicit pointers and increments to access matrix elements, aiming for faster memory access.
	// Invariant: After this block, At[j*N + i] will hold A[i*N + j] (A^T) and Bt[j*N + i] will hold B[i*N + j] (B^T).
	for(i = 0; i < N; i++) {
		// Inline: Pointers for column-major writing to 'At' and 'Bt' (transposed matrices).
		register double *At_ptr = At + i;
		register double *Bt_ptr = Bt + i;

		// Inline: Pointers for row-major reading from original matrices A and B.
		register double *A_ptr = A + i * N;
		register double *B_ptr = B + i * N;

		for(j = 0; j < N; j++) {
			*At_ptr = *A_ptr; // Inline: Copy A[i][j] to At[j][i].
			*Bt_ptr = *B_ptr; // Inline: Copy B[i][j] to Bt[j][i].

			At_ptr += N; // Inline: Move to the next row in the transpose matrix (effectively next column in original).
			Bt_ptr += N; // Inline: Move to the next row in the transpose matrix (effectively next column in original).

			A_ptr++; // Inline: Move to the next element in the current row of A.
			B_ptr++; // Inline: Move to the next element in the current row of B.
		}
	}

	// Block Logic: Compute intermediate matrix D = A * B (with implicit upper triangular A).
	// Optimization: Uses explicit pointer arithmetic for matrix element access.
	// Precondition: 'A' and 'Bt' are N x N matrices.
	// Invariant: D[i * N + j] will hold the sum of A[i][k] * B[k][j] for k from i to N-1.
	for(i = 0; i < N; i++) {
		// Inline: Pointer to the start of current row 'i' in result matrix 'D'.
		register double *D_ptr = D + i * N;

		for(j = 0; j < N; j++) {
			// Inline: Initialize sum for the current element D[i][j].
			register double result = 0;

			// Inline: Pointer for current element in matrix A (row 'i').
			register double *A_ptr = A + i * N;
			// Inline: Pointer for current element in transposed matrix 'Bt' (column 'j').
			register double *Bt_ptr = Bt + j * N;

			for(k = 0; k < N; k++) {
				// Conditional Logic: Implies A is upper triangular, only considering A[i][k] where i <= k.
				if(i <= k) {
					result += *(A_ptr + k) * *(Bt_ptr + k); // Inline: Accumulate product of elements.
				}
			}

			*D_ptr = result; // Inline: Store computed sum into D[i][j].
			D_ptr++; // Inline: Move to the next element in the current row of D.
		}
	}

	// Block Logic: Compute intermediate matrix E = D * B^T.
	// Optimization: Uses explicit pointer arithmetic for matrix element access.
	// Precondition: 'D' and 'B' are N x N matrices.
	// Invariant: E[i * N + j] will hold the sum of D[i][k] * B[j][k] for k from 0 to N-1.
	for(i = 0; i < N; i++) {
		// Inline: Pointer to the start of current row 'i' in result matrix 'E'.
		register double *E_ptr = E + i * N;

		for(j = 0; j < N; j++) {
			// Inline: Initialize sum for the current element E[i][j].
			register double result = 0;

			// Inline: Pointer for current element in matrix 'D' (row 'i').
			register double *D_ptr = D + i * N;
			// Inline: Pointer for current element in matrix 'B' (row 'j').
			register double *B_ptr = B + j * N;

			for(k = 0; k < N; k++) {
				result += *D_ptr * *B_ptr; // Inline: Accumulate product of elements.
				D_ptr++; // Inline: Move to next element in current row of D.
				B_ptr++; // Inline: Move to next element in current row of B.
			}

			*E_ptr = result; // Inline: Store computed sum into E[i][j].
			E_ptr++; // Inline: Move to the next element in the current row of E.
		}
	}

	// Block Logic: Compute intermediate matrix F = A^T * A (with implicit triangular access).
	// Optimization: Uses explicit pointer arithmetic for matrix element access.
	// Precondition: 'At' and 'A' are N x N matrices.
	// Invariant: F[i * N + j] will hold the sum of At[i][k] * At[j][k] (equivalent to A[k][i] * A[k][j]).
	for(i = 0; i < N; i++) {
		// Inline: Pointer to the start of current row 'i' in transposed matrix 'At'.
		register double *F_ptr = F + i * N;

		for(j = 0; j < N; j++) {
			// Inline: Initialize sum for the current element F[i][j].
			register double result = 0;

			// Inline: Pointer for current element in transposed matrix 'At' (row 'i').
			register double *At_ptr1 = At + i * N;
			// Inline: Pointer for current element in transposed matrix 'At' (row 'j').
			register double *At_ptr2 = At + j * N;

			for(k = 0; k < N; k++) {
				// Conditional Logic: Implies considering elements for A^T * A. The condition `k >= i || k <= j`
				// seems to be incorrect logic for standard A^T * A. It might be a specific problem constraint.
				// For a correct A^T * A, the condition should be removed or changed to k from 0 to N-1.
				if(k >= i || k <= j) { // This condition is problematic for a general A^T * A.
					result += *(At_ptr1 + k) * *(At_ptr2 + k); // Inline: Accumulate product of elements.
				}
			}

			*F_ptr = result; // Inline: Store computed sum into F[i][j].
			F_ptr++; // Inline: Move to the next element in the current row of F.
		}
	}

	// Block Logic: Compute final result: C = E + F.
	// This loop performs element-wise addition of the two intermediate result matrices using pointer arithmetic.
	// Precondition: 'E' and 'F' are N x N matrices.
	// Invariant: C[idx] holds the sum of E[idx] and F[idx].
	for(i = 0; i < N * N; i++) {
		// Inline: Pointers to the current elements in C, E, and F.
		register double *C_ptr = C + i;
		register double *E_ptr = E + i;
		register double *F_ptr = F + i;

		*C_ptr = *E_ptr + *F_ptr; // Inline: Perform element-wise addition.
	}

	// Block Logic: Free memory allocated for intermediate matrices.
	free(At);
	free(Bt);
	free(D);
	free(E);
	free(F);

	// Functional Utility: Return the final computed result matrix C.
	return C;
}
