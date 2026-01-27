
/**
 * @file solver_opt.c
 * @brief Implements an optimized matrix computation solver using manual optimizations.
 *
 * This module defines a solver function that calculates C = (A_upper * B) * B^T + (A^T * A)
 * with manual optimizations for improved performance. These optimizations include
 * explicit transposition, extensive use of `register` keywords, and manual pointer
 * arithmetic to potentially enhance cache utilization and reduce memory access latency.
 *
 * Algorithm: Optimized nested-loop matrix multiplication with manual register allocation and pointer arithmetic.
 * Time Complexity: O(N^3) (despite manual optimizations, matrix multiplication remains cubic).
 * Space Complexity: O(N^2) for intermediate matrix storage and transpositions.
 */
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	// Functional Utility: Indicates the use of the optimized solver.
	printf("OPT SOLVER\n");
	// Block Logic: Allocates memory for the result matrix C and several intermediate matrices (At, Bt, D, E, F).
	// Invariant: All allocated pointers must be checked for NULL to prevent dereferencing issues.
	double *C = malloc(N * N * sizeof(double));
	double *At = malloc(N * N * sizeof(double)); // Stores transpose of A
	double *Bt = malloc(N * N * sizeof(double)); // Stores transpose of B
	double *D = malloc(N * N * sizeof(double)); // Intermediate result D
	double *E = malloc(N * N * sizeof(double)); // Intermediate result E
	double *F = malloc(N * N * sizeof(double)); // Intermediate result F

	// Block Logic: Error handling for memory allocation failures.
	// Pre-condition: Memory allocation attempts have just been made.
	// Invariant: If any allocation fails, print an error and return NULL.
	if(C == NULL || At == NULL || Bt == NULL || D == NULL || E == NULL || F == NULL) { // Added checks for D, E, F
		printf("Eroare la alocare");
		return NULL;
	}

	// Inline: Declares loop counters as `register` for potential CPU register allocation,
	// aiming to reduce memory access time for frequently used variables.
	register int i;
	register int j;
	register int k;

	/**
	 * Block Logic: Computes the transposes of matrices A and B (At and Bt).
	 * This pre-computation is an optimization to facilitate column-major access
	 * during subsequent matrix multiplications, improving cache locality.
	 * Pre-condition: A and B are N x N matrices.
	 * Invariant: At[j][i] = A[i][j] and Bt[j][i] = B[i][j] after this block.
	 */
	for(i = 0; i < N; i++) {
		// Inline: Uses register pointers for faster access to matrix elements during transposition.
		register double *At_ptr = At + i; 
		register double *Bt_ptr = Bt + i; 

		register double *A_ptr = A + i * N; 
		register double *B_ptr = B + i * N; 

		for(j = 0; j < N; j++) {
			// Inline: Performs transposition for matrix A.
			*At_ptr = *A_ptr;
			// Inline: Performs transposition for matrix B.
			*Bt_ptr = *B_ptr;

			// Inline: Advances column pointers for transposed matrices.
			At_ptr += N;
			Bt_ptr += N;
			
			// Inline: Advances row pointers for original matrices.
			A_ptr++;
			B_ptr++;
		}
	}

	/**
	 * Block Logic: Computes the intermediate matrix D, where D = A_upper * B.
	 * A_upper refers to A where elements A[i][k] are considered zero if i > k.
	 * Utilizes register variables and manual pointer arithmetic for performance.
	 * Pre-condition: Matrices A and Bt are N x N, Bt contains the transpose of B.
	 * Invariant: D[i][j] contains the sum of A[i][k] * B[k][j] for k >= i.
	 */
	for(i = 0; i < N; i++) {
		register double *D_ptr = D + i * N; 

		for(j = 0; j < N; j++) {

			register double result = 0;

			// Inline: Uses register pointers for efficient access during matrix multiplication.
			register double *A_ptr = A + i * N;
			register double *Bt_ptr = Bt + j * N; // Accesses column j of B (row j of Bt)

			for(k = 0; k < N; k++) {
				// Inline: Applies the upper triangular condition for matrix A.
				result += (i <= k) ? (*(A_ptr + k) * *(Bt_ptr + k)) : 0;
			}

			*D_ptr = result;
			D_ptr++;
		}
	}

	/**
	 * Block Logic: Computes the intermediate matrix E, where E = D * B^T.
	 * Utilizes register variables and manual pointer arithmetic for performance.
	 * Pre-condition: Matrices D and B are N x N.
	 * Invariant: E[i][j] contains the sum of D[i][k] * B[j][k].
	 */
	for(i = 0; i < N; i++) {
		register double *E_ptr = E + i * N; 

		for(j = 0; j < N; j++) {

			register double result = 0;

			// Inline: Uses register pointers for efficient access during matrix multiplication.
			register double *D_ptr = D + i * N;
			register double *B_ptr = B + j * N;

			for(k = 0; k < N; k++) {
				result += *D_ptr * *B_ptr;
				D_ptr++;
				B_ptr++;
			}

			*E_ptr = result;
			E_ptr++;
		}
	}

	/**
	 * Block Logic: Computes the intermediate matrix F, where F = A^T * A.
	 * A^T * A refers to the product where only elements A[k][i] and A[k][j] are considered
	 * based on specific conditions (k >= i or k <= j).
	 * Utilizes register variables and manual pointer arithmetic for performance.
	 * Pre-condition: Matrix At contains the transpose of A.
	 * Invariant: F[i][j] contains the specific sum from A^T * A.
	 */
	for(i = 0; i < N; i++) {
		register double *F_ptr = F + i * N; 

		for(j = 0; j < N; j++) {
			register double result = 0;

			// Inline: Uses register pointers for efficient access during matrix multiplication.
			register double *At_ptr1 = At + i * N; // Corresponds to row i of A^T (column i of A)
			register double *At_ptr2 = At + j * N; // Corresponds to row j of A^T (column j of A)

			for(k = 0; k < N; k++) {
				// Inline: Applies conditional access for the elements of matrix At.
				result += ((k >= i) || (k <= j)) ? (*(At_ptr1 + k) * *(At_ptr2 + k)) : 0;
			}

			*F_ptr = result;
			F_ptr++;
		}
	}

	/**
	 * Block Logic: Computes the final matrix C by element-wise summation of E and F.
	 * Utilizes register pointers for efficient access.
	 * Pre-condition: Matrices E and F contain the results of previous computations.
	 * Invariant: Each element of C is the sum of the corresponding elements in E and F.
	 */
	for(i = 0; i < N * N; i++) {
		register double *C_ptr = C + i;
		register double *E_ptr = E + i;
		register double *F_ptr = F + i;

		*C_ptr = *E_ptr + *F_ptr;
	}

	// Functional Utility: Frees memory allocated for intermediate matrices.
	free(At);
	free(Bt);
	free(D);
	free(E);
	free(F);

	// Functional Utility: Returns the pointer to the result matrix C.
	return C;
}
