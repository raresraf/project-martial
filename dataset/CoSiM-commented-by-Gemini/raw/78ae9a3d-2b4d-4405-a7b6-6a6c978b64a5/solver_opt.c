/**
 * @file solver_opt.c
 * @brief An optimized version of the matrix operation sequence.
 * @details This file provides a `my_solver` function that implements the same logic as
 * `solver_neopt.c` but attempts to optimize performance through several techniques:
 * 1.  **Explicit Transposition**: Matrices A and B are explicitly transposed at the beginning.
 *     This allows for sequential memory access (better cache utilization) in subsequent loops
 *     that require column-wise traversal.
 * 2.  **Register Variables**: The `register` keyword is used to suggest to the compiler that
 *     certain variables (loop counters, pointers) should be stored in CPU registers for faster access.
 * 3.  **Pointer Arithmetic**: Instead of array indexing `M[i * N + j]`, the code extensively
 *     uses pointers to iterate through the matrices, which can sometimes lead to more efficient
 *     compiled code by reducing address calculation overhead.
 */
#include "utils.h"


double* my_solver(int N, double *A, double *B) {
	printf("OPT SOLVER\n");
	// Use of `register` is a hint to the compiler for optimization.
	register int i, j, k;
	double *aux1 = (double*) calloc(N * N, sizeof(double));
	double *aux3 = (double*) calloc(N * N, sizeof(double));
	// Allocate memory for explicit transposes of A and B.
	double *A_tr = (double*) calloc(N * N, sizeof(double));
	double *B_tr = (double*) calloc(N * N, sizeof(double));

	/**
	 * @brief Block Logic: Explicitly compute the transpose of matrices A and B.
	 * This pre-computation step allows for linear, sequential access patterns in later loops
	 * where columns of the original matrices are needed, improving cache performance.
	 */
	for (i = 0; i < N; ++i) {
		register int count1 = N * i;
		register double *pa = &A[count1];
		register double *pb = &B[count1];
		register double *pa_tr = &A_tr[i];
		register double *pb_tr = &B_tr[i];

		for (j = 0; j < N; ++j) {
			*pa_tr = *pa;
			*pb_tr = *pb;
			pa++;
			pb++;
			pa_tr += N;
			pb_tr += N;
		}
	}

	/**
	 * @brief First pass: concurrent computation of two intermediate matrices.
	 * This section is the optimized equivalent of the first loop in `solver_neopt.c`.
	 * It uses pointers to traverse the matrices.
	 */
	for (i = 0; i < N; ++i) {
		register int count1 = N * i;
		// Pointers to the beginning of the i-th rows of A and its transpose.
		register double *orig_pa = &A[count1];
		register double *orig_pa_tr = &A_tr[count1];
		for (j = 0; j < N; ++j) {
			register int count2 = N * i + j;
			register double *paux1 = &aux1[count2];
			register double *paux3 = &aux3[count2];

			register double *pa = orig_pa;
			register double *pa_tr = orig_pa_tr;
  			register double *pb = &B[j];
			register double *pa2 = &A[j];
			
			for (k = 0; k < N; ++k) {
				// Operation: aux1 = A_upper * B
				if (i <= k) {
					*paux1 += *pa * *pb;
				}
				// Operation: aux3 = A_lower^T * A_lower
				if (k <= i && k <= j) {
					// Here, pa_tr and pa2 are used to simulate the A_lower^T * A_lower multiplication
					// from the non-optimized version.
					*paux3 += *pa_tr * *pa2;
				}
				pa++;
				pa_tr++;
				pb += N;
				pa2 += N;
			}
		}
	}

	/**
	 * @brief Second pass: Update `aux3` with the product of `aux1` and `B^T`.
	 * This is the optimized version of the second main loop in `solver_neopt.c`.
	 * It benefits from the pre-computed transpose of B (`B_tr`) to ensure sequential memory access.
	 */
	for (i = 0; i < N; ++i) {
		register int count1 = N * i;
		register double *orig_pa = &aux1[count1];
		for (j = 0; j < N; ++j) {
			register int count2 = N * i + j;
			register double *paux3 = &aux3[count2];
			register double *pa = orig_pa;
			// Accessing the pre-transposed B matrix.
  			register double *pb = &B_tr[j];
			for (k = 0; k < N; ++k) {
				*paux3 += *pa * *pb;
				pa++;
				pb += N;
			}
		}
	}
	// Free all dynamically allocated memory.
	free(aux1);
	free(A_tr);
	free(B_tr);
	// Return the final result.
	return aux3;
}