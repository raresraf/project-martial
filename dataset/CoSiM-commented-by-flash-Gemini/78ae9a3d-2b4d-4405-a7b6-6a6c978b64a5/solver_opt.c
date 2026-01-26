
/**

 * @file solver_opt.c

 * @brief Implements an optimized matrix solver using manual optimizations for cache efficiency and register usage.

 *        This solver aims to improve performance over a naive implementation by reordering operations

 *        and using pre-transposed matrices.

 * Algorithm: Matrix operations are performed using nested loops with explicit pointer arithmetic

 *            and register hints, along with pre-computation of matrix transposes (A_tr, B_tr)

 *            to enhance data locality and reduce cache misses. The core computation involves

 *            multiple matrix multiplications with conditional accumulations.

 * Time Complexity: O(N^3) due to matrix multiplications, but with improved constant factors

 *                  compared to naive implementations due to optimization techniques.

 * Space Complexity: O(N^2) for auxiliary matrices (`aux1`, `aux3`) and transposed copies (`A_tr`, `B_tr`).

 */



#include "utils.h"



/**

 * @brief Optimized matrix solver function.

 *

 * This function performs a series of optimized matrix operations on input matrices A and B.

 * It uses auxiliary matrices and pre-transposed copies of input matrices to enhance performance.

 * The function's calculations are complex due to conditional logic and manual optimizations.

 *

 * @param N The dimension of the square matrices A and B.

 * @param A Pointer to the N x N matrix A.

 * @param B Pointer to the N x N matrix B.

 * @return A pointer to a newly allocated N x N matrix (`aux3`) containing the final result.

 */

double* my_solver(int N, double *A, double *B) {

	printf("OPT SOLVER\n");

	// i, j, k: Loop counters. Declared as register for potential CPU register allocation.

	register int i, j, k;

	// aux1: First auxiliary N x N matrix for intermediate results, initialized to zeros.

	double *aux1 = (double*) calloc(N * N, sizeof(double));

	// aux3: Second auxiliary N x N matrix for final results, initialized to zeros.

	double *aux3 = (double*) calloc(N * N, sizeof(double));

	// A_tr: Transposed copy of matrix A for optimized access patterns.

	double *A_tr = (double*) calloc(N * N, sizeof(double));

	// B_tr: Transposed copy of matrix B for optimized access patterns.

	double *B_tr = (double*) calloc(N * N, sizeof(double));



	// Block Logic: Pre-compute transposes of matrices A and B.

	// This loop transposes A into A_tr and B into B_tr to facilitate cache-friendly

	// column-major access patterns in subsequent matrix multiplications.

	for (i = 0; i < N; ++i) {

		register int count1 = N * i; // Optimized row start index for A and B.

		register double *pa = &A[count1];     // Pointer to current row of A.

		register double *pb = &B[count1];     // Pointer to current row of B.

		register double *pa_tr = &A_tr[i];    // Pointer to current column of A_tr.

		register double *pb_tr = &B_tr[i];    // Pointer to current column of B_tr.



		for (j = 0; j < N; ++j) {

			*pa_tr = *pa; // Copy A[i][j] to A_tr[j][i].

			*pb_tr = *pb; // Copy B[i][j] to B_tr[j][i].

			pa++;         // Move to next element in current row of A.

			pb++;         // Move to next element in current row of B.

			pa_tr += N;   // Move to next element in current column of A_tr.

			pb_tr += N;   // Move to next element in current column of B_tr.

		}

	}



	// Block Logic: First set of complex conditional matrix accumulations.

	// This block calculates intermediate results into `aux1` and `aux3` based on

	// specific index relationships (i <= k) and (k <= i && k <= j).

	// This likely represents a partial matrix multiplication or a specialized transformation,

	// leveraging pre-transposed matrices for potentially better performance.

	for (i = 0; i < N; ++i) {

		register int count1 = N * i;           // Optimized row start index for A and A_tr.

		register double *orig_pa = &A[count1];     // Pointer to current row of A.

		register double *orig_pa_tr = &A_tr[count1]; // Pointer to current row of A_tr.

		for (j = 0; j < N; ++j) {

			register int count2 = N * i + j;       // Optimized element index for aux1 and aux3.

			register double *paux1 = &aux1[count2]; // Pointer to current element of aux1.

			register double *paux3 = &aux3[count2]; // Pointer to current element of aux3.



			register double *pa = orig_pa;     // Current pointer for elements of A.

			register double *pa_tr = orig_pa_tr; // Current pointer for elements of A_tr.

  			register double *pb = &B[j];       // Current pointer for column 'j' of B.

			register double *pa2 = &A[j];      // Current pointer for column 'j' of A.

			

			for (k = 0; k < N; ++k) {

				if (i <= k) {

					// Accumulates sum for aux1[i][j] using elements from A[i][k] and B[k][j].

					*paux1 += *pa * *pb;

				}

				if (k <= i && k <= j) {

					// Accumulates sum for aux3[i][j] using elements from A_tr[i][k] and A[k][j].

					*paux3 += *pa_tr * *pa2;

				}

				pa++;     // Move to next element in current row of A.

				pa_tr++;  // Move to next element in current row of A_tr.

				pb += N;  // Move to next element in current column of B.

				pa2 += N; // Move to next element in current column of A.

			}

		}

	}



	// Block Logic: Second matrix multiplication involving aux1 and B_tr.

	// This block performs a matrix multiplication of `aux1` with `B_tr` (transpose of B)

	// and accumulates the result into `aux3`.

	for (i = 0; i < N; ++i) {

		register int count1 = N * i;           // Optimized row start index for aux1.

		register double *orig_pa = &aux1[count1]; // Pointer to current row of aux1.

		for (j = 0; j < N; ++j) {

			register int count2 = N * i + j;       // Optimized element index for aux3.

			register double *paux3 = &aux3[count2]; // Pointer to current element of aux3.

			register double *pa = orig_pa;     // Current pointer for elements of aux1.

  			register double *pb = &B_tr[j];    // Current pointer for column 'j' of B_tr.

			for (k = 0; k < N; ++k) {

				// Accumulates sum for aux3[i][j] using elements from aux1[i][k] and B_tr[k][j].

				*paux3 += *pa * *pb;

				pa++;     // Move to next element in current row of aux1.

				pb += N;  // Move to next element in current column of B_tr.

			}

		}

	}

	free(aux1); // Free memory allocated for aux1.

	free(A_tr); // Free memory allocated for A_tr.

	free(B_tr); // Free memory allocated for B_tr.

	return aux3; // Return the final computed matrix.

}
