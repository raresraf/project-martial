/**
 * @raw/842a6f4e-158e-4c42-b844-5d5699aef0e9/solver_opt.c
 * @brief Optimized dense matrix multiplication solver.
 * Algorithm: Matrix multiplication with loop unrolling, pointer arithmetic, and cache-friendly data access patterns (matrix transposition).
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ for intermediate matrices and transposed copies to maximize spatial locality.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	int i, j, k;

	/**
	 * Functional Utility: Allocates necessary structures including memory for 
	 * transposed versions of A and B to facilitate sequential memory access during multiplication.
	 */
	double *C = (double *)calloc(N * N, sizeof(double));
	if (C == NULL) return NULL;

	double *result1 = (double *)calloc(N * N, sizeof(double));
	if (result1 == NULL) return NULL;

	double *result2 = (double *)calloc(N * N, sizeof(double));
	if (result2 == NULL) return NULL;

	double *At = (double *)calloc(N * N, sizeof(double));
	if (At == NULL) return NULL;

	double *Bt = (double *)calloc(N * N, sizeof(double));
	if (Bt == NULL) return NULL;

	/**
	 * Block Logic: Creates transposed copies of matrices A and B.
	 * Invariant: Transposing matrices converts column-major traversals into cache-friendly 
	 * row-major traversals during dot products.
	 */
	for (i = 0; i < N; i++) {
		// Inline: Use pointer arithmetic and registers for fast indexing.
		register double *At_1 = &At[i * N];
		register double *Bt_1 = &Bt[i * N];
		for (j = 0; j < N; j++) {
			At_1[j] = A[j * N + i];
			Bt_1[j] = B[j * N + i];
		}
	}

	/**
	 * Block Logic: Computes intermediate matrix `result1` = A * B.
	 * Invariant: Exploits the upper triangular nature of A to skip unnecessary iterations (i > k).
	 * Loop order is reorganized to i-k-j to enhance spatial locality for matrix B access.
	 */
	for (i = 0; i < N; i++) {
		register double *A_1 = &A[i * N];
		register double *result1_1 = &result1[i * N];
		for (k = 0; k < N; k++) {
			if (i > k) {
				continue;
			}
			register double *B_1 = &B[k * N];
			register double A_2 = A_1[k];
			for (j = 0; j < N; j++) {
				result1_1[j] += A_2 * B_1[j];
			}
		}
	}

	/**
	 * Block Logic: Computes `result2` = `result1` * B^T (effectively `result1` * B).
	 * Invariant: Uses the transposed matrix `Bt` to ensure sequential access, 
	 * significantly reducing L1/L2 cache misses.
	 */
	for (i = 0; i < N; i++) {
		register double *result1_1 = &result1[i * N];
		register double *result2_1 = &result2[i * N];
		for (k = 0; k < N; k++) {
			register double result1_2 = result1_1[k];
			register double *Bt_1 = &Bt[k * N];
			for (j = 0; j < N; j++) {
				result2_1[j] += result1_2 * Bt_1[j];
			}
		}
	}

	/**
	 * Block Logic: Computes A^T * A and stores it into C.
	 * Invariant: Uses transposed matrix `At` and original matrix `A` sequentially, 
	 * exploiting upper triangular properties of A to prune iterations (k > i).
	 */
	for (i = 0; i < N; i++) {
		register double *C_1 = &C[i * N];
		register double *At_1 = &At[i * N];
		for (k = 0; k < N; k++) {
			if (k > i) {
				break;
			}
			register double *A_1 = &A[k * N];
			register double At_2 = At_1[k];
			for (j = 0; j < N; j++) {
				C_1[j] += At_2 * A_1[j];
			}
		}
	}

	/**
	 * Block Logic: Accumulates `result2` into the final matrix C.
	 * Invariant: Final sequential addition to complete the matrix computation target.
	 */
	for (i = 0; i < N; i++) {
		register double *C_1 = &C[i * N];
		register double *result2_1 = &result2[i * N];
		for (j = 0; j < N; j++) {
			C_1[j] += result2_1[j];
		}
	}

	free(At);
	free(Bt);
	free(result1);
	free(result2);

	return C;
}
