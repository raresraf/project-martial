/**
 * @raw/96d93c00-0061-439f-84b0-7b394d9b51f1/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing RES = (A * B) * B^T + A^T * A.
 * Algorithm: Standard nested loops evaluating separate mathematical terms. Explicit transpositions used.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ for storing multiple matrices including explicitly transposed copies.
 */

#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	double *A_star_B;
	double *B_tr;
	double *OP1;
	double *A_tr;
	double *OP2;
	double *RES;
	int numMatrixElems = N * N;

	/**
	 * Functional Utility: Initializes intermediate computation buffers matching the matrix dimensions.
	 */
	A_star_B = calloc(numMatrixElems, sizeof(*A_star_B));
	B_tr = calloc(numMatrixElems, sizeof(*B_tr));
	OP1 = calloc(numMatrixElems, sizeof(*OP1));
	A_tr = calloc(numMatrixElems, sizeof(*A_tr));
	OP2 = calloc(numMatrixElems, sizeof(*OP2));
	RES = calloc(numMatrixElems, sizeof(*RES));

	/**
	 * Block Logic: Computes the partial matrix multiplication A_star_B = A * B.
	 * Invariant: Matrix A is constrained structurally as an upper triangular matrix, pruning checks via `k >= i`.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				if (k >= i) {
					A_star_B[i * N + j] += A[i * N + k] * B[k * N + j];
				}
			}
		}
	}

	/**
	 * Block Logic: Explicitly computes the transposition of matrix B generating B_tr = B^T.
	 * Invariant: Caches B^T spatially to allow row-major iterations.
	 */
	for (int i = 0; i < N; i++) {
    	for (int j = 0; j < N; j++) {
    		B_tr[j * N + i] = B[i * N + j];
    	}
	}

	/**
	 * Block Logic: Computes OP1 = A_star_B * B_tr.
	 * Resolves to the expression OP1 = (A * B) * B^T.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				OP1[i * N + j] += A_star_B[i * N + k] * B_tr[k * N + j];
			}
		}
	}

	/**
	 * Block Logic: Derives A_tr = A^T dynamically.
	 */
	for (int i = 0; i < N; i++) {
    	for (int j = 0; j < N; j++) {
    		A_tr[j * N + i] = A[i * N + j];
    	}
	}

	/**
	 * Block Logic: Computes OP2 = A_tr * A.
	 * Invariant: Exploits the combined upper-lower triangular traits preventing calculation over `j < k`.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				if (j >= k) {
					OP2[i * N + j] += A_tr[i * N + k] * A[k * N + j];
				}
			}
		}
	}

	/**
	 * Block Logic: Final synthesis step producing RES = OP1 + OP2.
	 * Invariant: Evaluates the formula completely via element-wise addition.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			RES[i * N + j] = OP1[i * N + j] + OP2[i * N + j];
		}
	}

	free(A_star_B);
	free(B_tr);
	free(OP1);
	free(A_tr);
	free(OP2);

	return RES;
}
