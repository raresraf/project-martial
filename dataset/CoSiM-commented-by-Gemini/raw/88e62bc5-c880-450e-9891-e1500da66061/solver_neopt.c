/**
 * @raw/88e62bc5-c880-450e-9891-e1500da66061/solver_neopt.c
 * @brief Unoptimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Standard nested loops.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ for intermediate accumulation matrices.
 */
#include "utils.h"

/**
 * Inline: Macro for calculating the inclusive bounds dynamically during execution.
 */
#define min(x, y) (((x) < (y)) ? (x) : (y))

double* my_solver(int N, double *A, double* B) {

	/**
	 * Functional Utility: Allocates memory for final result C and intermediate term.
	 */
	double *C = (double *)malloc(N * N * sizeof(double));
	double *term = (double *)malloc(N * N * sizeof(double));

	/**
	 * Block Logic: Computes term = A * B.
	 * Invariant: Treats A as an upper triangular matrix (k >= i).
	 */
	for (int i = 0;i < N;i++) {
		for (int j = 0;j < N;j++) {
			term[i * N + j] = 0.0;
			for (int k = i;k < N;k++) {
				term[i * N + j] += (double)(A[i * N + k] * B[k * N + j]);
			}
		}
	}

	/**
	 * Block Logic: Computes C = term * B^T.
	 * Invariant: B is accessed with indices [j * N + k] implicitly evaluating its transposition.
	 */
	for (int i = 0;i < N;i++) {
		for (int j = 0;j < N;j++) {
			C[i * N + j] = 0.0;
			for (int k = 0;k < N;k++) {
				C[i * N + j] += (double)(term[i * N + k] * B[j * N + k]);
			}
		}
	}

	/**
	 * Block Logic: Overwrites term to compute A^T * A.
	 * Invariant: Limits iterations relying on upper triangular structure (`k <= min(i, j)`).
	 */
	for (int i = 0;i < N;i++) {
                for (int j = 0;j < N;j++) {
                        term[i * N + j] = 0.0;
                        for (int k = 0;k <= min(i, j);k++) {
                                term[i * N + j] += (double)(A[k * N + i] * A[k * N + j]);
                        }
                }
        }

	/**
	 * Block Logic: Final summation C += term.
	 * Invariant: Consolidates C = C + A^T * A.
	 */
	for (int i = 0;i < N;i++) {
                for (int j = 0;j < N;j++) {
                	C[i * N + j] += (double)term[i * N + j];
                }
        }

	free(term);

	return C;
}
