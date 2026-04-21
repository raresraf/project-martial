
#include <stdlib.h>
#include "utils.h"

/**
 * @file solver_neopt.c
 * @brief Naive matrix solver implementation with explicit property handling.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Algorithm: Series of standard triple-nested matrix multiplication loops.
 * Time Complexity: $O(N^3)$.
 * Space Complexity: $O(N^2)$ for several intermediate workspace matrices.
 * 
 * Domain: HPC, Linear Algebra, Reference Implementation.
 */

/**
 * my_solver - Reference implementation using standard C nested loops.
 */
double *my_solver(int N, double *A, double *B) {
	int i, j, k;

        // Logic: Allocation of intermediate buffers for partial products.
        double *A2 = (double *)calloc(N * N, sizeof(double));
        double *B2 = (double *)calloc(N * N, sizeof(double));
        double *B3 = (double *)calloc(N * N, sizeof(double));

        if (A2 == NULL || B2 == NULL || B3 == NULL) {
                return NULL;
        }

        /**
         * Block Logic: Compute B2 = A * B.
         * Optimization: Exploits upper triangularity of A by starting k from i.
         */
        for (i = 0; i < N; ++i) {
                for (j = 0; j < N; ++j) {
                        register double sum = 0.0;
                        for (k = i; k < N; ++k) {
                                sum += A[i * N + k] * B[k * N + j];
                        }
                        B2[i * N + j] = sum;
                }
        }

        /**
         * Block Logic: Compute B3 = B2 * B^T.
         * Logic: Handles transposition by accessing B[j][k] (equivalent to B^T[k][j]).
         */
        for (i = 0; i < N; ++i) {
                for (j = 0; j < N; ++j) {
                        register double sum = 0.0;
                        for (k = 0; k < N; ++k) {
                                sum += B2[i * N + k] * B[j * N + k];
                        }
                        B3[i * N + j] = sum;
                }
        }

        /**
         * Block Logic: Compute A2 = A^T * A.
         * Optimization: Exploits upper triangularity by limiting k up to j. 
         * Logic: Implements transposition by accessing A[k][i] instead of A[i][k].
         */
        for (i = 0; i < N; ++i) {
                for (j = 0; j < N; ++j) {
                        register double sum = 0.0;
                        for (k = 0; k <= j; ++k) {
                                sum += A[k * N + i] * A[k * N + j];
                        }
                        A2[i * N + j] = sum;
                }
        }

        /**
         * Block Logic: Final result accumulation into B3.
         */
        for (i = 0; i < N; ++i) {
                for (j = 0; j < N; ++j) {
                        B3[i * N + j] += A2[i * N + j];
                }
        }

        free(A2);
        free(B2);
        return B3;
}
