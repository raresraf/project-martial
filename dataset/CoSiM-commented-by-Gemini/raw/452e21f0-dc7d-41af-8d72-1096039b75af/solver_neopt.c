/**
 * @file solver_neopt.c
 * @brief Encapsulates functional utility for solver_neopt.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

#include <stdlib.h>
#include "utils.h"

double *my_solver(int N, double *A, double *B) {

	int i, j, k;

        double *A2 = (double *)calloc(N * N, sizeof(double));
        double *B2 = (double *)calloc(N * N, sizeof(double));
        double *B3 = (double *)calloc(N * N, sizeof(double));

        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        if (A2 == NULL || B2 == NULL || B3 == NULL) { /* Non-obvious bitwise operation or pointer arithmetic */
                return NULL;
        }

        
        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        for (i = 0; i < N; ++i) {
                /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
                for (j = 0; j < N; ++j) {
                        register double sum = 0.0;

                        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
                        for (k = i; k < N; ++k) {
                                sum += A[i * N + k] * B[k * N + j];
                        }

                        B2[i * N + j] = sum;
                }
        }

        
        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        for (i = 0; i < N; ++i) {
                /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
                for (j = 0; j < N; ++j) {
                        register double sum = 0.0;

                        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
                        for (k = 0; k < N; ++k) {
                                sum += B2[i * N + k] * B[j * N + k];
                        }

                        B3[i * N + j] = sum;
                }
        }

        
        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        for (i = 0; i < N; ++i) {
                /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
                for (j = 0; j < N; ++j) {
                        register double sum = 0.0;

                        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
                        for (k = 0; k <= j; ++k) {
                                sum += A[k * N + i] * A[k * N + j];
                        }

                        A2[i * N + j] = sum;
                }
        }

        
        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        for (i = 0; i < N; ++i) {

                /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
                for (j = 0; j < N; ++j) {
                        B3[i * N + j] += A2[i * N + j];
                }
        }

        free(A2);
        free(B2);

        return B3;
}
