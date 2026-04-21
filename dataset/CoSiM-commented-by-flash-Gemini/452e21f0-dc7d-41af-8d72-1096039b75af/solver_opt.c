
#include <stdlib.h>
#include "utils.h"

/**
 * @file solver_opt.c
 * @brief Manually optimized matrix solver implementation using pointer arithmetic and pre-transposition.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs several manual C tuning strategies:
 * 1. Explicit Transposition: Pre-computes A^T to transform strided column access 
 *    into sequential row-major increments, significantly improving spatial 
 *    locality and cache throughput.
 * 2. Pointer Arithmetic: Replaces array indexing with direct pointer 
 *    dereferences and increments (*ptr++).
 * 3. Register Caching: Caches frequently used row base pointers and 
 *    intermediate sums in CPU registers.
 * 4. Cache-aware multiplication: Uses a row-major dot product for the (B2 * B^T) 
 *    term to avoid strided access.
 * 
 * Domain: HPC, Linear Algebra, Manual Performance Tuning.
 */

/**
 * my_solver - Optimized matrix solver with cache-friendly traversal and register hints.
 */
double *my_solver(int N, double *A, double *B) {
        register int i, j, k;

        // Logic: Workspace allocation for partial products and the transposed copy.
        double *At = calloc(N * N, sizeof(double));
        double *B2 = calloc(N * N, sizeof(double));
        double *B3 = calloc(N * N, sizeof(double));

        if (At == NULL || B2 == NULL || B3 == NULL) {
                return NULL;
        }

        /**
         * Block Logic: Pre-compute At = A^T.
         * Optimization: Sequential-to-Strided copy.
         * Logic: While the input A is read sequentially, the output At is written 
         * with stride N to perform the transposition in a single pass.
         */
        for (i = 0; i != N; ++i) {
                register double *ptr_At = At + i;
                register double *ptr_A = A + i * N;
                for (j = 0; j != N; ++j) {
                       *ptr_At = *ptr_A;
                        ++ptr_A;
                        ptr_At += N;
                }
        }

        /**
         * Block Logic: Compute B2 = A * B.
         * Optimization: Triangular pointer offset.
         * Logic: Exploys upper triangularity of A by starting k from i. 
         * Uses pointer-based dot product between A[i] and B[][j].
         */
        for (i = 0; i != N; ++i) {
                register double *ptrB2 = B2 + i * N;
                register double *B_aux = B + i * N;
                for (j = 0; j != N; ++j) {
                        register double sum = 0.0;
                        register double *ptrA = A + i * N + i;
                        register double *ptrB = B_aux + j;
                        for (k = i; k != N; ++k) {
                                sum += *ptrA * *ptrB;
                                ++ptrA;
                                ptrB += N;
                        }
                        *ptrB2 = sum;
                        ++ptrB2;
                }
        }

        /**
         * Block Logic: Compute B3 = B2 * B^T.
         * Optimization: Row-major row product.
         * Logic: Multiplies B2's rows by B's rows (as columns of B^T). 
         * This achieves dual sequential memory access, maximizing cache throughput.
         */
        for (i = 0; i != N; ++i) {
                register double *ptrB3 = B3 + i * N;
                for (j = 0; j != N; ++j) {
                        register double sum = 0.0;
                        register double *ptrB2 = B2 + i * N;
                        register double *ptrB = B + j * N;
                        for (k = 0; k != N; ++k) {
                                sum += *ptrB2 * *ptrB;
                                ++ptrB2;
                                ++ptrB;
                        }
                        *ptrB3 = sum;
                        ++ptrB3;
                }
        }

        /**
         * Block Logic: Accumulate B3 += A^T * A.
         * Optimization: Pre-transposed access.
         * Logic: Uses the pre-computed At to multiply columns of A (as rows of A^T) 
         * by rows of A.
         */
        for (i = 0; i != N; ++i) {
                register double *aux_B3 = B3;
                register double *ptr_At = At + i;
                for (j = 0; j != N; ++j) {
                        register double *ptr_B3 = aux_B3;
                        register double *ptrA = A + i * N;
                        for (k = 0; k != N; ++k) {
                            *ptr_B3 += *ptr_At * *ptrA;
                            ++ptr_B3;
                            ++ptrA;
                        }
                        ptr_At += N;
                        aux_B3 += N;
                }
        }
        
        free(At);
        free(B2);
	return B3;
}
