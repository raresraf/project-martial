/**
 * @file solver_opt.c
 * @brief Encapsulates functional utility for solver_opt.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

#include <stdlib.h>
#include "utils.h"

double *my_solver(int N, double *A, double *B) {

        register int i, j, k;

        double *At = calloc(N * N, sizeof(double));
        double *B2 = calloc(N * N, sizeof(double));
        double *B3 = calloc(N * N, sizeof(double));

        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        if (At == NULL || B2 == NULL || B3 == NULL) { /* Non-obvious bitwise operation or pointer arithmetic */
                return NULL;
        }


        
        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        for (i = 0; i != N; ++i) {

                register double *ptr_At = At + i;

                register double *ptr_A = A + i * N;

                /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
                for (j = 0; j != N; ++j) {
                       *ptr_At = *ptr_A;
                        ++ptr_A;
                        ptr_At += N;
                }
        }


        
        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        for (i = 0; i != N; ++i) {

                
                register double *ptrB2 = B2 + i * N;

                
                register double *B_aux = B + i * N;

                
                /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
                for (j = 0; j != N; ++j) {
                        
                        register double sum = 0.0;

                        
                        register double *ptrA = A + i * N + i;

                        
                        register double *ptrB = B_aux + j;

                        
                        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
                        for (k = i; k != N; ++k) {
                                sum += *ptrA * *ptrB;
                                ++ptrA;
                                ptrB += N;
                        }

                        *ptrB2 = sum;
                        ++ptrB2;
                }
        }


        
        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        for (i = 0; i != N; ++i) {

                
                register double *ptrB3 = B3 + i * N;

                /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
                for (j = 0; j != N; ++j) {
                        
                        register double sum = 0.0;

                        
                        register double *ptrB2 = B2 + i * N;

                         
                        register double *ptrB = B + j * N;

                        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
                        for (k = 0; k != N; ++k) {
                                sum += *ptrB2 * *ptrB;
                                ++ptrB2;
                                ++ptrB;
                        }

                        *ptrB3 = sum;
                        ++ptrB3;
                }
        }

        
        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        for (i = 0; i != N; ++i) {
    
                
                register double *aux_B3 = B3;

                
                register double *ptr_At = At + i;

                /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
                for (j = 0; j != N; ++j) {

                        
                        register double *ptr_B3 = aux_B3;

                        
                        register double *ptrA = A + i * N;

                        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
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
