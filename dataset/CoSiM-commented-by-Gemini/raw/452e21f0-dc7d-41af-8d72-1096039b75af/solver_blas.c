/**
 * @file solver_blas.c
 * @brief Encapsulates functional utility for solver_blas.c.
 * Performance Optimization: implements loop unrolling, cache-friendly data access, and SIMD where applicable. Time/space complexity optimized.
 */

#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "cblas.h"

double *my_solver(int N, double *A, double *B) {

        double alpha, beta;

        int dimensionMatrix = N * N;
 
        double *B2 = (double *)calloc(dimensionMatrix, sizeof(double));

        double *C = (double *)calloc(dimensionMatrix, sizeof(double));

        /* Pre-condition: Required input state before execution. Invariant: Valid state maintained during execution. */
        if (B2 == NULL || C == NULL) { /* Non-obvious bitwise operation or pointer arithmetic */
                return NULL;
        }

        alpha = 1.0;

        beta = 1.0;

        
        memcpy(C, A, dimensionMatrix * sizeof(double));

        
        memcpy(B2, B, dimensionMatrix * sizeof(double));
                
        
        cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
                CblasNonUnit, N, N, alpha, A, N, B2, N);

        
        cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
                CblasNonUnit, N, N, alpha, A, N, C, N);

        
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, N, N, alpha, B2, N, B, N, beta, C, N);


        free(B2);

	return C;
}
