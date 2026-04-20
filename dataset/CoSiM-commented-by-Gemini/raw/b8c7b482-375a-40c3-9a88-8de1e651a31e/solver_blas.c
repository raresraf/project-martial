/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include "utils.h"
#include "cblas.h"
#include <string.h>

#define DEBUG 0



/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double *my_solver(int M, double *a, double *b)
{
    printf("BLAS SOLVER\n");

    double *b_bt = malloc(M * M * sizeof(double));
    double *a_at = malloc(M * M * sizeof(double));

    memcpy(a_at, a, M * M * sizeof(double));

    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, M, M, 1.00, b, M, b, M, 0.00, b_bt, M);
    
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, M, M, 1.00, a, M, a_at, M);
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, M, M, 1.00, a, M, b_bt, M, 1.00, a_at, M);

    free(b_bt);

    return a_at;
}
