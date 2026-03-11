
/**
 * @file solver_blas.c
 * @brief A matrix solver implementation using BLAS routines.
 *
 * This file contains a function that uses the Basic Linear Algebra Subprograms (BLAS)
 * library to perform a series of matrix operations. It appears to solve a matrix
 * equation involving triangular and general matrix multiplications.
 */

#include "utils.h"
#include <cblas.h>
#include <string.h>

/**
 * @def CHECK_ALLOC(x)
 * @brief A macro to check if a memory allocation was successful.
 *        If not, it prints an error and exits the program.
 */
#define CHECK_ALLOC(x) 
            if (x == NULL) 
            { 
                perror("Allocation failed
"); 
                exit(-1); 
            }


/**
 * @brief Solves a matrix equation using a sequence of BLAS operations.
 *
 * This function performs a series of matrix multiplications to solve a matrix equation.
 * The exact equation being solved is not immediately clear from the code, but it involves
 * the input matrices A and B, and their transposes. The operations are:
 * 1. R1 = B * B^T
 * 2. R1 = A * R1 (where A is upper triangular)
 * 3. A = A^T * A (where A is upper triangular)
 * 4. A = R1 * I + A = R1 + A
 * 5. R1 = A
 *
 * @param N The size of the square matrices.
 * @param A A pointer to the first input matrix (N x N). This matrix is modified in place.
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix. The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double* B) {
    // Allocate memory for the result matrix R1
    double* R1 = (double*)calloc(N * N, sizeof(double));
    CHECK_ALLOC(R1);

    // R1 = B * B^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 0, R1, N);

    // R1 = A * R1, where A is an upper triangular matrix
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, R1, N);

    // A = A^T * A, where A is an upper triangular matrix
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, A, N);

    // Allocate memory for an identity matrix I
    double* I = (double*)calloc(N * N, sizeof(double));
    CHECK_ALLOC(I);

    // Initialize I as an identity matrix
    int i;
    for (i = 0; i < N * N; i += N + 1)
    {
        I[i] = 1;
    }

    // A = R1 * I + A = R1 + A
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, R1, N, I, N, 1, A, N);

    // Copy the final result from A to R1
    memcpy(R1, A, N * N * sizeof(double));

    // Free the identity matrix
    free(I);

    return R1;
}
