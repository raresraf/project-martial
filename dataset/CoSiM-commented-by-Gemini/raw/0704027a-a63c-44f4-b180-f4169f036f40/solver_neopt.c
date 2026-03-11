
/**
 * @file solver_neopt.c
 * @brief A non-optimized, straightforward implementation of a matrix solver.
 *
 * This file provides a basic implementation of the matrix operations required by the solver,
 * using simple nested loops. This version serves as a baseline for correctness and for
 * performance comparison against optimized versions (e.g., using BLAS or other optimizations).
 */

#include "utils.h"
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

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
 * @brief Computes the matrix product C = A * A^T.
 * @param N The size of the square matrix A.
 * @param A A pointer to the input matrix.
 * @return A pointer to the resulting matrix C.
 */
double* simple_multiplication(int N, double* A)
{
    double* result = (double*)calloc(N * N, sizeof(double));
    CHECK_ALLOC(result);

    int i, j, k;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < N; k++)
            {
                result[i * N + j] += A[i * N + k] * A[j * N + k];
            }
        }
    }

    return result;
}

/**
 * @brief Computes the matrix product C = A * B, where A is an upper triangular matrix.
 * @param N The size of the square matrices.
 * @param A A pointer to the upper triangular input matrix.
 * @param B A pointer to the second input matrix.
 * @return A pointer to the resulting matrix C.
 */
double* triangular_multiplication(int N, double* A, double* B)
{
    double* result = (double*)calloc(N * N, sizeof(double));
    CHECK_ALLOC(result);

    int i, j, k;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = i; k < N; k++)
            {
                result[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

    return result;
}

/**
 * @brief Computes the matrix product C = A^T * A, where A is an upper triangular matrix.
 * @param N The size of the square matrix A.
 * @param A A pointer to the upper triangular input matrix.
 * @return A pointer to the resulting matrix C.
 */
double* double_triangular_multiplication(int N, double* A)
{
    double* result = (double*)calloc(N * N, sizeof(double));
    CHECK_ALLOC(result);

    int i, j, k;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k <= MIN(i, j); k++)
            {
                result[i * N + j] += A[k * N + i] * A[k * N + j];
            }
        }
    }

    return result;
}

/**
 * @brief Computes the matrix sum C = A + B.
 * @param N The size of the square matrices.
 * @param A A pointer to the first input matrix.
 * @param B A pointer to the second input matrix.
 * @return A pointer to the resulting matrix C.
 */
double* matrix_addition(int N, double* A, double* B)
{
    double* result = (double*)calloc(N * N, sizeof(double));
    CHECK_ALLOC(result);

    int i, j;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            result[i * N + j] = A[i * N + j] + B[i * N + j];
        }
    }

    return result;
}

/**
 * @brief Solves the matrix equation by calling the elementary matrix operation functions.
 *
 * This function implements the same logic as the BLAS version, but calls the non-optimized
 * helper functions defined in this file. The sequence of operations is:
 * 1. R1 = B * B^T
 * 2. R2 = A * R1 (where A is upper triangular)
 * 3. R3 = A^T * A (where A is upper triangular)
 * 4. result = R2 + R3
 *
 * @param N The size of the square matrices.
 * @param A A pointer to the first input matrix.
 * @param B A pointer to the second input matrix.
 * @return A pointer to the resulting matrix.
 */
double* my_solver(int N, double *A, double* B) {
	double* R1 = simple_multiplication(N, B);
    double* R2 = triangular_multiplication(N, A, R1);
    double* R3 = double_triangular_multiplication(N, A);
    double* result = matrix_addition(N, R2, R3);

    free(R1);
    free(R2);
    free(R3);

	return result;
}
