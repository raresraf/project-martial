
/**
 * @file solver_opt.c
 * @brief An optimized implementation of a matrix solver.
 *
 * This file provides an optimized version of the matrix operations required by the solver.
 * The optimizations include loop tiling (block-based processing) to improve cache locality
 * and the use of register variables for frequently accessed pointers.
 */

#include "utils.h"
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

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

// The block size used for loop tiling.
int block_size = 100;

/**
 * @brief Computes the matrix product C = A * A^T using loop tiling.
 * @param N The size of the square matrix A.
 * @param A A pointer to the input matrix.
 * @return A pointer to the resulting matrix C.
 */
double* simple_multiplication(int N, double* A)
{
    double* result = (double*)calloc(N * N, sizeof(double));
    CHECK_ALLOC(result);

    int i, j, k;
    int bi, bj, bk;
    register int min_i, min_j, min_k;
    register double *orig_pa, *pa, *pb, *pr;

    for (bi = 0; bi < N; bi += block_size)
    {
        for (bj = 0; bj < N; bj += block_size)
        {
            for (bk = 0; bk < N; bk += block_size)
            {
                min_i = MIN(bi + block_size, N);

                for (i = bi; i < min_i; i++)
                {
                    orig_pa = &A[i * N + bk];
                    pr = &result[i * N + bj];
                    min_j = MIN(bj + block_size, N);
                    
                    for (j = bj; j < min_j; j++)
                    {
                        pa = orig_pa;
                        pb = &A[j * N + bk];
                        min_k = MIN(bk + block_size, N);

                        for (k = bk; k < min_k; k++)
                        {
                            *pr += *pa * *pb;
                            pa++;
                            pb++;
                        }
                        pr++;
                    }
                }
            }
        }
    }
    

    return result;
}

/**
 * @brief Computes the matrix product C = A * B, where A is an upper triangular matrix, using loop tiling.
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
    int bi, bj, bk;
    int start;
    register int min_i, min_j, min_k;
    register double *orig_pa, *pa, *pb, *pr;

    for (bi = 0; bi < N; bi += block_size)
    {
        for (bj = 0; bj < N; bj += block_size)
        {
            for (bk = 0; bk < N; bk += block_size)
            {
                min_i = MIN(bi + block_size, N);

                for (i = bi; i < min_i; i++)
                {
                    orig_pa = &A[i * N];
                    pr = &result[i * N + bj];
                    min_j = MIN(bj + block_size, N);
                 
                    for (j = bj; j < min_j; j++)
                    {
                        start = MAX(bk, i);
                        pa = orig_pa + start;
                        pb = &B[start * N + j];
                        min_k = MIN(bk + block_size, N);

                        for (k = start; k < min_k; k++)
                        {
                            *pr += *pa * *pb;
                            pa++;
                            pb += N;
                        }
                        pr++;
                    }
                }
            }
        }
    }

    return result;
}

/**
 * @brief Computes the matrix product C = A^T * A, where A is an upper triangular matrix, using loop tiling.
 * @param N The size of the square matrix A.
 * @param A A pointer to the upper triangular input matrix.
 * @return A pointer to the resulting matrix C.
 */
double* double_triangular_multiplication(int N, double* A)
{
    double* result = (double*)calloc(N * N, sizeof(double));
    CHECK_ALLOC(result);

    int i, j, k;
    int bi, bj, bk;
    register int min_i, min_j, min_k;
    register double *orig_pa, *pa, *pb, *pr;

    for (bi = 0; bi < N; bi += block_size)
    {
        for (bj = 0; bj < N; bj += block_size)
        {
            for (bk = 0; bk < N; bk += block_size)
            {
                min_i = MIN(bi + block_size, N);

                for (i = bi; i < min_i; i++)
                {
                    orig_pa = &A[bk * N + i];
                    pr = &result[i * N + bj];
                    min_j = MIN(bj + block_size, N);
                    
                    for (j = bj; j < min_j; j++)
                    {
                        pa = orig_pa;
                        pb = &A[bk * N + j];
                        min_k = MIN(bk + block_size, MIN(i, j) + 1);

                        for (k = bk; k < min_k; k++)
                        {
                            *pr += *pa * *pb;
                            pa += N;
                            pb += N;
                        }
                        pr++;
                    }
                }
            }
        }
    }

    return result;
}

/**
 * @brief Computes the matrix sum C = A + B using loop tiling.
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
    int bi, bj;
    register int min_i, min_j;
    register double *pa, *pb, *pr;

    for (bi = 0; bi < N; bi += block_size)
    {
        for (bj = 0; bj < N; bj += block_size)
        {
            min_i = MIN(bi + block_size, N);

            for (i = bi; i < min_i; i++)
            {
                pa = &A[i * N + bj];
                pb = &B[i * N + bj];
                pr = &result[i * N + bj];
                min_j = MIN(bj + block_size, N);

                for (j = bj; j < min_j; j++)
                {
                    *pr = *pa + *pb;
                    pa++;
                    pb++;
                    pr++;
                }
            }
        }
    }

    return result;
}

/**
 * @brief Solves the matrix equation by calling the optimized elementary matrix operation functions.
 *
 * This function implements the same logic as the non-optimized version, but calls the
 * optimized helper functions defined in this file.
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
