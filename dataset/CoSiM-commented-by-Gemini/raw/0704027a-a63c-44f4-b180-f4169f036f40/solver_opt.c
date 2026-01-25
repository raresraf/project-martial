
#include "utils.h"
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CHECK_ALLOC(x) \
            if (x == NULL) \
            { \
                perror("Allocation failed\n"); \
                exit(-1); \
            }


int block_size = 100;

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
