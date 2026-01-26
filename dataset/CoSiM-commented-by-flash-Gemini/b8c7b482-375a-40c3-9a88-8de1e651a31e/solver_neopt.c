
#include "utils.h"

#define DEBUG 0




void transpose(int N, double *a, double *res)
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            res[i * N + j] = a[j * N + i];
        }
    }
}


void mul(int N, double *a, double *b, double *res)
{

    int i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            res[i * N + j] = 0.0;
            for (k = 0; k < N; k++)
            {
                res[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}


void add_sq_m(int N, double *a, double *b)
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            a[i * N + j] += b[i * N + j];
        }
    }
}


void at_a_mul(int N, double *at, double *res)
{
    int i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            res[i * N + j] = 0.0;

            int max = (i < j) ? i : j;

            for (k = 0; k <= max; k++)
            {
                res[i * N + j] += at[i * N + k] * at[j * N + k];
            }
        }
    }
}


void sup_tri_mul(int N, double *a, double *b, double *res)
{
    int i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            res[i * N + j] = 0.0;
            for (k = i; k < N; k++)
            {
                res[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}


void regular_x_trans(int N, double *a, double *res)
{
    int i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            res[i * N + j] = 0.0;
            for (k = 0; k < N; k++)
            {
                res[i * N + j] += a[i * N + k] * a[j * N + k];
            }
        }
    }
}

double *my_solver(int N, double *a, double *b)
{
    printf("NEOPT SOLVER\n");

    double *at = malloc(N * N * sizeof(double));
    double *b_bt = malloc(N * N * sizeof(double));
    double *at_a = malloc(N * N * sizeof(double));
    double *a_b_bt = malloc(N * N * sizeof(double));
    double *res = a_b_bt;

    
    transpose(N, a, at);
    
    regular_x_trans(N, b, b_bt);
    
    at_a_mul(N, at, at_a);
    
    sup_tri_mul(N, a, b_bt, a_b_bt);

    
    add_sq_m(N, a_b_bt, at_a);

    free(at);
    free(b_bt);
    free(at_a);

    return res;
}
