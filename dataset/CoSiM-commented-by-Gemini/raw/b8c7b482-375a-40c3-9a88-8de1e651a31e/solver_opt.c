
#include "utils.h"




void matrix_x_matrix(int N, double *a, double *b, double *c)
{
    unsigned int i, k;

    register int j;

    for (i = 0; i < N; i++)
    {
        double *c_aux_org = &c[i * N];

        for (k = 0; k < N; k++)
        {
            register double aux = a[i * N + k];
            register double *b_org = &b[k * N];
            register double *c_org = c_aux_org;

            for (j = 0; j < N; j++)
            {
                *c_org += aux * (*b_org);
                b_org++;
                c_org++;
            }
        }
    }
}


void upper_tri_transpose_x_self(int N, double *at, double *res)
{
    unsigned int i, j;

    register int k;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            register int max = (i < j) ? i : j;
            register double sum = 0.00;
            register double *at_org = &at[i * N];
            register double *at_org2 = &at[j * N];

            for (k = 0; k <= max; k++)
            {
                sum += (*at_org) * (*at_org2);
                at_org++;
                at_org2++;
                
            }
            res[i * N + j] = sum;
        }
    }
}


void transpose(int N, double *a, double *res)
{
    register int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            res[i * N + j] = a[j * N + i];
        }
    }
}


void upper_tri_x_matrix(int N, double *a, double *b, double *c)
{
    unsigned int i, k;

    register int j;

    for (i = 0; i < N; i++)
    {
        double *c_aux_org = &c[i * N];

        for (k = i; k < N; k++)
        {
            register double aux = a[i * N + k];
            register double *b_org = &b[k * N];
            register double *c_org = c_aux_org;

            for (j = 0; j < N; j++)
            {
                *c_org += aux * (*b_org);
                b_org++;
                c_org++;
            }
        }
    }
}


void add_sq_m(int N, double *a, double *b)
{
    register int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            a[i * N + j] += b[i * N + j];
        }
    }
}


double *my_solver(int N, double *a, double *b)
{
    printf("OPT SOLVER\n");

    double *bt = calloc(1, N * N * sizeof(double));
    double *at = calloc(1, N * N * sizeof(double));
    double *a_b = calloc(1, N * N * sizeof(double));
    double *a_b_bt = calloc(1, N * N * sizeof(double));
    double *at_a = calloc(1, N * N * sizeof(double));

    transpose(N, b, bt);
    transpose(N, a, at);

    
    upper_tri_transpose_x_self(N, at, at_a);
    free(at);

    
    upper_tri_x_matrix(N, a, b, a_b);

    
    matrix_x_matrix(N, a_b, bt, a_b_bt);
    free(a_b);
    free(bt);

    
    add_sq_m(N, a_b_bt, at_a);
    free(at_a);

    return a_b_bt;
}
