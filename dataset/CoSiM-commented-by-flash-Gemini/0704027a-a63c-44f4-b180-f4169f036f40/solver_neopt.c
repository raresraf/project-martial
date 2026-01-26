
#include "utils.h"
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define CHECK_ALLOC(x) \
            if (x == NULL) \
            { \
                perror("Allocation failed\n"); \
                exit(-1); \
            }


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
