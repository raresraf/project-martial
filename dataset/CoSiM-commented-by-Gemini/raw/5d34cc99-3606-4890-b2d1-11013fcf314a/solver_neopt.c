/**
 * @file solver_neopt.c
 * @brief Simple nested iteration solver for validation.
 */
#define MIN(a,b) (((a)<(b))?(a):(b))

#include "utils.h"
#include <string.h>

/**
 * @brief Zero initializes matrix arrays.
 */
void allocate_matrix(int N, double **AB, double **ABBt,
                        double **AtA, double **C) {

    *AB = calloc (N * N, sizeof (double));
    *ABBt = calloc (N * N, sizeof (double));
    *AtA = calloc (N * N, sizeof (double));
    *C = calloc (N * N, sizeof (double));
}

/**
 * @brief Executes standard nested iterations.
 */
double *my_solver(int N, double *A, double* B) {

    double *AB, *ABBt, *AtA, *C;
    int i, j, k;

    allocate_matrix(N, &AB, &ABBt, &AtA, &C);

    /**
     * @pre Memory chunks empty.
     * @post Computes AB inner product bounds.
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = i; k < N; k++) {
                AB[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

    /**
     * @pre Matrix AB populated.
     * @post Generates AB * B transpose form.
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
            }
        }
    }

    /**
     * @pre Initial matrix A given.
     * @post Creates symmetric AtA sub calculations.
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k <= MIN(i, j); k++) {
                AtA[i * N + j] += A[k * N + i] * A[k * N + j];
            }
        }
    }

    /**
     * @pre Final matrices available.
     * @post Combines intermediate calculations into C.
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
        }
    }

    free(AB);
    free(ABBt);
    free(AtA);

    return C;
}
