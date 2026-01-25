
#include "utils.h"
#include <cblas.h>
#include <string.h>
#define CHECK_ALLOC(x) \
            if (x == NULL) \
            { \
                perror("Allocation failed\n"); \
                exit(-1); \
            }


double* my_solver(int N, double *A, double* B) {
    double* R1 = (double*)calloc(N * N, sizeof(double));
    CHECK_ALLOC(R1);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 0, R1, N);
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, R1, N);
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, A, N);

    double* I = (double*)calloc(N * N, sizeof(double));
    CHECK_ALLOC(I);

    int i;
    for (i = 0; i < N * N; i += N + 1)
    {
        I[i] = 1;
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, R1, N, I, N, 1, A, N);
    memcpy(R1, A, N * N * sizeof(double));

    free(I);

    return R1;
}
