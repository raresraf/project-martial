
#include "utils.h"

#include <cblas.h>

#include <string.h>


double * my_solver(int N, double * A, double * B) {

    double * C;
    double * AB;

    C = calloc(N * N, sizeof( * C)); 
    if (C == NULL) {
        return NULL;
    }

    AB = calloc(N * N, sizeof( * C)); 
    if (AB == NULL) {
        return NULL;
    }

    
    cblas_dcopy(
        N * N,
        B, 1,
        AB, 1
    );

    cblas_dtrmm(
        CblasRowMajor,
        CblasLeft,
        CblasUpper,
        CblasNoTrans,
        CblasNonUnit,
        N, N,
        1.0, A, N,
        AB, N
    );

    
    cblas_dcopy(
        N * N,
        A, 1,
        C, 1
    );

    cblas_dtrmm(
        CblasRowMajor,
        CblasLeft,
        CblasUpper,
        CblasTrans,
        CblasNonUnit,
        N, N,
        1.0, A, N,
        C, N
    );
    
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        N, N, N,
        1.0, AB, N,
        B, N, 1.0, C, N
    );

    free(AB);

    return C;
}