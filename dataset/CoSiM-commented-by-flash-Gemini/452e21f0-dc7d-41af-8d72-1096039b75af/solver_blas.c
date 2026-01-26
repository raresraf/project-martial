
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "cblas.h"

double *my_solver(int N, double *A, double *B) {

        double alpha, beta;

        int dimensionMatrix = N * N;
 
        double *B2 = (double *)calloc(dimensionMatrix, sizeof(double));

        double *C = (double *)calloc(dimensionMatrix, sizeof(double));

        if (B2 == NULL || C == NULL) {
                return NULL;
        }

        alpha = 1.0;

        beta = 1.0;

        
        memcpy(C, A, dimensionMatrix * sizeof(double));

        
        memcpy(B2, B, dimensionMatrix * sizeof(double));
                
        
        cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
                CblasNonUnit, N, N, alpha, A, N, B2, N);

        
        cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
                CblasNonUnit, N, N, alpha, A, N, C, N);

        
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, N, N, alpha, B2, N, B, N, beta, C, N);


        free(B2);

	return C;
}
