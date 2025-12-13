
 #include "cblas.h"
#include "utils.h"


double* my_solver(int N, double *A, double *B) {
    double *arr2 = calloc(N * N, sizeof(double));
    double *arr3 = calloc(N * N, sizeof(double));
    double *arr = calloc(N * N, sizeof(double));
    double *arr1 = calloc(N * N, sizeof(double));


    for(int i = 0; i < N ; i ++){
	for(int j=0; j < N; j++)
        	arr[i * N + j] = A[i * N + j];
    }

    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, arr, N);


    for(int i = 0; i < N ; i ++){
	for(int j=0; j < N; j++)
        	arr1[i * N + j] = B[i * N + j];
    }

    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, arr1, N);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, arr1, N, B, N, 1, arr2, N);

    for(int i = 0; i < N * N; i++){
        arr3[i] = arr2[i] + arr[i];
    }

    for(int i = 0; i < N ; i ++){
	for(int j=0; j < N; j++)
        	arr3[i * N + j] = arr2[i * N + j] + arr[i * N + j];
    }

    free(arr2);
    free(arr1);
    free(arr);

    return arr3;
}
