
#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

    double *AB = calloc(N * N, sizeof(double));
    double *C = calloc(N * N, sizeof(double));
    int i, j, k;
    
    for (i = 0; i < N; ++i)
        for (j = 0; j < N; ++j) {
            if (i < j)
                for (k = 0; k <= i; ++k)
                    C[i * N + j] += A[k * N + i] * A[k * N + j];
            else 
                for (k = 0; k <= j; ++k)
                    C[i * N + j] += A[k * N + i] * A[k * N + j];
        }
    
    
    for (i = 0; i < N; ++i)
        for (j = 0; j < N; ++j) {
            for (k = i; k < N; ++k)
                AB[i * N + j] += A[i * N + k] * B[k * N + j];
        }
    
    for (i = 0; i < N; ++i)
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k)
                C[i * N + j] += AB[i * N + k] * B[j * N + k];
        }
    free(AB);
    return C;
	
}
