
#include <stdlib.h>
#include "utils.h"

double *my_solver(int N, double *A, double *B) {

	int i, j, k;

        double *A2 = (double *)calloc(N * N, sizeof(double));
        double *B2 = (double *)calloc(N * N, sizeof(double));
        double *B3 = (double *)calloc(N * N, sizeof(double));

        if (A2 == NULL || B2 == NULL || B3 == NULL) {
                return NULL;
        }

        
        for (i = 0; i < N; ++i) {
                for (j = 0; j < N; ++j) {
                        register double sum = 0.0;

                        for (k = i; k < N; ++k) {
                                sum += A[i * N + k] * B[k * N + j];
                        }

                        B2[i * N + j] = sum;
                }
        }

        
        for (i = 0; i < N; ++i) {
                for (j = 0; j < N; ++j) {
                        register double sum = 0.0;

                        for (k = 0; k < N; ++k) {
                                sum += B2[i * N + k] * B[j * N + k];
                        }

                        B3[i * N + j] = sum;
                }
        }

        
        for (i = 0; i < N; ++i) {
                for (j = 0; j < N; ++j) {
                        register double sum = 0.0;

                        for (k = 0; k <= j; ++k) {
                                sum += A[k * N + i] * A[k * N + j];
                        }

                        A2[i * N + j] = sum;
                }
        }

        
        for (i = 0; i < N; ++i) {

                for (j = 0; j < N; ++j) {
                        B3[i * N + j] += A2[i * N + j];
                }
        }

        free(A2);
        free(B2);

        return B3;
}
