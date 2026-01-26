
#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	double *C = (double *)malloc(N * N * sizeof(double));
	double *D = (double *)malloc(N * N * sizeof(double));
	double *E = (double *)malloc(N * N * sizeof(double));

	
	
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            C[i * N + j] = 0;

            for (int k = i; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
		}
    }

	
	for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

			D[i * N + j] = 0;

            for (int k = 0; k < N; k++) {
               D[i * N + j]  += C[i * N + k] * B[j * N + k];
            }
		}
    }

	
	
	for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {

			E[i * N + j] = 0;

            for (int k = 0; k <= i; k++) {
                E[i * N + j] +=  A[k * N + i] * A[k * N + j];
            }
			
			if (i != j)
				E[j * N + i ] = E[i * N + j];
		}
    }
	
	
	for (int i = 0; i < N * N ; i++) {
			D[i] += E[i];
    }

	free(C);
	free(E);

	return D;
}

