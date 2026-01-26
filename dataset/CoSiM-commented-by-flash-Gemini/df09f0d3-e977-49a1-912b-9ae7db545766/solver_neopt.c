
#include "utils.h"
#include <string.h>


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	int i, j, k;
	double *At = (double*) calloc(N * N, sizeof(double));
	double *Bt = (double*) calloc(N * N, sizeof(double));

	for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            At[j * N + i] = A[i * N + j];
        }
    }
	for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            Bt[j * N + i] = B[i * N + j];
        }
    }

	
	double *C1 = (double*) calloc(N * N, sizeof(double));
	for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				for (k = i; k < N; k++) { 
						C1[i * N + j] += A[i * N + k] * B[k * N + j];
				}
			}
		}
	
	double *C2 = (double*) calloc(N * N, sizeof(double));
	for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				for (k = 0; k < N; k++) { 
						C2[i * N + j] += C1[i * N + k] * Bt[k * N + j];
				}
			}
		}

	
	for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				for (k = 0; k <= i && k <= j; k++) { 
						C2[i * N + j] += At[i * N + k] * A[k * N + j];
				}
			}
		}

	free(At);
	free(Bt);
	free(C1);

	return C2;
}
