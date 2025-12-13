
#include "utils.h"





double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	int i, j, k;

	
	double *C = calloc(N * N, sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			
			
			for (k = i; k < N; k++) {
				*(C + i * N + j) += *(A + i * N + k) * *(B + k * N + j);
			}
		}
	}

	
	double *D = calloc(N * N, sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				*(D + i * N + j) += *(C + i * N + k) * *(B + j * N + k);
			}
		}
	}

	
	double *E = calloc(N * N, sizeof(double));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			int min = i < j ? i : j;
			
			for (k = 0; k <= min; k++) {
				*(E + i * N + j) += *(A + k * N + i) * *(A + k * N + j);
			}
		}
	}

	double *F = calloc(N * N, sizeof(double));
	
	for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
			*(F + i * N + j) = *(D + i * N + j) + *(E + i * N + j);
        }
    }

	free(C);
	free(D);
	free(E);
	return F;
}
