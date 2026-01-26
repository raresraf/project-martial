
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");
	int k, j, i;
	double *C = (double*) calloc(N * N, sizeof(double));
    if (C == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}
    double *D = (double*) calloc(N * N, sizeof(double));
    if (D == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}

    
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = i; k < N; k++) {
                *(C + i * N + j) += *(A + i * N + k) * *(B + k * N + j);
            }
        }
    }
    
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                *(D + i * N + j) += *(C + i * N + k) * *(B + j * N + k);
            }
        }
    }
    
	for (k = 0; k < N; k++) {
		for (i = k; i < N; i++) {
            for (j = k; j < N; j++) {
                *(D + i * N + j) += *(A + k * N + i) * *(A + k * N + j);
            }
		}
	}
	free(C);
	return D;
}
