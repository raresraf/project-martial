
#include "utils.h"



double* my_solver(int N, double *A, double* B) {
	
	double *AB;
	double *ABBt;
	double *AtA;
	double *C;
	int i, j, k;

	AB = calloc(N * N,  sizeof(double));
	ABBt = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));
	AtA = calloc(N * N, sizeof(double));
	if (AB == NULL || ABBt == NULL || C == NULL || AtA == NULL) {
        perror("malloc failed\n");
        exit(EXIT_FAILURE);
    }

	
	for (i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for (k = i; k < N; k++) {
               AB[i * N + j] += A[i * N + k] * B[k * N + j];
            }
		}
	}

	
	for (i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
               	ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
            }
		}
	}

	  
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				if (A[N * k + i] == 0 || A[N * k + j] == 0)
					break;
				AtA[N * i + j] += A[N * k + i] * A[N * k + j];
			}
		}
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);
	return C;

}
