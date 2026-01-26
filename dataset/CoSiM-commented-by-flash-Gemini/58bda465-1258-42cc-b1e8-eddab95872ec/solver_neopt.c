
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	
	double *AB = malloc(N * N * sizeof(double));
	double *ABBt = malloc(N * N * sizeof(double));
	double *AtA = malloc(N * N * sizeof(double));
	double *C = malloc(N * N * sizeof(double));

	if (AB == NULL || ABBt == NULL || AtA == NULL || C == NULL){
		fprintf(stderr, "malloc error\n");
		exit(EXIT_FAILURE);
	}

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			AB[i * N + j] = 0;
			for (int k = 0; k < N; k++) {
				
				if (i <= k)
					AB[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			ABBt[i * N + j] = 0;
			for (int k = 0; k < N; k++) {
				ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			AtA[i * N + j] = 0;
			for (int k = 0; k < N; k++) {
				
				if (i >= k && k <= j)
					AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	
	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
