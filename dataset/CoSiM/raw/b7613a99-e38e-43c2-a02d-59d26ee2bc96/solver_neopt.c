

#include <stdlib.h>
#include <string.h>
#include "utils.h"


double* my_solver(int N, double *A, double* B) {

	double *C;
	double *AA;
	double *AB;

	
	C = calloc(N * N, sizeof(*C));
	AA = calloc(N * N, sizeof(*AA));
	AB = calloc(N * N, sizeof(*AB));

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < j + 1; k++) {
				AA[i  * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = i; k < N; k++) {
				AB[i  * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				C[i  * N + j] += AB[i * N + k] * B[j * N + k];
			}
		}
	}

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C[i * N + j] += AA[i * N + j];
		}
	}


	printf("NEOPT SOLVER\n");
	
	free(AA);
	free(AB);

	return C;
}
