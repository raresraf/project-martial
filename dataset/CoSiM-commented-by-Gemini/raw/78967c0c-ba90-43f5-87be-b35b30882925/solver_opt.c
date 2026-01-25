
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	
	double *At;
	double *Bt;
	double *C;
	
	double *Aux1;
	double *Aux2;
	
	register int i, j, k;
	
	At = calloc(N * N, sizeof(double));
	Bt = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));
	
	Aux1 = calloc(N * N, sizeof(double));
	Aux2 = calloc(N * N, sizeof(double));
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			At[j * N + i] = A[i * N + j];
		}
	}
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			Bt[j * N + i] = B[i * N + j];
		}
	}
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			register double suma = 0.0;
			
			for (k = i; k < N; k++) {
				suma += A[i * N + k] * B[k * N + j];
			}
			
			Aux1[i * N + j] = suma;
		}
	}
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			register double suma = 0.0;
			
			for (k = 0; k < N; k++) {
				suma += Aux1[i * N + k] * Bt[k * N + j];
			}
			
			Aux2[i * N + j] = suma;
		}
	}
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			Aux1[i * N + j] = 0;
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			register double suma = 0.0;
			
			for (k = 0; k <= i; k++) {
				suma += At[i * N + k] * A[k * N + j];
			}
			
			Aux1[i * N + j] = suma;
		}
	}
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = Aux1[i * N + j] + Aux2[i * N + j];
		}
	}
	
	free(At);
	free(Bt);
	
	free(Aux1);
	free(Aux2);
	
	return C;
}
