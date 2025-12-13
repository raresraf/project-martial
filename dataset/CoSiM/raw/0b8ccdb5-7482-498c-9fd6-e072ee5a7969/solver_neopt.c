
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	int i,j,k;
	double *C, *D;
	C = (double *)calloc(N * N, sizeof(double));
	D = (double *)calloc(N * N, sizeof(double));

	
	for (i = 0; i < N; i++) {
		for (j = i; j < N; j++) {
			for (k = 0; k < N; k ++) {
				C[j + N * i] += 
					B[k + N * i] *
					B[k + N * j];
			}

			C[i + N * j] = C[j + N * i]; 
		}	
	}

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = i; k < N; k ++) {
				D[j + N * i] += 
					A[k + N * i] *
					C[j + N * k];
			}
		}	
	}

	free(C);
	C = (double *)calloc(N * N, sizeof(double));

	
	for (i = 0; i < N; i++) {
		for (j = i; j < N; j++) {
			for (k = 0; k < N; k++) {
				C[j + N * i] += 
					A[i + N * k] *
					A[j + N * k];
			}

			
			
			D[i + N * j] += C[j + N * i]; 
			D[j + N * i] += C[j + N * i]; 
		}	
	}

	free(C);
	return D;
}

