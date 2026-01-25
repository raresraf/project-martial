
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	double c[N][N], d[N][N];
	double *e = malloc(N * N * sizeof(double));
	
	for(int i = 0; i < N; 	i++) {
		for(int j = 0; j < N; j++) {
				c[i][j] = 0.0;
				for (int k = 0; k < N; k++) {
					if(i <= k) {
						c[i][j] += A[i * N + k] * B[k * N + j];
					}
					
				}
		}
	}

	for(int i = 0; i < N; 	i++) {
		for(int j = 0; j < N; j++) {
			d[i][j] = 0.0;
			for (int k = 0; k < N; k++) {
				d[i][j] += c[i][k] * B[j * N + k];
			}
		}
	}

	
	for(int i = 0; i < N; 	i++) {
		for(int j = 0; j < N; j++) {
			e[i * N + j]= 0.0;

			for (int k = 0; k < N; k++) {
				if(k <= i || k < j) {
					e[i * N + j] += A[k * N + i] *  A[k * N + j];
				}
				
			}

			e[i * N + j] += d[i][j];
		}
	}

	
	return e;

}