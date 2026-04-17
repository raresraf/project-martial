/*
 * Module: @raw/7d87b802-1fe0-4a7d-911d-4c5f132c4e4b/solver_neopt.c
 * Purpose: Unoptimized, naive matrix solver implementation.
 */
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

	double c[N][N], d[N][N];
	double *e = malloc(N * N * sizeof(double));
	
    // Pre-condition: A and B are N*N matrices. Invariant: c holds unoptimized matrix products.
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

    // Pre-condition: c and B are N*N matrices. Invariant: d holds products.
	for(int i = 0; i < N; 	i++) {
		for(int j = 0; j < N; j++) {
			d[i][j] = 0.0;
			for (int k = 0; k < N; k++) {
				d[i][j] += c[i][k] * B[j * N + k];
			}
		}
	}

	// Pre-condition: A and d are evaluated. Invariant: e accumulates final matrix result.
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