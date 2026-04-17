/*
 * Module: @raw/39adfa5f-9347-4f4c-9ab7-097418b531e3/solver_neopt.c
 * High-level purpose: Non-optimized solver.
 */
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

    /* Pre-condition: Matrices initialized. Invariant: Loop variables bounded by N. */
	double *BBt = (double *)calloc(N * N, sizeof(double));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				BBt[i * N + j] += B[i * N + k] * B[j * N + k];
			}
		}
	}

    
    double *A_BBt = (double *)calloc(N * N, sizeof(double));
    for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = i; k < N; k++) {
				A_BBt[i * N + j] += A[i * N + k] * BBt[k * N + j];
			}
		}
	}

    
    double *AtA = (double *)calloc(N * N, sizeof(double));
    for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k <= i; k++) {
				AtA[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

    
    for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A_BBt[i * N + j] += AtA[i * N + j];
		}
	}

	
    free(AtA);
    free(BBt);
	return A_BBt;
}
