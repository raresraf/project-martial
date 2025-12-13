
#include "utils.h"
#include <string.h>




double* my_solver(int N, double *A, double* B) {	
	double *ab = (double *) calloc(N * N, sizeof(double));
	double *abbt = (double *) calloc(N * N, sizeof(double));
	double *ata = (double *) calloc(N * N, sizeof(double));
	double *result =  (double *) calloc(N * N, sizeof(double));

	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = i; k < N; ++k) {
				ab[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				abbt[i * N + j] += ab[i * N + k] * B[j * N + k];
			}
		}
	}
	
	
	for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                        for (int k = 0; k <= i; ++k) {
                                ata[i * N + j] += A[k * N + i] * A[k * N + j];
                        }
                }
        }
	
	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			result[i * N + j] = abbt[i * N + j] + ata[i * N + j];
		}
	}
	
	free(ab);
	free(abbt);
	free(ata);
	return result;
}
