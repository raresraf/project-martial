
#include "utils.h"



#define min(x, y) (((x) < (y)) ? (x) : (y))

double* my_solver(int N, double *A, double* B) {

	double *C = (double *)malloc(N * N * sizeof(double));
	double *term = (double *)malloc(N * N * sizeof(double));

	for (int i = 0;i < N;i++) {
		for (int j = 0;j < N;j++) {
			term[i * N + j] = 0.0;
			for (int k = i;k < N;k++) {
				term[i * N + j] += (double)(A[i * N + k] * B[k * N + j]);
			}
		}
	}

	for (int i = 0;i < N;i++) {
		for (int j = 0;j < N;j++) {
			C[i * N + j] = 0.0;
			for (int k = 0;k < N;k++) {
				C[i * N + j] += (double)(term[i * N + k] * B[j * N + k]);
			}
		}
	}

	for (int i = 0;i < N;i++) {
                for (int j = 0;j < N;j++) {
                        term[i * N + j] = 0.0;
                        for (int k = 0;k <= min(i, j);k++) {
                                term[i * N + j] += (double)(A[k * N + i] * A[k * N + j]);
                        }
                }
        }

	for (int i = 0;i < N;i++) {
                for (int j = 0;j < N;j++) {
                	C[i * N + j] += (double)term[i * N + j];
                }
        }

	free(term);

	return C;
}
