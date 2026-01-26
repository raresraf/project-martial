
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	int i, j, k;

	
	double *C = calloc(N * N, sizeof(double));
	
	for (i = 0; i < N; i++) {
		double* ptrc_i = C + i * N;
		double* ptrb_i = B + i * N;
		for (j = 0; j <= i; j++) {
			double* ptrc_j = C + j * N;
			double* ptrb_j = B + j * N;
			double temp = 0.0;
			for (k = 0; k < N; k++) {
				temp += *(ptrb_i + k) * *(ptrb_j + k);
			}
			*(ptrc_i + j) = temp;
			*(ptrc_j + i) = temp;
		}
	}

	
	double *D = calloc(N * N, sizeof(double));
	
	for (i = 0; i < N; i++) {
		double* ptrd_i = D + i * N;
		double* ptra_i = A + i * N;
		for (k = i; k < N; k++) {
			
			
			double* ptrc_k = C + k * N;
			for (j = 0; j < N; j++) {
				*(ptrd_i + j) += *(ptra_i + k) * *(ptrc_k + j);
			}
		}
	}

	
	double *E = calloc(N * N, sizeof(double));
	
	for (i = 0; i < N; i++) {
		double* ptre_i = E + i * N;
		double* ptra_i = A + i;
		
		for (j = 0; j <= i; j++) {
			int min = i < j ? i : j;
			double temp = 0;
			double* ptre_j = E + j * N;
			double* ptra_j = A + j;
			
			for (k = 0; k <= min; k++) {
				temp += *(ptra_i + k * N) * *(ptra_j + k * N);
			}
			*(ptre_i + j) = temp;
			*(ptre_j + i) = temp;
		}
	}
	
	double *F = calloc(N * N, sizeof(double));
	
	for(i = 0; i < N; i++) {
		double* ptrf_i = F + i * N;
		double* ptrd_i = D + i * N;
		double* ptre_i = E + i * N;
        for(j = 0; j < N; j++) {
			*(ptrf_i + j) = *(ptrd_i + j) + *(ptre_i + j);
        }
    }

	free(C);
	free(D);
	free(E);
	return F;
	
}
