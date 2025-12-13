
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	int i, j, k;

	double *C1 = calloc(N * N, sizeof(double));

	
	
	for (i = 0; i < N; i++) {
		double *orig_pa = &B[i * N];

		for (j = i; j < N; j++) {
			double *pa = orig_pa;
			
			double *pb = &B[j * N];
			register double sum = 0.0;

			for (k = 0; k < N; k++) {
				sum += *pa * *pb;
				pa++;
				pb++;
			}
			C1[i * N + j] = sum;
			C1[j * N + i] = sum;
		}
	}

	double *C2 = calloc(N * N, sizeof(double));

	
	
	
	
	for (i = 0; i < N; i++) {
		
		double *pa = &A[i * N + i];
		double *pb = &C1[i * N];

		for (k = i; k < N; k++) {

			for (j = 0; j < N; j++) {
				C2[i * N + j] += *pa * *pb;
				pb++;
			}

			pa++;
		}

	}

	double *C3 = calloc(N * N, sizeof(double));

	
	
	
	for (i = 0; i < N; i++) {
		double *orig_pa = &A[i];

		for (j = i; j < N; j++) {
			double *pa = orig_pa; 
			double *pb = &A[j];
			register double sum = 0.0;

			for (k = 0; k <= i; k++) {
				
				sum += *pa * *pb;
				pa += N;
				pb += N;
			}
			C3[i * N + j] = sum;
			C3[j * N + i] = sum;
		}
	}

	for (i = 0; i < N * N; i++)
		C2[i] += C3[i];

	free(C1);
	free(C3);

	return C2;	
}
