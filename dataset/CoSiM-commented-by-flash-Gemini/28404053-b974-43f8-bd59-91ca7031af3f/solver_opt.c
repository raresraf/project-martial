#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	
	register int i = 0;
	register int j = 0;
	register int k = 0;

	double *fst = malloc(N * N * sizeof(double));
	
	for(i = 0; i < N; i++){
  		double *orig_pa = &A[i * N];
  		for(j = 0; j < N; j++){
    		double *pa = orig_pa;
    		double *pb = &B[j];
    		register double suma = 0;
    		for(k = 0; k < N; k++){
      			suma += *pa * *pb;
      			pa++;
      			pb += N;
    		}
    		fst[i * N + j] = suma;
  		}
	}

	double *snd = malloc(N * N * sizeof(double));

	for (i = 0; i < N; i++) {
		register int index = i * N;
		for (j = 0; j < N; j++) {
			register double sum = 0;
			for (k = 0; k < N; k++) {
				sum += fst[index + k] * B[j * N + k];
			}
			snd[index + j] = sum;
		}
	}

	double *third = malloc(N * N * sizeof(double));
	
	for (i = 0; i < N; i++) {
		register int index = i * N;
		for (k = 0; k < N; k++) {
			register int kn = k * N;
			for (j = 0; j < N; j++) {
				third[index + j] += A[kn + i] * A[kn + j];
			}
		}
	}

	double *C = malloc(N * N * sizeof(double));

	for (i = 0; i < N; i++) {
		register int index = i * N;
		for (j = 0 ; j < N; j++) {
			register int index_j = index + j;
			C[index_j] += third[index_j] + snd[index_j];
		}
	}

	free(fst);
	free(snd);
	free(third);

	return C;	
}

