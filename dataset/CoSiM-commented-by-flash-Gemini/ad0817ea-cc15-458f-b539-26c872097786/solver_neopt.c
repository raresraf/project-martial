
#include "utils.h"



double* my_solver(int N, double *A, double* B) {

	int i, j, k;
	double *res1, *res2, *res3;
	int min;

	res1 = malloc(N*N*sizeof(double));
	res2 = malloc(N*N*sizeof(double));
	res3 = malloc(N*N*sizeof(double));

	
	for (i = 0; i < N; i++) {
   		for (j = 0; j < N; j++) {
      		res1[i * N + j] = 0.0;

      		if (j < i) {
      			min = j;
      		} else {
      			min = i;
      		}

			for (k = 0; k <= min; k++) {
				res1[i * N + j] += A[k * N + i] * A[k * N + j];
      		}
   		}
	}

	
	for (i = 0; i < N; i++) {
   		for (j = 0; j < N; j++) {
      		res2[i * N + j] = 0.0;

	      	for (k = 0; k < N; k++) {
				res2[i * N + j] += A[i * N + k] * B[k * N + j];
      		}
   		}
	}

	
	for (i = 0; i < N; i++) {
   		for (j = 0; j < N; j++) {
      		res3[i * N + j] = 0.0;

	      	for (k = 0; k < N; k++) {
				res3[i * N + j] += res2[i * N + k] * B[j * N + k];
      		}
   		}
	}

	
	for (i = 0; i < N; i++) {
   		for (j = 0; j < N; j++) {
      		res3[i * N + j] += res1[i * N + j];

   		}
	}

	free(res1);
	free(res2);
	

	return res3;

	printf("NEOPT SOLVER\n");
	return NULL;
}
