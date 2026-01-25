
#include <string.h>
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *C = (double*) calloc(N * N, sizeof(double));
	if (!C)
		return NULL;

	double *aux = (double*) calloc(N * N, sizeof(double));
	if (!aux)
		return NULL;
	int i, j, k;
	for (i = 0; i < N; i++) {
		double *orig_pa = &A[i * N];
		for (j = 0; j < N; j++) {
			register double *pa = orig_pa + i;
			register double *pb = &B[i * N + j];
			register double suma = 0;
			for (k = i; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			aux[i * N + j] = suma;
		}
	}

	for (i = 0; i < N; i++) {
		double *origin_aux = &aux[i * N];
		for (j = 0; j < N; j++) {
			register double suma = 0;
			register double *paux = origin_aux;
			register double *pb_t = &B[j *  N];
			for (k = 0; k < N; k++) {
				suma += *paux * *pb_t;
				paux++;
				pb_t ++;
			}
			C[i * N + j] = suma;
		}
	}

	memset(aux, 0, N * N * sizeof(double));
	for (i = 0; i < N; i++) {
		double *origin_at = &A[i];
		for (j = 0; j < N; j++) {
			register double suma = 0;
			register double *pa_t = origin_at;
			register double *pa = &A[j];
			for (k = 0; k < i + 1; k++) {
				suma += *pa_t * *pa;
				pa_t+= N;
				pa += N;
			}
			aux[i * N + j] = suma;
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += aux[i * N + j];
		}
	}

	free(aux);

	return C;
}
