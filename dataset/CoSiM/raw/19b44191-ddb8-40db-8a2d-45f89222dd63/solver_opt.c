
#include "utils.h"


double* my_solver(int N, double *A, double* B) {

	register double *rez, *C, *A_trans;
	register int i, j, k;
	

	C = calloc(N * N, sizeof(double));
	rez = calloc(N * N, sizeof(double));
	A_trans = calloc(N * N, sizeof(double));

	for (i = 0; i < N; ++i) {
		register double *pointer_trans = A_trans + i;
		register double *pointer = A + i * N;
		for (j = 0; j < N; ++j) {
			*pointer_trans = *pointer;
			pointer_trans += N;
			pointer++;
		}
	}
	
	for (i = 0; i < N; ++i) {
		register double *orig_pa = B + i * N;
		for (j = 0; j < N; ++j) {
			register double suma = 0.0;
			register double *pa = orig_pa; 
			register double *pb = B + j * N;
			for (k = 0; k < N; ++k) {
				suma += *pa * *pb;
				pa++;
				pb++;
			}
			rez[i * N + j] = suma;
		}
	}

	
	for (i = 0; i < N; ++i) {
		register double *orig_pa = A + i * N;
		for (j = 0; j < N; ++j) {
			register double suma = 0.0;
			register double *pa = orig_pa; 
			register double *pb = rez + j;
			for (k = 0; k < N; ++k) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			C[i * N + j] = suma;
		}
	}

	
	for (i = 0; i < N; ++i) {
		register double *orig_pa = A_trans + i * N;
		for (j = 0; j < N; ++j) {
			register double suma = 0.0;
			register double *pa = orig_pa; 
			register double *pb = A + j;
			for (k = 0; k < N; ++k) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			C[i * N + j] += suma;
		}
	}

	free(A_trans);
	free(rez);
	return C;
}
