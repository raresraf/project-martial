
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double *C;
	double *D;
	double *E;
	register int i, j, k;
	register int size = N*N;
	
	C = calloc(size, sizeof(double));
	D = calloc(size, sizeof(double));
	E = calloc(size, sizeof(double));

	for(i = 0; i < N; i++){
		double *orig_pa = &A[i * N];
		for (j = 0; j < N; j++) {
			register double *pa = orig_pa;
			register double *pb = &B[j];
			register double suma = 0;
			int var = 0;
			if (i >= j) {
				var = i;
			}
			pa += var;
			pb += var * N;
			for (k = var; k < N; k++) {
				suma += *pa * *pb;
				pa++;
				pb += N;
			}
			C[i * N + j] = suma;
		}
	}
	for(i = 0; i < N; i++) {
		double *orig_pc = &C[i * N];
		for(j = 0; j < N; j++) {
			register double *pc = orig_pc;
			register double *pb1 = &B[j * N];
			register double sum = 0;
			for(k = 0; k < N; k++) {
				sum += *pc * *pb1;
				pc++;
				pb1++;
			}
			D[i * N + j] = sum;
		}
	}

	for(i = 0; i < N; i++){
		for (j = 0; j < N; j++) {
		register double *pa1 = &A[i];
		register double *pa2 = &A[j];
		register double sum3 = 0;	
		for (k = 0; k <= j; k++) {
			sum3 += *pa1 * *pa2;
			pa1 += N;
			pa2 += N;
		}
		E[i * N + j] = D[i * N + j] + sum3;
		}
	}

	free(C);
	free(D);

 	return E;	
}
