
#include "utils.h"
#include <string.h>


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	register size_t i, j, k;

	
	double *C = calloc(sizeof(double), N * N);
	double *transA = calloc(sizeof(double), N * N);
	double *transB = calloc(sizeof(double), N * N);

	for (i = 0; i < N; ++i) {
		register double *ptransA = transA + i;
		register double *ptransB = transB + i;
		register double *pA = A + N * i;
		register double *pB = B + N * i;
		for (j = 0; j < N; ++j) {
			*ptransA = *pA;
			*ptransB = *pB;
			ptransA += N;
			ptransB += N;
			pA++;
			pB++;
		}
	}

	
	for (i = 0; i < N; ++i) {
		register double *pc = C + N * i;
		register double *orig_pa = A + N * i;
		for (j = 0; j < N; ++j) {
			register double suma = 0.0;
			register double *pa = orig_pa;
			register double *pb = transB + N * j;
			for (k = i; k < N; ++k) {
				suma += *(pa + k) * *(pb + k);
			}
			*pc = suma;
			pc++;
		}
	}

	double *tmp = calloc(sizeof(double), N * N);
	memcpy(tmp, C, N * N * sizeof(double));

	
	for (i = 0; i < N; ++i) {
		register double *pc = C + N * i;
		register double *orig_tmp = tmp + N * i;
		for (j = 0; j < N; ++j) {
			register double suma = 0;
			register double *ptmp = orig_tmp;
			register double *pb = B + N * j;
			for (k = 0; k < N; ++k) {
				suma += *(ptmp + k) * *(pb + k);
			}
			*pc = suma;
			pc++;
		}
	}

	memcpy(tmp, C, N * N * sizeof(double));
	

	double *tmp2 = calloc(sizeof(double), N * N);
	
	for (i = 0; i < N; ++i) {
		register double *orig_transA = transA + N * i;
		register double *ptmp2 = tmp2 + N * i;
		for (j = 0; j < N; ++j) {
			register double suma = 0;
			register double *ptransA1 = orig_transA;
			register double *ptransA2 = transA + N * j;
			for (k = 0; k < N; ++k) {
				suma += *(ptransA1 + k) * *(ptransA2 + k);
			}
			*ptmp2 = suma;
			ptmp2++;
		}
	}
	
	for (i = 0; i < N; ++i) {
		register double *pc = C + N * i;
		register double *ptmp = tmp + N * i;
		register double *ptmp2 = tmp2 + N * i;
		for (j = 0; j < N; ++j) {
			*pc = *ptmp + *ptmp2;
			pc++, ptmp++, ptmp2++;
		}
	}

	free(tmp);
	free(tmp2);
	free(transA);
	free(transB);

	return C;
}
