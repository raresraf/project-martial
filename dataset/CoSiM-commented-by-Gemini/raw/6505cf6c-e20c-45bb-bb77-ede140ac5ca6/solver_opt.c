
#include <stdlib.h>
#include "utils.h"

double* my_solver(int N,  double *A,  double* B) {
	double *AB;
	double *C;
	register int i, j, k;

	AB = calloc(N * N , sizeof(double));
	if (AB == NULL) 
		exit(EXIT_FAILURE);
	C = calloc(N * N , sizeof(double));
	if (C == NULL)
		exit(EXIT_FAILURE);
	
	
	
	for (i = 0; i < N; i++) {
		
		register double *orig_pa = A + i * N;
		 
		register double *orig_ab = AB + i * N;
		for (j = 0; j < N; j++, orig_ab++) {
			register double sum = 0;
			
			register double *pa = orig_pa + i; 
			register double *pb = B + i * N + j;
			
			for (k = i; k < N; k++, pa++, pb += N) {
				sum += *pa * *pb;
			}
			*orig_ab = sum;
		}
	}

	

	for (i = 0; i < N; i++) {
		 
		register double *orig_ab = AB + i * N;
		 
		register double *orig_c = C + i * N; 
		for (j = 0; j < N; j++, orig_c++) {
			register double sum = 0;
			register double *pab = orig_ab;
			
			register double *pbt = B + j * N;

			for (k = 0; k < N; k++, pab++, pbt++) {
				sum += *pab * *pbt;
			}
			*orig_c = sum;
		}
	}

	

	for (i = 0; i < N; i++) {
		 
		register double *orig_at = A + i;
		 
		register double *orig_c = C + i * N; 
		for (j = 0; j < N; j++, orig_c++) {
			register double sum = 0;
			register double *pat = orig_at;
			register double *pa = A + j;
			for (k = 0; k <= i; k++, pat += N, pa += N) {
				sum += *pat * *pa;
			}
			*orig_c += sum;
		}
	}
	

	free(AB);
	return C;	
}
