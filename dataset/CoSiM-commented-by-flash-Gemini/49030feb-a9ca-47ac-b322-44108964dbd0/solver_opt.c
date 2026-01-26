
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	register double *C = malloc(N * N * sizeof(double));
	register double *mat = malloc(N * N * sizeof(double));
	register double *tmp, *point_i, *point_j, *p_C;
	
	for (register int i = 0; i < N; i++) {
		tmp = &A[i * N];
		p_C = &mat[i * N];
		for (register int j = 0; j < N; j++) {
			point_i = tmp + i;
			point_j = &B[j];
			point_j += i * N;
			register double tmp_res = 0.0;
			for (register int k = i; k < N; k++) {
					tmp_res += *point_i * *point_j;
					point_i++;
					point_j += N;

			}
			*p_C = tmp_res;
			p_C++;
		}
	}

	for (register int i = 0; i < N; i++) {
		tmp = &mat[i * N];
		p_C = &C[i * N];
		for (register int j = 0; j < N; j++) {
			point_i = tmp;
			point_j = &B[j * N];
			register double tmp = 0.0;
			for (register int k = 0; k < N; k++) {
					tmp += *point_i * *point_j;
					point_i++;
					point_j++;
			}
			*p_C = tmp;
			p_C++;
		}
	}
	
	for (register int i = 0; i < N; i++) {
		p_C = &C[i * N];
		for (register int j = 0; j < N; j++) {
			register double tmp_res = 0.0;
			for (register int k = 0; k <= i; k++) {
				register double *point = &A[k * N];
				tmp_res += *(point + i) * *(point + j);
			}
			*p_C += tmp_res;
			p_C++;
		}
	}
	free(mat);
	return C;
}