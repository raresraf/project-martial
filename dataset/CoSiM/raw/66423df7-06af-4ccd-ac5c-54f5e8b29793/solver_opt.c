
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	
	register int i, j, k;
	double *AB = (double *) calloc(N * N, sizeof(double));
	double *ABBt = (double *)calloc(N * N, sizeof(double));
	double *AtA = (double *)calloc(N * N, sizeof(double));
	double *C = (double *)calloc(N * N, sizeof(double));
	register double *orig_pa, *pa, *pb, suma, *reg_a, *reg_b;

	
	for(i = 0; i < N; i++){
		orig_pa = A + N * i + i;
		for(j = 0; j < N; j++){
			pa = orig_pa;
			pb = B + N * i + j;
			suma = 0;
				for(k = i; k < N; k++){
					suma += *pa * *pb;
					pa++;
					pb += N;
				}
			AB[N * i + j] = suma;
		}
	}
	
	for(i = 0; i < N; i++){
		orig_pa = AB + N * i;
		for(j = 0; j < N; j++){
			pa = orig_pa;
			pb = B + N * j;
			suma = 0;
				for(k = 0; k < N; k++){
					suma += *pa * *pb;
					pa++;
					pb++;
				}
			ABBt[N * i + j] = suma;
		}
	}
	
	for(i = 0; i < N; i++){
		orig_pa = A + i;
		for(j = 0; j < N; j++){
			pa = orig_pa;
			pb = A + j;
			suma = 0;
				for(k = 0; k <= i; k++){
					suma += *pa * *pb;
					pa += N;
					pb += N;
				}
			AtA[N * i + j] = suma;
		}
	}
	
	for (i = 0; i < N; ++i) {
		reg_a = ABBt + i * N;
		reg_b = AtA + i * N;
		for (j = 0; j < N; j += 4) {
			C[i * N + j] = *(reg_a + j) + *(reg_b + j);
			C[i * N + j + 1] = *(reg_a + j + 1) + *(reg_b + j + 1);
			C[i * N + j + 2] = *(reg_a + j + 2) + *(reg_b + j + 2);
			C[i * N + j + 3] = *(reg_a + j + 3) + *(reg_b + j + 3);
		}
	}
	
	free(AB);
	free(ABBt);
	free(AtA);
	return C;
}
