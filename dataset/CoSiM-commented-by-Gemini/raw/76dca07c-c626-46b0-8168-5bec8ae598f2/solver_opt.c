
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	register int i, j, k;
	register double  *pa, *pb, *orig_pa;
	register double suma;
	double *M1 = (double *)calloc(N * N, sizeof(double)); 
	double *M2 = (double *)calloc(N * N, sizeof(double)); 
	double *M3 = (double *)calloc(N * N, sizeof(double)); 
	if (!M1 || !M2 || !M3)
		return NULL;

	for (i = 0; i < N; i++) {
		orig_pa = &(A[i * N + i]);
		for (j = 0; j < N; j++) {
			pa = orig_pa;
			pb = &(B[i * N + j]);
			suma = 0.0;
			for(k = i; k < N; k++) {
				suma += *(pa++) * *pb;
				pb += N;
			}
			M1[i * N + j] = suma;
		}
	}

	for (i = 0; i < N; i++) {
		orig_pa = &(M1[i * N]);
		for (j = 0; j < N; j++) {
			pa = orig_pa;
			pb = &(B[j * N]);
			suma = 0.0;
			for(k = 0; k < N; k++) {
			 	suma += *pa * *pb;
				pa++;
				pb++;
			}
			M2[i * N + j] = suma;
		}
	}
	register int end;
	for (i = 0; i < N; i++) {
		orig_pa = &(A[i]);
		for (j = 0; j < N; j++) {
			if (i < j)
				end = i;
			else
				end = j;
			pa = orig_pa;
			pb = &(A[j]);
			suma = 0.0;
			for(k = 0; k <= end; k++) {
			 	suma += *pa * *pb;
				pa += N;
				pb += N;
			}
			M3[i * N + j] = suma;
		}

	}
	for (i = 0; i < N; i++) {
		pa = &(M2[i]);
		pb = &(M3[i]);
		for (j = 0; j < N; j++) {
			*pa += *pb;
			pa += N;
			pb += N;
		}
	}
	free(M1);
	free(M3);
	return M2;
}
