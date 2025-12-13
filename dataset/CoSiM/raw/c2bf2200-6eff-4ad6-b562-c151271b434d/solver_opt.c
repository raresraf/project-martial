
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	double *AB = malloc(N * N * sizeof(double));
	double *ABBt = malloc(N * N * sizeof(double));
	double *AtA = malloc(N * N * sizeof(double));
	double *R = malloc(N * N * sizeof(double));
	register int i, j, k;
	register double *pab, *pa, *pb, *pbt, *pabbt, *pata, *pr; 
	register double sum;

	pab = AB;
	for (i = 0; i < N; ++i) {
		register double *orig_pa = A + i * N;
		for (j = 0; j < N; ++j) {

			pa = orig_pa + i;
			pb = B + i * N + j;
			sum = 0.0;
			
			for (k = i; k < N; ++k) {
				sum += (*pa) * (*pb);
				++pa;
				pb += N;
			}
			*pab = sum;
			++pab;
		}
	}

	pabbt = ABBt;
	for (i = 0; i < N; ++i) {
		register double *orig_pab = AB + i * N;
		for (j = 0; j < N; ++j) {
			pab = orig_pab;
			pbt = B + j * N;
			sum = 0.0;
			for (k = 0; k < N; ++k) {
				sum += (*pab) * (*pbt);
				++pab;
				++pbt;
			}
			*pabbt = sum;
			++pabbt;
		}
	}

	pata = AtA;
	for (i = 0; i < N; ++i) {
		register double *orig_pa = A;
		
		for (j = 0; j < N; ++j) {
			pa = orig_pa;
			sum = 0.0;
			
			for (k = 0; k < N; ++k) {
				if (i < k || j < k)
					break;
				sum += (*(pa + i)) * (*(pa + j));
				pa += N;
			}
			*pata = sum;
			++pata;
		}
	}

	pr = R;
	pabbt = ABBt;
	pata = AtA;
	for (i = 0; i < N * N; ++i) {
		*pr = *pabbt + *pata;
		++pr;
		++pabbt;
		++pata;
	}

	free(AB);
	free(ABBt);
	free(AtA);

	return R;	
}
