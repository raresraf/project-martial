/*
 * Module: @raw/7d87b802-1fe0-4a7d-911d-4c5f132c4e4b/solver_opt.c
 * Purpose: Optimized matrix solver using loop unrolling and pointer arithmetic.
 */
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");
	register double *e = malloc(N * N * sizeof(double));
	register double *d = malloc(N * N * sizeof(double));
	register double *c = calloc(N * N , sizeof(double));


	register double *orig_pc, *orig_pd, *orig_pa, *orig_pb, *orig_pe;
	register double *pc, *pa;

	
	// Pre-condition: A and B are N*N matrices. Invariant: c holds partial matrix products.
	for(register int i = 0; i < N; i++) {
		orig_pc = &c[i * N];
		orig_pa = &A[i * N] + i;
		orig_pb = &B[i * N];

		for (register int k = i; k < N; k++) {
			pc = orig_pc;
			for (register int j = 0; j < N; j++) {
				*pc += (*orig_pa) * *orig_pb;
				pc ++;
				orig_pb ++;
			}
			orig_pa++;
		}
	}

	// Pre-condition: c and B are populated. Invariant: d is filled with the result.
	for(register int i = 0; i < N; 	i++) {
		orig_pc = &c[i * N];
		orig_pd = &d[i * N];

		for(register int j = 0; j < N; j++) {
			pc = orig_pc;
			orig_pb = &B[j * N];
			register double suma = 0;

			for (register int k = 0; k < N; k++) {
				suma += (*pc) * (*orig_pb);
				pc ++;
				orig_pb ++;
			}

			*orig_pd = suma;
			orig_pd ++;
		}
	}

	// Pre-condition: d and A are populated. Invariant: e is filled with the final result.
	for(register int i = 0; i < N; 	i++) {
		orig_pd = &d[i * N];
		orig_pa = &A[0];
		orig_pe = &e[i * N];


		for(register int j = 0; j < N; j++) {
			register double suma = 0;
			pa = orig_pa;
			for (register int k = 0; k <= i; k++) {
				suma += *(pa + i) *  *(pa + j);
				pa += N;
			}

			suma += *orig_pd;
			*orig_pe = suma;
			orig_pd ++;
			orig_pe ++;
		}
	}
	free(d);
	free(c);
	return e;	
}