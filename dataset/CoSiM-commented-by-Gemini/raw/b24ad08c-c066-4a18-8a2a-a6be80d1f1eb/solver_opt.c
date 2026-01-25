
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	printf("OPT SOLVER\n");

	register size_t i;
	register size_t j;
	register size_t k;

	
	double *C = calloc(N * N, sizeof(double));

	if (C == NULL) {
		perror("Calloc C");
		exit(EXIT_FAILURE);
	}

	
	double *AB = calloc(N * N, sizeof(double));

	if (AB == NULL) {
		perror("Calloc AB");
		exit(EXIT_FAILURE);
	}

	
	double *P1 = calloc(N * N, sizeof(double));

	if (P1 == NULL) {
		perror("Calloc P1");
		exit(EXIT_FAILURE);
	}

	
	double *P2 = calloc(N * N, sizeof(double));

	if (P2 == NULL) {
		perror("Calloc P2");
		exit(EXIT_FAILURE);
	}

	
	for (i = 0; i < N; ++i) {
		register double *orig_pa = &A[i * N];

		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa;
			register double *pb = &B[j];
			register double sum = 0.0;

			pa += i;
			pb += N * i;

			for (k = i; k < N; ++k) {
				sum += *pa * *pb;

				pa++;
				pb += N;
			}

			AB[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; ++i) {
		register double *orig_pa = &AB[i * N];

		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa;
			register double *pb = &B[j * N];
			register double sum = 0.0;

			for (k = 0; k < N; k += 4) {
				sum += *pa * *pb;

				pa++;
				pb++;

				sum += *pa * *pb;

				pa++;
				pb++;

				sum += *pa * *pb;

				pa++;
				pb++;

				sum += *pa * *pb;

				pa++;
				pb++;
			}

			P1[i * N + j] = sum;
		}
	}

	
	for (i = 0; i < N; ++i) {
		register double *orig_pa = &A[i];

		for (j = 0; j < N; ++j) {
			register double *pa = orig_pa;
			register double *pb = &A[j];
			register double sum = 0.0;

			for (k = 0; k <= i && k <= j; ++k) {
				sum += *pa * *pb;

				pa += N;
				pb += N;
			}

			P2[i * N + j] = sum;
		}
	}

	
	register double *pc = C;
	register double *pp1 = P1;
	register double *pp2 = P2;

	for (i = 0; i < N * N; i += 4) {
		*pc = *pp1 + *pp2;

		pc++;
		pp1++;
		pp2++;

		*pc = *pp1 + *pp2;

		pc++;
		pp1++;
		pp2++;

		*pc = *pp1 + *pp2;

		pc++;
		pp1++;
		pp2++;

		*pc = *pp1 + *pp2;

		pc++;
		pp1++;
		pp2++;
	}

	free(AB);
	free(P1);
	free(P2);

	return C;
}
