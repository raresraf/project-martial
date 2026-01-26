
#include "utils.h"

int mini(int a, int b) {
	return a < b ? a : b;
}

void transpose(double *At, double *Bt, double *A, double *B, int N) {
	for (register int i = 0; i < N; ++i) {
		register double *A_line = A + i * N;
		register double *B_line = B + i * N;

		register double *At_col = At + i;
		register double *Bt_col = Bt + i;

		for (register int j = 0; j < N; ++j) {
			*At_col = *A_line;
			*Bt_col = *B_line;

			At_col += N;
			Bt_col += N;
			++A_line;
			++B_line;
		}
	}
}

void multiply_AB(double *AB, double *A, double *Bt, int N) {
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			register double sum = 0;
			register double *Aptr = A + i * N + i;
			register double *Bptr = Bt + j * N + i;

			for (int k = i; k < N; ++k) {
				sum += (*Aptr) * (*Bptr);

				++Aptr;
				++Bptr;
			}
			AB[i * N + j] = sum;
		}
	}
}

void multiply_ABBt(double *ABBt, double *AB, double *B, int N) {
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			register double sum = 0;
			register double *Aptr = AB + i * N;
			register double *Bptr = B + j * N;

			for (register int k = 0; k < N; ++k) {
				sum += (*Aptr) * (*Bptr);
				++Aptr;
				++Bptr;
			}
			ABBt[i * N + j] = sum;
		}
	}
}

void multiply_AtA(double *AtA, double *At, int N) {
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			register double sum = 0;
			register double *Aptr = At + i * N;
			register double *Bptr = At + j * N;

			for (register int k = 0; k <= mini(i, j); ++k) {
				sum += (*Aptr) * (*Bptr);
				++Aptr;
				++Bptr;
			}
			AtA[i * N + j] = sum;
		}
	}
}

void addition(double *C, double *A, double *B, int N) {
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			C[i * N + j] = *A + *B;
			++A;
			++B;
		}
	}
}

double* my_solver(int N, double *A, double* B) {
	double *C = calloc(N * N, sizeof(double));
	if (!C) {
		exit(-1);
	}

	double *AB = calloc(N * N, sizeof(double));
	if (!AB) {
		exit(-1);
	}

	double *ABBt = calloc(N * N, sizeof(double));
	if (!ABBt) {
		exit(-1);
	}

	double *AtA = calloc(N * N, sizeof(double));
	if (!AtA) {
		exit(-1);
	}

	double *At = calloc(N * N, sizeof(double));
	if (!At) {
		exit(-1);
	}

	double *Bt = calloc(N * N, sizeof(double));
	if (!Bt) {
		exit(-1);
	}

	transpose(At, Bt, A, B, N);

	multiply_AB(AB, A, Bt, N);

	multiply_ABBt(ABBt, AB, B, N);

	multiply_AtA(AtA, At, N);

	addition(C, ABBt, AtA, N);

	free(AB);
	free(ABBt);
	free(AtA);
	free(At);
	free(Bt);

	return C;
}
