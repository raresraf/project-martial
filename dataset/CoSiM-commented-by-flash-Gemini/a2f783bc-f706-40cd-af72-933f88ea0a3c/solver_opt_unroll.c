
#include "utils.h"

void wait_for_input(const char *msg)
{
	printf("%s\n", msg);
	getc(stdin);
}


double* my_solver(int N, double *A, double* B)
{
	double *res = calloc(N * N, sizeof(*res));
	double *tmp1 = calloc(N * N, sizeof(*res));
	double *tmp2 = calloc(N * N, sizeof(*res));
	int blockSize = 40;

	if (res == NULL) {
		printf("Malloc error\n");
		exit(-1);
	}

	if (tmp1 == NULL) {
		printf("Malloc error\n");
		exit(-1);
	}

	if (tmp2 == NULL) {
		printf("Malloc error\n");
		exit(-1);
	}

	
	for (register int bi = 0; bi < N; bi += blockSize) {
		
		for (register int bj = bi; bj < N; bj += blockSize) {
			for (register int bk = 0; bk < N; bk += blockSize) {

				
				if (bk < bi + blockSize) {
					for (register int i = 0; i < blockSize; i++) {
						register double *orig_c = &tmp1[(bi + i) * N + bj];
						register double *orig_a = &B[(bi + i) * N + bk];
						register double *orig_b = &B[bj * N + bk];

						register double *orig_c2 = &tmp2[(bi + i) * N + bj];
						register double *orig_a2 = &A[bk * N + bi + i];
						register double *orig_b2 = &A[bk * N + bj];

						for (register int k = 0; k < blockSize; k++) {
							register double *pc1 = orig_c;
							register double *pa1 = orig_a + k;
							register double *pb1 = orig_b + k;

							
							if (bi + i >= bk + k) {
								register double *pa2 = orig_a2 + k * N;
								register double *pc2 = orig_c2;
								register double *pb2 = orig_b2 + k * N;
								
									
								
								
								
								

								
								

								
								
								
								

								
								
								
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc2 += *pa2 * *pb2;
								pc2++;
								pb2++;
							} else {
								
								
								
									 
								

								
								
								
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
								*pc1 += *pa1 * *pb1;
								pc1++;
								pb1 += N;
							}
						}
					}
				} else {
					for (register int i = 0; i < blockSize; i++) {
						register double *orig_c = &tmp1[(bi + i) * N + bj];
						register double *orig_a = &B[(bi + i) * N + bk];
						register double *orig_b = &B[bj * N + bk];

						for (register int k = 0; k < blockSize; k++) {
							register double *pc1 = orig_c;
							register double *pa1 = orig_a + k;
							register double *pb1 = orig_b + k;

							
							
							
								 
							

							
							
							

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;

							*pc1 += *pa1 * *pb1;

							pc1++;
							pb1 += N;
						}
					}
				}
			}
		}
	}

	
	for (int i = 0; i < N ; i++) {
		for (int j = 0; j < N; j++) {
			tmp2[j * N + i] = tmp2[i * N + j];
			tmp1[j * N + i] = tmp1[i * N + j];
		}
	}

	
	for (register int bi = 0; bi < N; bi += blockSize) {
		for (register int bj = 0; bj < N; bj += blockSize) {
			for (register int bk = bi; bk < N; bk += blockSize) {
				for (register int i = 0; i < blockSize; i++) {
					register double *orig_c = &res[(bi + i) * N + bj];
					register double *orig_a = &A[(bi + i) * N + bk];
					register double *orig_b = &tmp1[bk * N + bj];
					register double start_k = bi + i - bk < 0 ? 0 : bi + i - bk;

					
					for (register int k = start_k; k < blockSize; k++) {
						register double *pc = orig_c;
						register double *pa = orig_a + k;
						register double *pb = orig_b + k * N;

						
						
						
								 
						

						
						
						
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
						*pc += *pa * *pb;

						pc++;
						pb++;
					}
				}
			}
		}
	}

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) { 
			res[i * N + j] += tmp2[i * N + j];
		}
	}

	free(tmp1);
	free(tmp2);

	return res;
}
