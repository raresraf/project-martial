/**
 * @raw/8a7b5be2-e828-4ed1-add7-9ff564200fc3/solver_opt.c
 * @brief Optimized dense matrix multiplication solver computing C = (A * B) * B^T + A^T * A.
 * Algorithm: Matrix multiplication employing pointer arithmetic and CPU registers for localized sum aggregation.
 * Time Complexity: $O(N^3)$
 * Space Complexity: $O(N^2)$ tracking internal partial product evaluations.
 */
#include "utils.h"

double* my_solver(int N, double *A, double* B) {
	double *AB;
	double *first;
	double *second;
	double *C;

	/**
	 * Functional Utility: Initializes dynamically allocated matrix buffers and caches for intermediate outputs.
	 */
	AB = calloc(N * N, sizeof(double));
	if(AB == NULL){
		exit(-1);
	}
	first = calloc(N * N, sizeof(double));
	if(first == NULL){
		exit(-1);
	}
	second = calloc(N * N, sizeof(double));
	if(second == NULL){
		exit(-1);
	}
	C = calloc(N * N, sizeof(double));
	if(C == NULL){
		exit(-1);
	}
    
	register int i, j, k = 0;
	
	/**
	 * Block Logic: Computes AB = A * B.
	 * Optimization: Employs explicit pointer tracking to evade multiplication overheads during indexing.
	 */
	for(i = 0; i < N; i++){
		register double *line = A + i * N;
		for(j = 0; j < N; j++){
			register double *cop = line + i;
			register double *colum = B + j + i * N;
			register double sum = 0;
			for(k = i; k < N; k++, cop++, colum += N){
				sum += *cop * *colum;
			}
			AB[i * N + j] = sum;
		}
	}
    
	/**
	 * Block Logic: Computes first = AB * B^T.
	 * Optimization: Accesses B simulating transposed layout, striding sequentially.
	 */
	for(i = 0; i < N; i++){
		register double *line = AB + i * N;
		for(j = 0; j < N; j++){
			register double *cop = line;
			register double *colum = B + j * N;
			register double sum = 0;
			for(k = 0; k < N; k++, cop++, colum++){
				sum += *cop * *colum;
			}
			first[i * N + j] = sum;
		}
	}
	
	/**
	 * Block Logic: Computes second = A^T * A.
	 * Optimization: Strides the internal pointer arrays vertically while bounding `k` sequentially.
	 */
	for(i = 0; i < N; i++){
		register double *line = A + i;
		for(j = 0; j < N; j++){
			register double *cop = line;
			register double *colum = A + j;
			register double sum = 0;
			for(k = 0; k <= i; k++, cop+=N, colum+=N){
				sum += *cop * *colum;
			}
			second[i * N + j] = sum;
		}
	}
    
	/**
	 * Block Logic: Final summation step synthesizing C = first + second.
	 * Invariant: Aggregates computed matrices into a cohesive single array.
	 */
	for(i = 0; i < N; i++){
		for(j = 0; j < N; j++){
			C[i * N + j] = first[i * N + j] + second[i * N + j];
		}
	}
	
	free(AB);
	free(first);
	free(second);
	return C;	
}
