
#include "utils.h"


double* my_solver(int N, double *A, double* B) {
	double *AB;
	double *first;
	double *second;
	double *C;
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
