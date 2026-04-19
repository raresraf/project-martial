/**
 * @raw/99113ad8-03b9-45a3-a87a-1ec710110a68/compare.c
 * @brief Memory-mapped verification utility for confirming accuracy between binary matrix outputs.
 * Algorithm: Element-wise numeric validation utilizing memory mapping to optimize file I/O for large datasets.
 * Time Complexity: $O(N^2)$ for comparing elements.
 * Space Complexity: $O(1)$ RAM usage excluding virtual memory bounds from mmap.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>

/**
 * Inline: Precision-aware equality checker avoiding false negatives from floating-point accumulation errors.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)

/**
 * @brief Compares two binary dumps of dense matrices element-by-element using mmap.
 * @param file_path1 Path to the baseline/first binary matrix file.
 * @param file_path2 Path to the candidate/second binary matrix file.
 * @param precision Configurable threshold for acceptable floating point drift.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2;
	double *mat1, *mat2;
	int i, j, fd1, fd2, N, ret = 0;

	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ\n");
		close(fd1);
		close(fd2);
		return -1;
	}

	/**
	 * Functional Utility: Employs mmap for shared memory projection, enabling
	 * efficient large-file traversal without explicit read buffer allocations.
	 */
	mat1 = (double*) mmap(0, fileInfo1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
	if (mat1 == MAP_FAILED)
	{
		close(fd1);
		close(fd2);
		printf("Error mmapping the first file");
		return -1;
	}

	mat2 = (double*) mmap(0, fileInfo2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
	if (mat2 == MAP_FAILED)
	{
		munmap(mat1, fileInfo1.st_size);
		close(fd1);
		close(fd2);
		printf("Error mmapping the second file");
		return -1;
	}

	N = sqrt(fileInfo1.st_size / sizeof(double));

	/**
	 * Block Logic: Linearly sweeps mapped memory regions performing element-wise tolerance validations.
	 * Invariant: Retains synchronization of pointer offsets over the unified spatial dimension (N * N).
	 */
	for (i = 0; i < N; i++ ) {
		for (j = 0; j< N; j++) {
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision); 
			if (ret != 0) {
				printf("Matrixes differ on index [%d, %d]. Expected %.8lf got %.8lf\n",
						i, j, mat1[i * N + j], mat2[i * N + j]);
				goto done;
			}
		}
	}

done:
	munmap(mat1, fileInfo1.st_size);
	munmap(mat2, fileInfo2.st_size);

	close(fd1);
	close(fd2);

	return ret;
}

int main(int argc, const char **argv)
{    
	double precision;
	int ret = 0;

	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance\n",argv[0]);
		exit(-1);
	}

	sscanf(argv[3], "%lf", &precision);

	ret = cmp_files(argv[1],argv[2],precision);
	
	printf("%s %s %s %s\n", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret;
}
