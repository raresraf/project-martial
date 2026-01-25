/**
 * @file compare.c
 * @brief Compares two binary files containing square matrices of doubles for equality
 *        within a specified precision.
 *
 * This utility takes two file paths and a tolerance value as input. It memory-maps the two files,
 * treats them as square matrices of doubles, and compares them element by element.
 *
 * @b Algorithm:
 * 1.  Open both files and retrieve their sizes. If sizes differ, the files are not equal.
 * 2.  Memory-map both files into the process's address space for direct memory access.
 * 3.  Iterate through each element of the matrices, comparing the corresponding values.
 * 4.  The comparison is done using a floating-point tolerance (precision) to account for
 *     minor representation differences.
 * 5.  If any pair of elements differs by more than the specified precision, the comparison
 *     fails, and the program reports the discrepancy.
 *
 * @b Usage: ./compare <matrix_file_1> <matrix_file_2> <tolerance>
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
 * @brief Checks if the absolute difference between two double-precision numbers is within a given error margin.
 * @param a The first double value.
 * @param b The second double value.
 * @param err The tolerance for the comparison.
 * @return 0 if the values are within the tolerance, -1 otherwise.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)


/**
 * @brief Memory-maps and compares two files containing double-precision square matrices.
 *
 * @param file_path1 Path to the first binary file.
 * @param file_path2 Path to the second binary file.
 * @param precision The maximum allowed absolute difference between corresponding elements.
 * @return 0 if the matrices are identical within the given precision, -1 on error or if they differ.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2;
	double *mat1, *mat2;
	int i, j, fd1, fd2, N, ret = 0;

	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	// Pre-condition: Check if file sizes are identical. If not, matrices cannot match.
	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ\n");
		close(fd1);
		close(fd2);
		return -1;
	}

	// Memory-map the first file for efficient read-only access.
	mat1 = (double*) mmap(0, fileInfo1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
	if (mat1 == MAP_FAILED)
	{
		close(fd1);
		close(fd2);
		printf("Error mmapping the first file");
		return -1;
	}

	// Memory-map the second file.
	mat2 = (double*) mmap(0, fileInfo2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
	if (mat2 == MAP_FAILED)
	{
		munmap(mat1, fileInfo1.st_size);
		close(fd1);
		close(fd2);
		printf("Error mmapping the second file");
		return -1;
	}

	// Determine the dimension of the square matrix.
	N = sqrt(fileInfo1.st_size / sizeof(double));

	/**
	 * @brief This nested loop iterates through the matrices to compare each element.
 *
	 * Invariant: All elements up to the current (i, j) have been found to be
	 * within the specified precision.
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
	// Unmap the memory regions and close the file descriptors.
	munmap(mat1, fileInfo1.st_size);
	munmap(mat2, fileInfo2.st_size);

	close(fd1);
	close(fd2);

	return ret;
}

/**
 * @brief Main entry point for the comparison utility.
 * Parses command-line arguments and invokes the file comparison function.
 */
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