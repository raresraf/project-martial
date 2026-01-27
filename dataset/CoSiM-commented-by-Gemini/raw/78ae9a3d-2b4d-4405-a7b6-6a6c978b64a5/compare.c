/**
 * @file compare.c
 * @brief Utility to compare two matrices stored in binary files.
 * @details This program takes two file paths and a precision value as input.
 * It memory-maps the two files, which are expected to contain square matrices of double-precision floating-point numbers.
 * It then compares the two matrices element by element, checking if the absolute difference
 * is within the specified precision.
 *
 * Usage: ./compare <matrix_file_1> <matrix_file_2> <tolerance>
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
 * @def check_err(a, b, err)
 * @brief Macro to check if the absolute difference between two doubles is within a tolerance.
 * @param a First double value.
 * @param b Second double value.
 * @param err The tolerance for the comparison.
 * @return 0 if the values are within tolerance, -1 otherwise.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)

/**
 * @brief Compares two matrix files for near-equality.
 * @param file_path1 Path to the first matrix file.
 * @param file_path2 Path to the second matrix file.
 * @param precision The maximum allowed absolute difference between corresponding elements.
 * @return 0 if the matrices are considered equal within the given precision, -1 on error or if they differ.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2;
	double *mat1, *mat2;
	int i, j, fd1, fd2, N, ret = 0;

	// Open both files for reading.
	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	// Retrieve file stats to get their sizes.
	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	// Pre-condition: Files must be of the same size to contain equivalent matrices.
	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ\n");
		close(fd1);
		close(fd2);
		return -1;
	}

	/**
	 * Map the files into memory for direct access. This avoids explicit read calls
	 * and allows the OS to handle paging, which is efficient for large files.
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

	// Assuming a square matrix, calculate its dimension N.
	N = sqrt(fileInfo1.st_size / sizeof(double));

	/**
	 * Block Logic: Element-wise comparison of the two matrices.
	 * Invariant: All elements checked so far are within the specified precision.
	 */
	for (i = 0; i < N; i++ ) {
		for (j = 0; j< N; j++) {
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision); 
			// If a difference is found, print the details and exit the loops.
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
 * @brief Main entry point for the matrix comparison utility.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success (matrices are equal), -1 on failure or if they differ.
 */
int main(int argc, const char **argv)
{	
double precision;
	int ret = 0;

	// Validate command-line arguments.
	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance\n",argv[0]);
		exit(-1);
	}

	// Parse the precision from the command line.
	sscanf(argv[3], "%lf", &precision);

	// Call the comparison function.
	ret = cmp_files(argv[1],argv[2],precision);
	
	// Report the final result.
	printf("%s %s %s %s\n", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret;
}