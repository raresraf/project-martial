/**
 * @file compare.c
 * @brief A utility for comparing two matrices stored in binary files for approximate equality.
 * @details This program takes two file paths and a tolerance value as input. It memory-maps the files,
 * treats them as square matrices of doubles, and compares them element-by-element to check if
 * the absolute difference is within the specified tolerance.
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
 * @brief Checks if the absolute difference between two double-precision numbers is within a tolerance.
 * @param a The first number.
 * @param b The second number.
 * @param err The tolerance (precision).
 * @return 0 if `|a - b| <= err`, otherwise -1.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)


/**
 * @brief Compares two matrices stored in binary files.
 * @param file_path1 Path to the first binary file containing a matrix.
 * @param file_path2 Path to the second binary file containing a matrix.
 * @param precision The maximum allowed absolute difference for each pair of elements.
 * @return 0 if the matrices are considered equal within the given precision, -1 otherwise.
 *
 * @details The function performs the following steps:
 * 1. Opens both files and checks if their sizes are identical.
 * 2. Memory-maps both files into memory as read-only, shared mappings.
 * 3. Calculates the matrix dimension N, assuming they are square matrices of doubles.
 * 4. Iterates through each element of the matrices and compares them using the `check_err` macro.
 * 5. If a difference is found, it prints the location and values and exits the loop.
 * 6. Unmaps the memory and closes the files before returning.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2;
	double *mat1, *mat2;
	int i, j, fd1, fd2, N, ret = 0;

	// Open both files for reading.
	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	// Retrieve file stats to check their sizes.
	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	// Pre-condition: Files must have the same size to contain equivalent matrices.
	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ
");
		close(fd1);
		close(fd2);
		return -1;
	}

	/**
	 * Block Logic: Memory-map the first file.
	 * This avoids reading the entire file into a separate buffer, providing efficient access.
	 */
	mat1 = (double*) mmap(0, fileInfo1.st_size, PROT_READ, MAP_SHARED, fd1, 0);
	if (mat1 == MAP_FAILED)
	{
		close(fd1);
		close(fd2);
		printf("Error mmapping the first file");
		return -1;
	}

	/**
	 * Block Logic: Memory-map the second file.
	 */
	mat2 = (double*) mmap(0, fileInfo2.st_size, PROT_READ, MAP_SHARED, fd2, 0);
	if (mat2 == MAP_FAILED)
	{
		munmap(mat1, fileInfo1.st_size);
		close(fd1);
		close(fd2);
		printf("Error mmapping the second file");
		return -1;
	}

	// Assumption: The file contains a square matrix of doubles. Calculate its dimension N.
	N = sqrt(fileInfo1.st_size / sizeof(double));

	/**
	 * Block Logic: Perform element-wise comparison of the two matrices.
	 * The loop terminates as soon as a pair of elements is found to be outside the tolerance.
	 */
	for (i = 0; i < N; i++ ) {
		for (j = 0; j< N; j++) {
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision); 
			// Invariant: 'ret' is 0 if all elements checked so far are within tolerance.
			if (ret != 0) {
				printf("Matrixes differ on index [%d, %d]. Expected %.8lf got %.8lf
",
						i, j, mat1[i * N + j], mat2[i * N + j]);
				goto done;
			}
		}
	}

done:
	// Cleanup: Unmap memory and close file descriptors.
	munmap(mat1, fileInfo1.st_size);
	munmap(mat2, fileInfo2.st_size);

	close(fd1);
	close(fd2);

	return ret;
}

/**
 * @brief Main entry point for the matrix comparison utility.
 * @param argc Argument count.
 * @param argv Argument vector. Expects 3 arguments: path to matrix 1, path to matrix 2, and tolerance.
 * @return 0 on success (matrices are equal), -1 on failure.
 */
int main(int argc, const char **argv)
{    
	double precision;
	int ret = 0;

	// Validate command-line arguments.
	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance
",argv[0]);
		exit(-1);
	}

	// Parse the tolerance value from the command line.
	sscanf(argv[3], "%lf", &precision);

	// Perform the comparison.
	ret = cmp_files(argv[1],argv[2],precision);
	
	// Report the final result to standard output.
	printf("%s %s %s %s
", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret;
}
