
/**
 * @file compare.c
 * @brief Utility to compare two matrices stored in binary files.
 * @details This program takes two file paths and a precision value as input. It
 * memory-maps the files, treats them as square matrices of doubles, and
 * compares them element-by-element to check if they are equal within the given
 * tolerance. This is often used to verify the correctness of a matrix
 * computation by comparing its output to a known-correct result.
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
 * @brief Checks if two double-precision numbers are within a given tolerance.
 * @param a The first value.
 * @param b The second value.
 * @param err The tolerance.
 * @return 0 if the absolute difference is within tolerance, -1 otherwise.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)

/**
 * @brief Compares two matrices from binary files using memory-mapping.
 * @param file_path1 Path to the first binary file containing a square matrix.
 * @param file_path2 Path to the second binary file containing a square matrix.
 * @param precision The maximum allowed absolute difference between corresponding elements.
 * @return 0 if the matrices are identical within the given precision, -1 on error or if they differ.
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

	/**
	 * Pre-condition: Files must be of the same size to represent matrices of
	 * the same dimensions. This is a fast initial check.
	 */
	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ\n");
		close(fd1);
		close(fd2);
		return -1;
	}

	/**
	 * Block Logic: Use memory-mapping for efficient file access.
	 * This avoids reading the entire file into a separate buffer in RAM,
	 * allowing the OS to handle paging directly from the file.
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

	// Infer matrix dimension N assuming a square matrix of doubles.
	N = sqrt(fileInfo1.st_size / sizeof(double));

	/**
	 * Block Logic: Iterate through the matrices and compare each element.
	 * Invariant: All elements checked so far are within the specified precision.
	 */
	for (i = 0; i < N; i++ ) {
		for (j = 0; j< N; j++) {
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision);
			// If a difference is found, print details and exit the loops.
			if (ret != 0) {
				printf("Matrixes differ on index [%d, %d]. Expected %.8lf got %.8lf\n",
						i, j, mat1[i * N + j], mat2[i * N + j]);
				goto done;
			}
		}
	}

done:
	// Cleanup: Unmap the memory regions and close the file descriptors.
	munmap(mat1, fileInfo1.st_size);
	munmap(mat2, fileInfo2.st_size);

	close(fd1);
	close(fd2);

	return ret;
}

/**
 * @brief Main entry point for the command-line comparison utility.
 */
int main(int argc, const char **argv)
{
	double precision;
	int ret = 0;

	// Check for the correct number of command-line arguments.
	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance\n",argv[0]);
		exit(-1);
	}

	// Parse the floating-point precision value from the command line.
	sscanf(argv[3], "%lf", &precision);

	// Perform the comparison.
	ret = cmp_files(argv[1],argv[2],precision);
	
	// Report the final result to the console.
	printf("%s %s %s %s\n", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret;
}
