/**
 * @file compare.c
 * @brief A command-line utility to compare two binary files containing floating-point square matrices.
 * @details This program uses memory-mapping to efficiently compare two files that are
 * expected to contain N x N matrices of doubles. It checks if the matrices are
 * element-wise equal within a specified numerical precision. It is likely used
 * as a verification tool to check the output of numerical solvers.
 *
 * @algorithm Memory-mapped file I/O with element-wise floating-point comparison.
 * @time_complexity O(N^2), where N is the dimension of the square matrix, as every
 * element must be compared.
 * @space_complexity O(1) in terms of heap/stack allocation, as the matrix data is
 * accessed directly from memory-mapped files, not loaded into separate buffers.
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
 * @brief Checks if the absolute difference between two doubles is within a given tolerance.
 * @return 0 if they are considered equal, -1 otherwise.
 */
#define check_err(a,b,err) ((fabs((a) - (b)) <= (err)) ? 0 : -1)

/**
 * @brief Memory-maps and compares two files containing matrix data.
 * @param file_path1 Path to the first file.
 * @param file_path2 Path to the second file.
 * @param precision The maximum allowed absolute difference between corresponding elements.
 * @return 0 if the matrices are equal within the given precision, -1 otherwise.
 */
int cmp_files(char const *file_path1, char const *file_path2, double precision) {
	struct stat fileInfo1, fileInfo2;
	double *mat1, *mat2;
	int i, j, fd1, fd2, N, ret = 0;

	// Open both files for reading.
	fd1 = open(file_path1, O_RDONLY, (mode_t)0600);
	fd2 = open(file_path2, O_RDONLY, (mode_t)0600);

	// Retrieve file metadata, primarily for size comparison.
	fstat(fd1, &fileInfo1);
	fstat(fd2, &fileInfo2);

	// Pre-condition: Files must be the same size to contain equivalent matrices.
	if(fileInfo1.st_size != fileInfo2.st_size) {
		printf("Files length differ\n");
		close(fd1);
		close(fd2);
		return -1;
	}

	// Memory-map the first file into the process's address space.
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

	// Assuming a square matrix, calculate its dimension N.
	N = sqrt(fileInfo1.st_size / sizeof(double));

	// Block Logic: Iterate through each element of the matrices for comparison.
	for (i = 0; i < N; i++ ) {
		for (j = 0; j < N; j++) {
			ret = check_err(mat1[i * N + j], mat2[i * N + j], precision); 
			if (ret != 0) {
				printf("Matrixes differ on index [%d, %d]. Expected %.8lf got %.8lf\n",
						i, j, mat1[i * N + j], mat2[i * N + j]);
				goto done; // Exit immediately upon finding a difference.
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
 * @brief Main entry point for the command-line utility.
 * @param argc Argument count.
 * @param argv Argument vector. Expects: ./compare <file1> <file2> <tolerance>
 * @return 0 on success (files are equal), -1 on failure or error.
 */
int main(int argc, const char **argv)
{	
double precision;
	int ret = 0;

	// Argument validation.
	if(argc < 4) {
		printf("Usage: %s mat1 mat2 tolerance\n",argv[0]);
		exit(-1);
	}

	// Parse the precision from command-line arguments.
	sscanf(argv[3], "%lf", &precision);

	ret = cmp_files(argv[1],argv[2],precision);
	
	printf("%s %s %s %s\n", argv[0], argv[1], argv[2], (ret == 0 ? "OK" : "Incorrect results!"));
	return ret;
}