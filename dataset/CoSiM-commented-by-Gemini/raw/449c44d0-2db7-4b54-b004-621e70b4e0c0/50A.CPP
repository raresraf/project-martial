/**
 * @file raw/449c44d0-2db7-4b54-b004-621e70b4e0c0/50A.CPP
 * @brief Solution to Codeforces problem 50A - Domino piling.
 *
 * @algorithm
 * The problem is to find the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N.
 *
 * The total area of the board is M * N. Each domino covers an area of 2.
 * Therefore, the maximum number of non-overlapping dominoes that can fit on
 * the board is the total area of the board divided by the area of a single
 * domino.
 *
 * Maximum dominoes = floor((M * N) / 2)
 *
 * Since M and N are integers, this is equivalent to integer division: (M * N) / 2.
 *
 * The algorithm is as follows:
 * 1. Read the board dimensions, M and N.
 * 2. Calculate the total area (M * N).
 * 3. Perform integer division of the area by 2.
 * 4. Print the result.
 *
 * @complexity
 * - Time Complexity: O(1), as the solution involves a few arithmetic operations.
 * - Space Complexity: O(1), as only a few variables are used to store the dimensions.
 */
#include "stdio.h"

int main() {
  // n, m: dimensions of the rectangular board.
  int n, m;

  // Read the board dimensions from standard input.
  scanf("%d%d", &n, &m);

  // Calculate and print the maximum number of dominoes.
  // This is the total area of the board divided by the area of one domino (2).
  // Integer division automatically handles the floor operation.
  printf("%d", (n * m) / 2);

  return 0;
}
