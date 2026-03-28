/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem from Codeforces.
 *
 * The program reads a 5x5 matrix containing a single '1' and all other '0's.
 * It calculates the minimum number of single-step moves (up, down, left, right)
 * required to move the '1' to the center of the matrix (row 3, column 3).
 *
 * @algorithm
 * 1. Read the 5x5 matrix and find the coordinates (i, j) of the element '1'.
 * 2. Calculate the Manhattan distance between the '1's current position and the
 *    center (3,3). The Manhattan distance |x1 - x2| + |y1 - y2| gives the
 *    minimum number of single-step moves on a grid.
 *
 * @time_complexity O(1), as the input matrix size is fixed at 5x5.
 * @space_complexity O(1), as the storage is a fixed-size 5x5 matrix.
 */
#include <stdio.h>

int main() {
  int a[5][5], i, j, b, c, d;
  /**
   * @block_logic This nested loop reads the 5x5 matrix from standard input.
   * When the element with value '1' is found, its 1-based coordinates
   * are stored in variables `b` (row) and `c` (column).
   */
  for (i = 0; i < 5; i++)
    for (j = 0; j < 5; j++) {
      scanf("%d", &a[i][j]);
      if (a[i][j] == 1) {
        b = i + 1;
        c = j + 1;
      }
    }
  /**
   * @block_logic Calculates the Manhattan distance from the position of '1' (b, c)
   * to the center of the matrix (3, 3).
   * This is equivalent to `abs(3 - b) + abs(3 - c)`.
   */
  if (b <= 3)
    d = 3 - b;
  else
    d = b - 3;
  if (c <= 3)
    d = d + 3 - c;
  else
    d = d + c - 3;
  printf("%d", d);
  return 0;
}