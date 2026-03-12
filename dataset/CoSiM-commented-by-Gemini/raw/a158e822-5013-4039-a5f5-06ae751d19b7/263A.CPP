/**
 * @file 263A.CPP
 * @brief Solution to the "Beautiful Matrix" problem (Codeforces 263A).
 * @details The problem requires finding the minimum number of swaps of adjacent
 *          rows or columns to move the single '1' in a 5x5 matrix to the
 *          center, which is at position (3, 3) in 1-based indexing or (2, 2)
 *          in 0-based indexing.
 *
 * @algorithm The problem can be solved by calculating the Manhattan distance
 *            between the coordinates of the '1' and the center of the matrix.
 *            The Manhattan distance |r1 - r2| + |c1 - c2| gives the minimum
 *            number of single row and column moves.
 *            This implementation finds the 0-indexed coordinates of the '1' and
 *            calculates the distance to the center (2, 2).
 *
 * @time_complexity O(1), as the matrix size is fixed at 5x5, requiring 25 reads.
 * @space_complexity O(1), as the 5x5 matrix is of constant size.
 */
#include <cmath>
#include <iostream>

using namespace std;

int main() {
  int i, j, m = 0, pr, pc;
  int a[5][5];

  /**
   * @brief Read the 5x5 matrix and find the 0-indexed coordinates of the '1'.
   */
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      cin >> a[i][j];
      if (a[i][j] == 1) {
        pr = i; // Store the row index.
        pc = j; // Store the column index.
      }
    }
  }

  /**
   * @brief Calculate the Manhattan distance to the center (2, 2).
   * abs(pr - 2) is the number of row moves.
   * abs(pc - 2) is the number of column moves.
   */
  m = abs(pr - 2) + abs(pc - 2);
  cout << m;
  return 0;
}
