
#include <bits/stdc++.h>
using namespace std;

/**
 * @file 263A.CPP
 * @brief Solution to Codeforces problem 263A, "Beautiful Matrix".
 * @details The program calculates the minimum number of moves to make a 5x5
 * matrix "beautiful". A beautiful matrix has a single '1' at its center
 * (row 3, column 3). A move consists of swapping two adjacent rows or columns.
 *
 * Algorithm:
 * The problem is equivalent to finding the Manhattan distance between the current
 * position of the '1' and the center of the matrix (3,3). The program first
 * iterates through the 5x5 matrix to find the coordinates (row `x`, column `y`)
 * of the '1'. It then calculates the number of row moves needed as `abs(3 - x)`
 * and the number of column moves needed as `abs(3 - y)`. The sum of these two
 * values is the minimum total moves required. The code uses 1-based indexing
 * for the matrix.
 *
 * Time Complexity: O(1), as the matrix size is fixed at 5x5. The loops run a
 * constant number of times (25).
 * Space Complexity: O(1), for the 5x5 matrix storage.
 */
int main() {
  // `ara` stores the 5x5 matrix. A 6x6 array is used for 1-based indexing.
  // `x` and `y` will store the 1-based coordinates of the '1'.
  int ara[6][6], i, j, x, y;

  /**
   * @brief Read the matrix and find the coordinates of the '1'.
   */
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> ara[i][j];
      if (ara[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  }

  /**
   * @brief Calculate the Manhattan distance to the center (3,3).
   * This is done by finding the absolute difference for row and column
   * coordinates.
   */
  // This is a verbose way of calculating `x = abs(3 - x)`.
  if (3 - x < 0)
    x = -1 * (3 - x);
  else
    x = 3 - x;
  
  // This is a verbose way of calculating `y = abs(3 - y)`.
  if (3 - y < 0)
    y = -1 * (3 - y);
  else
    y = 3 - y;

  // The total moves is the sum of row moves and column moves.
  cout << x + y << endl;
}
