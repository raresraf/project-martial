/**
 * @file 263A.CPP
 * @brief Calculates the minimum moves to center a '1' in a 5x5 matrix.
 *
 * This program solves the "Beautiful Matrix" problem. It reads a 5x5 matrix
 * that contains a single '1' and the rest are '0's. The goal is to find the
 * minimum number of single-step moves (up, down, left, or right) required to
 * move the '1' to the center of the matrix, which is at position (3, 3).
 * This is equivalent to calculating the Manhattan distance from the '1's
 * position to the center. The matrix is treated as 1-indexed.
 *
 * @author f2043910-25f0-4fa1-a607-7b1d831c1d5f
 *
 * @algorithm Matrix Traversal and Manhattan Distance Calculation
 * @complexity
 * Time: O(1), as the matrix size is fixed at 5x5. The loops run a constant number of times.
 * Space: O(1), for the 10x10 array to hold the 5x5 matrix.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Reads matrix elements from standard input.
 * @param Array The 2D array to populate.
 * @param r The number of rows to read.
 * @param c The number of columns to read.
 * @note The loops are 1-indexed.
 */
void read_arr(int Array[10][10], int r, int c) {
  int i, j;
  for (i = 1; i <= r; i++) {
    for (j = 1; j <= c; j++) {
      scanf("%d", &Array[i][j]);
    }
  }
}

/**
 * @struct location
 * @brief A simple structure to store 2D coordinates.
 */
struct location {
  int x; // Represents the row index.
  int y; // Represents the column index.
};

int main() {
  // A large number of variables are declared, many of which are unused.
  int temp = 0;
  location l;
  int a, b, c = 5, d, m, n, i, j, Array[10][10], m2[10][10], cnt = 0, r = 5,
            N = 5, M = 5;

  // Read the 5x5 matrix using 1-based indexing.
  read_arr(Array, r, c);

  /**
   * @brief Loop to find the location of the '1'.
   * It iterates through the matrix to find the maximum element, which is the '1',
   * and stores its coordinates in the `location` struct `l`.
   * Pre-condition: The matrix `Array` is populated.
   * Invariant: `temp` holds the max value found so far, `l.x` and `l.y` hold its location.
   */
  for (int i = 1; i <= 5; i++)
    for (int j = 1; j <= 5; j++)
      if (temp < Array[i][j]) {
        temp = Array[i][j];
        l.x = i;
        l.y = j;
      }

  /**
   * @brief Calculate and print the Manhattan distance.
   * The distance from the found location (l.x, l.y) to the center (3, 3)
   * is calculated as the sum of the absolute differences of the coordinates.
   */
  cout << fabs(l.x - 3) + fabs(l.y - 3) << endl;

  // The line `// system("pause");` is commented out and has no effect.
  return 0;
}