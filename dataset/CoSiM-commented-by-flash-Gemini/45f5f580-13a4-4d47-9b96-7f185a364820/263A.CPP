/**
 * @file 263A.CPP
 * @brief Solution to the "Beautiful Matrix" problem (Codeforces 263A).
 *
 * This program reads a 5x5 matrix where exactly one element is '1' and
 * the rest are '0'. It finds the coordinates of the '1' and calculates
 * the minimum number of moves (Manhattan distance) required to move
 * the '1' to the center of the matrix (position (3,3) if 1-indexed, or (2,2) if 0-indexed).
 *
 * Algorithm: Matrix traversal to find a specific element, then Manhattan distance calculation.
 * Time Complexity: O(R*C) where R and C are the number of rows and columns of the matrix (here, O(5*5) = O(25) which is constant time).
 * Space Complexity: O(R*C) to store the matrix (here, O(5*5) = O(25) which is constant space).
 */
#include <iostream>
#include <math.h>
using namespace std;

/**
 * @brief Main function to find the '1' in a 5x5 matrix and calculate its Manhattan distance to the center.
 *
 * Reads a 5x5 matrix, identifies the 1-indexed (row, column) position of the
 * element '1', and then computes the absolute difference in rows and columns
 * from the center position (3,3) to determine the minimum moves.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Declare a 5x5 integer array 'a' to store the matrix.
  // Declare integers 'x' and 'y' to store the 1-indexed row and column of the '1'.
  int a[5][5], x, y;
  /**
   * @brief Block Logic: Reads the 5x5 matrix and finds the position of '1'.
   *
   * Iterates through each element of the 5x5 matrix. When '1' is found, its
   * 1-indexed row ('x') and column ('y') are stored.
   * Pre-condition: Input is a 5x5 matrix with exactly one '1'.
   * Invariant: After the loops, 'x' and 'y' hold the 1-indexed coordinates of '1'.
   */
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      cin >> a[i][j];
      // If the current element is '1', store its 1-indexed row and column.
      if (a[i][j] == 1) {
        x = i + 1;
        y = j + 1;
      }
    }
  }
  // Calculate and print the Manhattan distance from the position of '1' (x, y)
  // to the center of the matrix (3, 3).
  // The Manhattan distance is |x1 - x2| + |y1 - y2|.
  cout << abs(x - 3) + abs(y - 3) << endl;
  return 0;
}