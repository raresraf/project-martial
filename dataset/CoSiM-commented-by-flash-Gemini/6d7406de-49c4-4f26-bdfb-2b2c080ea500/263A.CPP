/**
 * @file 263A.CPP
 * @brief Solution to a competitive programming problem: "Beautiful Matrix".
 * Finds the position of the number '1' in a 5x5 matrix and calculates the
 * minimum number of moves (Manhattan distance) required to bring it to the
 * center cell (3,3).
 * Algorithm: Matrix traversal to locate '1', then Manhattan distance calculation.
 * Time Complexity: O(R*C) where R and C are the dimensions of the matrix (5x5).
 * Space Complexity: O(R*C) to store the matrix.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to solve the Beautiful Matrix problem.
 * Reads a 5x5 matrix to find the coordinates of the number '1'.
 * Then calculates and prints the Manhattan distance from '1' to the center (3,3).
 */
int main() {
  int ara[6][6], i, j, x, y; // Declare a 2D array for the matrix and variables for coordinates.
  /**
   * Block Logic: Reads the 5x5 matrix and finds the coordinates of the number '1'.
   * The matrix is 1-indexed (from 1 to 5) for convenience.
   * Pre-condition: 'ara' is a 6x6 integer array, 'x' and 'y' are uninitialized.
   * Invariant: After loops, 'x' and 'y' hold the row and column of '1'.
   */
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> ara[i][j]; // Read the element at current position.

      /**
       * Block Logic: Stores the coordinates if the current element is '1'.
       * This effectively finds the single '1' in the matrix.
       */
      if (ara[i][j] == 1) {
        x = i; // Store row index.
        y = j; // Store column index.
      }
    }
  }
  /**
   * Block Logic: Calculates the absolute difference for the row coordinate.
   * This determines the vertical moves needed to reach row 3.
   * Pre-condition: 'x' holds the row of '1'.
   * Invariant: 'x' is updated to be the absolute difference from row 3.
   */
  if (3 - x < 0)
    x = -1 * (3 - x); // If (3-x) is negative, take its absolute value.
  else
    x = 3 - x; // Otherwise, it's already positive or zero.
  /**
   * Block Logic: Calculates the absolute difference for the column coordinate.
   * This determines the horizontal moves needed to reach column 3.
   * Pre-condition: 'y' holds the column of '1'.
   * Invariant: 'y' is updated to be the absolute difference from column 3.
   */
  if (3 - y < 0)
    y = -1 * (3 - y); // If (3-y) is negative, take its absolute value.
  else
    y = 3 - y; // Otherwise, it's already positive or zero.
  cout << x + y << endl; // The sum of absolute differences is the Manhattan distance.
}
