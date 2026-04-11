/**
 * @file 263A.CPP
 * @brief This program finds the position of the number '1' in a 5x5 matrix
 * and calculates the minimum number of moves required to bring '1' to the
 * center of the matrix (position (3,3)). Moves are restricted to adjacent cells
 * (up, down, left, right).
 *
 * Algorithm: Matrix traversal and Manhattan distance calculation.
 * Time Complexity: O(R*C) where R and C are the number of rows and columns (5x5 in this case).
 * Space Complexity: O(1)
 */

#include <iostream>
#include <cmath> // Required for abs()

using namespace std;

int main() {
  // Functional Utility: 'idx' stores the row index of the number '1'.
  int idx;
  // Functional Utility: 'idy' stores the column index of the number '1'.
  int idy;
  // Functional Utility: 'temp' temporarily stores the integer read from input.
  int temp;

  /**
   * Block Logic: Iterates through rows of a 5x5 matrix to find the position of '1'.
   * Invariant: 'i' represents the current row number (1-indexed).
   */
  for (int i = 1; i < 6; i++) {
    /**
     * Block Logic: Iterates through columns of the current row to find '1'.
     * Invariant: 'j' represents the current column number (1-indexed).
     */
    for (int j = 1; j < 6; j++) {
      // Block Logic: Reads an integer from standard input.
      cin >> temp;
      /**
       * Block Logic: Checks if the read integer is '1'. If so, stores its coordinates.
       * Pre-condition: 'temp' holds the value of the current cell.
       * Invariant: 'idx' and 'idy' will hold the 1-indexed coordinates of '1'.
       */
      if (temp == 1) {
        idx = i;
        idy = j;
      }
    }
  }

  // Functional Utility: Calculates and outputs the Manhattan distance from the
  //                     position of '1' to the center (3,3) of the 5x5 matrix.
  //                     This represents the minimum moves.
  cout << (abs(3 - idx) + abs(3 - idy)) << endl;
}
