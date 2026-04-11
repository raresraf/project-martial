/**
 * @file 263A.CPP
 * @brief This program searches for the number '1' in a 5x5 matrix read from input.
 * Upon finding '1', it calculates and immediately prints the minimum number of moves
 * required to bring '1' to the center of the matrix (position (3,3)). Moves are
 * restricted to adjacent cells (up, down, left, right).
 *
 * Algorithm: Matrix traversal and Manhattan distance calculation.
 * Time Complexity: O(R*C) where R and C are the number of rows and columns (5x5 in this case).
 * Space Complexity: O(1)
 */

// Implementation.
#include <bits/stdc++.h>

using namespace std;

int main() {
  // Functional Utility: 'x' temporarily stores the integer read from input.
  int x = 0;
  /**
   * Block Logic: Iterates through rows of a 5x5 matrix.
   * Invariant: 'i' represents the current row number (1-indexed) from 1 to 5.
   */
  for (int i = 1; i <= 5; ++i) {
    /**
     * Block Logic: Iterates through columns of the current row.
     * Invariant: 'j' represents the current column number (1-indexed) from 1 to 5.
     */
    for (int j = 1; j <= 5; ++j) {
      // Block Logic: Reads an integer from standard input into 'x'.
      cin >> x;
      /**
       * Block Logic: Checks if the read integer is '1'.
       * Pre-condition: 'x' holds the value of the current cell.
       */
      if (x == 1) {
        // Functional Utility: Calculates and outputs the Manhattan distance from the
        //                     position of '1' (i,j) to the center (3,3) of the 5x5 matrix.
        //                     This represents the minimum moves.
        cout << abs(i - 3) + abs(j - 3) << endl;
      }
    }
  }
  return 0;
}
