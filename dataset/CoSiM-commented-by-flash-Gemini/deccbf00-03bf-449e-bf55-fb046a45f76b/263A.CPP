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

#include <algorithm> // Not explicitly used for its functions, but common header.
#include <iostream>
#include <cmath>     // Required for abs()
using namespace std;

// Functional Utility: 'b' temporarily stores the integer read from input.
int b;
// Functional Utility: 'sti' (start_i) stores the row index of the number '1'.
int sti;
// Functional Utility: 'stj' (start_j) stores the column index of the number '1'.
int stj;
// Functional Utility: 'pas' (steps/path) stores the calculated Manhattan distance.
int pas;

int main() {
  /**
   * Block Logic: Iterates through rows of a 5x5 matrix to find the position of '1'.
   * Invariant: 'i' represents the current row number (1-indexed) from 1 to 5.
   */
  for (int i = 1; i < 6; i++) {
    /**
     * Block Logic: Iterates through columns of the current row to find '1'.
     * Invariant: 'j' represents the current column number (1-indexed) from 1 to 5.
     */
    for (int j = 1; j < 6; j++) {
      // Block Logic: Reads an integer from standard input into 'b'.
      cin >> b;
      /**
       * Block Logic: Checks if the read integer is '1'. If so, stores its coordinates.
       * Pre-condition: 'b' holds the value of the current cell.
       * Invariant: 'sti' and 'stj' will hold the 1-indexed coordinates of '1'.
       */
      if (b == 1) {
        sti = i;
        stj = j;
      }
    }
  }
  // Functional Utility: Calculates the Manhattan distance from the position of '1'
  //                     (sti, stj) to the center (3,3) of the 5x5 matrix.
  pas = abs(sti - 3) + abs(stj - 3);
  // Output the calculated minimum number of moves ('pas').
  cout << pas;
}
