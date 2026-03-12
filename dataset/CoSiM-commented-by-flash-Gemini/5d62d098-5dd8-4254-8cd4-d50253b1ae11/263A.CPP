/**
 * @5d62d098-5dd8-4254-8cd4-d50253b1ae11/263A.CPP
 * @brief This program solves problem 263A, "Beautiful Matrix". It reads a fixed 5x5 matrix
 * where exactly one cell contains the value 1. The goal is to determine the minimum
 * number of moves (which is the Manhattan distance) required to bring the cell
 * containing 1 to the center of the matrix (position (2,2) for 0-indexed coordinates).
 *
 * Algorithm: Matrix traversal to locate a specific element, followed by Manhattan distance calculation.
 * Time Complexity: O(1) as the matrix size (5x5) is constant.
 * Space Complexity: O(1) for storing the fixed-size matrix and a few variables.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream, cmath.
using namespace std;     // Uses the standard namespace.

int main() {
  int a[5][5];        // Declares a 5x5 integer array (matrix).
  int i, j, c, d, e;  // Declares loop counters (i, j), coordinates of '1' (c, d), and result (e).
  
  /**
   * Block Logic: Reads the 5x5 matrix and finds the coordinates of the cell containing the value 1.
   * Invariant: After this nested loop, 'c' and 'd' store the 0-indexed row and column
   * of the cell containing the value 1.
   */
  for (i = 0; i < 5; i++) { // Outer loop iterates through rows (0 to 4).
    for (j = 0; j < 5; j++) { // Inner loop iterates through columns (0 to 4).
      cin >> a[i][j]; // Reads the integer value for the current cell.
      // Inline: Checks if the current cell contains the value 1.
      if (a[i][j] == 1) { 
        c = i; // Stores the row index of '1'.
        d = j; // Stores the column index of '1'.
      }
    }
  }
  
  // Block Logic: Calculates the minimum number of moves, which is the Manhattan distance.
  // The Manhattan distance between (x1, y1) and (x2, y2) is |x1 - x2| + |y1 - y2|.
  // Here, (c, d) are the coordinates of '1', and (2,2) is the center of the 5x5 matrix.
  e = abs(2 - c) + abs(2 - d); 
  cout << e; // Prints the calculated Manhattan distance.
  // The program implicitly returns 0 on successful completion of main.
}