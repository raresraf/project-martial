/**
 * @file 263A.CPP
 * @brief Solution for Codeforces Problem 263A - Beautiful Matrix.
 *
 * This problem involves a 5x5 matrix where exactly one cell contains the number '1',
 * and all other cells contain '0'. The goal is to find the minimum number of moves
 * required to bring the cell containing '1' to the center of the matrix (row 2, column 2,
 * assuming 0-indexed rows/columns). A move consists of shifting the '1' one cell
 * up, down, left, or right.
 *
 * Algorithm: Manhattan distance calculation (implemented iteratively).
 * The code first scans the 5x5 matrix to locate the row and column indices of the '1'.
 * Then, it iteratively moves the '1' towards the center, counting each move. This
 * iterative movement effectively calculates the Manhattan distance (sum of absolute
 * differences in row and column indices) from the '1''s initial position to the center (2,2).
 *
 * Time Complexity: O(1) because the matrix size is fixed at 5x5 (a constant).
 * More precisely, it's 25 iterations for input + up to 4 moves (max Manhattan distance).
 * Space Complexity: O(1) for storing the 5x5 matrix and a few integer variables.
 */

#include <bits/stdc++.h> // Includes most standard C++ libraries, common in competitive programming.

using namespace std; // Functional Utility: Brings all identifiers from the std namespace into the current scope.

/**
 * @brief Main function: Solves the Beautiful Matrix problem.
 * Functional Utility: Reads a 5x5 matrix, finds the position of the '1',
 * and calculates the minimum moves to bring it to the center (row 2, col 2).
 * Algorithm: Iterative calculation of Manhattan distance to center.
 */
int main() {
  // `n` is the 5x5 matrix input, `m` is an unused 1x1 matrix.
  // `inrow`, `incol` store the row and column of '1'.
  // `cnt` counts the moves.
  int n[5][5], m[1][1], inrow, incol, cnt = 0;

  // Block Logic: Read the 5x5 matrix and find the position of the '1'.
  // Precondition: Input is a 5x5 matrix with exactly one '1'.
  // Invariant: After nested loops, `inrow` and `incol` accurately store the 0-indexed position of '1'.
  for (int i = 0; i < 5; i++) { // Loop for rows.
    for (int j = 0; j < 5; j++) { // Loop for columns.
      cin >> n[i][j]; // Read matrix element.
      if (n[i][j] == 1) { // If '1' is found.
        inrow = i; // Store its row.
        incol = j; // Store its column.
      }
    }
  }

  // Block Logic: Iteratively move '1' to the center (2,2) and count moves.
  // This loop implements the Manhattan distance calculation by moving towards the target (2,2).
  // The loop `j < 25` is an upper bound on moves; `return 0` inside ensures early exit.
  // Invariant: `cnt` stores the number of moves made so far to get closer to the center.
  for (int j = 0; j < 25; j++) {
    if (incol > 2) { // If '1' is to the right of the center column.
      incol--; // Move left.
      cnt++; // Increment move count.
    } else if (incol < 2) { // If '1' is to the left of the center column.
      incol++; // Move right.
      cnt++; // Increment move count.
    } else if (incol == 2 && inrow > 2) { // If '1' is in the center column but below the center row.
      inrow--; // Move up.
      cnt++; // Increment move count.
    } else if (incol == 2 && inrow < 2) { // If '1' is in the center column but above the center row.
      inrow++; // Move down.
      cnt++; // Increment move count.
    } else if (incol == 2 && inrow == 2) { // If '1' is exactly at the center (2,2).
      cout << cnt << "\n"; // Print the total number of moves.
      return 0; // Exit program as solution is found.
    }
  }

  return 0; // Should not be reached in a valid execution if '1' is always moved to center.
}