/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem from Codeforces.
 *
 * @details The problem involves a 5x5 matrix containing a single '1' and 24 '0's.
 * The goal is to find the minimum number of moves (swapping adjacent rows or
 * columns) to move the '1' to the center of the matrix, which is at
 * position (3,3) (or 0-indexed at (2,2)).
 *
 * @algorithm
 * The minimum number of moves corresponds to the Manhattan distance between the
 * current position of the '1' and the center of the matrix. The Manhattan
 * distance between (r1, c1) and (r2, c2) is `|r1 - r2| + |c1 - c2|`.
 * 1. Iterate through the 5x5 matrix, keeping track of the row and column indices.
 * 2. When the '1' is found at position (i, j).
 * 3. Calculate the distance to the center (2, 2): `abs(i - 2) + abs(j - 2)`.
 * 4. Since there's only one '1', this distance is the final answer.
 *
 * @complexity
 * Time Complexity: O(1), because the matrix size is fixed at 5x5 (25 cells).
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int moves_needed = 0; // Will store the final answer.
  int number_input;   // To store the value of each cell as it's read.

  /**
   * @block
   * @brief Iterate through the 5x5 matrix to find the '1'.
   * @param i The current row index (0-4).
   * @param j The current column index (0-4).
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> number_input;

      if (number_input == 1) {
        /**
         * @brief Calculate the Manhattan distance to the center (2, 2).
         * abs(j - 2) is the number of column moves required.
         * abs(i - 2) is the number of row moves required.
         */
        moves_needed = abs(j - 2) + abs(i - 2);
        // Since we found the only '1', we could break here, but continuing
        // the loops has no impact on the result.
      }
    }
  }

  // Print the calculated minimum number of moves.
  cout << moves_needed << "\n";
}

// A sample compilation command left by the author.
// g++ -std=c++11 -O2 -Wall 263A.cpp -o out
