/**
 * @file 263A.CPP
 * @brief Calculates the minimum number of moves to bring a '1' to the center of a 5x5 matrix.
 *
 * Problem Name: Beautiful Matrix (Codeforces 263A)
 * Algorithm: Manhattan distance. Locates the single '1' in the matrix and calculates its Manhattan distance
 *            to the center cell (index 2,2 for a 0-indexed 5x5 matrix).
 * Time Complexity: O(R * C) which is O(1) for a fixed 5x5 matrix.
 * Space Complexity: O(1) as it uses a fixed number of variables.
 */
#include <bits/stdc++.h> // Includes standard C++ libraries, e.g., <iostream>, <cmath> for abs.

using namespace std;

/**
 * @brief Main function to read the matrix and calculate moves to center the '1'.
 * @pre Input consists of 25 integers (0 or 1) representing a 5x5 matrix, with exactly one '1'.
 * @post Prints the minimum number of moves required to bring the '1' to the center.
 */
int main() {
  int ans = 0; // Variable to store the minimum moves (Manhattan distance).
  int number_input; // Variable to store the integer read from the matrix.

  /**
   * Block Logic: Iterates through each cell of the 5x5 matrix.
   * Invariant: `i` represents the current row index, `j` represents the current column index.
   *            When `number_input` is 1, it implies the '1' has been found.
   */
  for (int i = 0; i < 5; i++) { // Outer loop for rows (0 to 4).
    for (int j = 0; j < 5; j++) { // Inner loop for columns (0 to 4).
      // Functional Utility: Reads the integer value for the current cell.
      cin >> number_input;

      /**
       * Block Logic: If the current cell contains '1', calculate Manhattan distance to the center.
       * Pre-condition: `number_input` is the value of the current cell.
       * Invariant: If `number_input` is 1, `ans` is updated with the Manhattan distance.
       *            The target center is (2,2) for 0-indexed matrix.
       */
      if (number_input == 1) { // Checks if the '1' is found.
        // Calculates Manhattan distance: |current_row - target_row| + |current_col - target_col|.
        ans += abs(j - 2) + abs(i - 2);
      }
    }
  }

  // Functional Utility: Prints the calculated minimum moves.
  cout << ans << "\n"; // Prints the result followed by a newline.

  // Indicates successful program execution.
  return 0;
}

// g++ -std=c++11 -O2 -Wall 263A.cpp -o out