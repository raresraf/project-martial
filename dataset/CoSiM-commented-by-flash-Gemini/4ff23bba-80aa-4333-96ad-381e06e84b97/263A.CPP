/**
 * @file 263A.CPP
 * @brief Calculates the minimum number of moves to bring a '1' to the center of a 5x5 matrix.
 *
 * Problem Name: Beautiful Matrix (Codeforces 263A)
 * Algorithm: Manhattan distance. Locates the single '1' in the matrix and calculates its Manhattan distance
 *            to the center cell (index 2,2 for a 0-indexed 5x5 matrix).
 * Time Complexity: O(R * C) which is O(1) for a fixed 5x5 matrix.
 * Space Complexity: O(R * C) for storing the 5x5 matrix.
 */
#include <cmath>    // Required for std::abs.
#include <iostream> // Required for std::cin, std::cout.

using namespace std;

// Global 2D array to store the 5x5 matrix.
int board[5][5];

/**
 * @brief Main function to read the matrix and calculate moves to center the '1'.
 * @pre Input consists of 25 integers (0 or 1) representing a 5x5 matrix, with exactly one '1'.
 * @post Prints the minimum number of moves required to bring the '1' to the center.
 */
int main() {
  int x, y; // Variables to store the row (x) and column (y) coordinates of the '1'.

  /**
   * Block Logic: Iterates through each cell of the 5x5 matrix to read its value and locate the '1'.
   * Invariant: `i` represents the current row index, `j` represents the current column index.
   *            When `board[i][j]` is 1, its coordinates are stored in `x` and `y`.
   */
  for (int i = 0; i < 5; i++) // Outer loop for rows (0 to 4).
    for (int j = 0; j < 5; j++) { // Inner loop for columns (0 to 4).
      // Functional Utility: Reads the integer value for the current cell into the board.
      cin >> board[i][j];

      /**
       * Block Logic: If the current cell contains '1', store its coordinates.
       * Pre-condition: `board[i][j]` is the value of the current cell.
       * Invariant: If `board[i][j]` is 1, `x` and `y` are updated with the current `i` and `j`.
       */
      if (board[i][j] == 1) { // Checks if the '1' is found.
        x = i;                // Stores the row index of '1'.
        y = j;                // Stores the column index of '1'.
      }
    }

  // Functional Utility: Calculates and prints the Manhattan distance to the center.
  // The center is (2,2) for a 0-indexed 5x5 matrix.
  cout << abs(2 - x) + abs(2 - y) << endl; // Prints the result.

  // Indicates successful program execution.
  return 0;
}
