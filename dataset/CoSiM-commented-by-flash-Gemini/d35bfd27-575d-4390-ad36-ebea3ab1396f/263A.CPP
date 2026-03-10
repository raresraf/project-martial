/**
 * @file 263A.CPP
 * @brief This program solves the "Beautiful Matrix" problem (Codeforces 263A).
 *   It determines the minimum number of moves required to bring the number '1' to the center
 *   of a 5x5 matrix (position (3,3) assuming 1-indexed rows/columns).
 *   A move consists of swapping adjacent cells (horizontally or vertically).
 *   Problem description: https://codeforces.com/problemset/problem/263/A
 * * Algorithm: Matrix traversal and Manhattan distance calculation.
 *   The program iterates through a 5x5 grid, and once '1' is found, it calculates
 *   the Manhattan distance between its current coordinates and the center (3,3).
 * Time Complexity: O(1) because the matrix size is fixed at 5x5, leading to a constant number of operations.
 * Space Complexity: O(1) for a few integer variables.
 */
#include <cmath>    // Required for std::abs.
#include <iostream> // Required for std::cin, std::cout.
using namespace std;

/**
 * @brief Main function to find the '1' in a 5x5 matrix and calculate moves to center.
 * Functional Utility: Reads the 5x5 matrix, identifies the position of '1',
 *                     calculates the Manhattan distance to the center (3,3), and prints it.
 */
int main() {
  int a; // Variable to store each integer read from the matrix.
  /**
   * Block Logic: Outer loop iterates through rows of the 5x5 matrix (1-indexed).
   * Pre-condition: `i` represents the current row number, from 1 to 5.
   * Invariant: Each iteration processes one row of the matrix.
   */
  for (int i = 1; i <= 5; i++) {
    /**
     * Block Logic: Inner loop iterates through columns of the 5x5 matrix (1-indexed).
     * Pre-condition: `j` represents the current column number, from 1 to 5.
     * Invariant: Each iteration processes one cell in the current row.
     */
    for (int j = 1; j <= 5; j++) {
      // Reads the integer value for the current cell.
      cin >> a;
      /**
       * Block Logic: Checks if the current cell contains the number '1'.
       * If '1' is found, it calculates and prints the minimum moves to bring it to the center.
       * Inline: `abs(3 - i) + abs(3 - j)` calculates the Manhattan distance from the
       *         current cell (i, j) to the target center cell (3, 3).
       *         This distance represents the minimum number of horizontal and vertical swaps required.
       */
      if (a == 1)
        cout << abs(3 - i) + abs(3 - j);
    }
  }
  return 0;
}
