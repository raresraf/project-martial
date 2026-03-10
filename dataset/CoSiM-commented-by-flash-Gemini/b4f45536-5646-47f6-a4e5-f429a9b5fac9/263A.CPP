/**
 * @file 263A.CPP
 * @brief This program solves the "Beautiful Matrix" problem (Codeforces 263A).
 *   It determines the minimum number of moves required to bring the number '1' to the center
 *   of a 5x5 matrix (position (2,2) assuming 0-indexed rows/columns).
 *   A move consists of swapping adjacent cells (horizontally or vertically).
 *   Problem description: https://codeforces.com/problemset/problem/263/A
 * * Algorithm: Matrix traversal and Manhattan distance calculation.
 *   The program iterates through a 5x5 grid, and once '1' is found, it calculates
 *   the Manhattan distance between its current coordinates (0-indexed) and the center (2,2).
 * Time Complexity: O(1) because the matrix size is fixed at 5x5, leading to a constant number of operations.
 * Space Complexity: O(1) for the small `a[5][5]` array and a few integer variables.
 */
#include <algorithm> // Included but not directly used.
#include <cctype>    // Included but not directly used.
#include <cmath>     // Required for `abs`.
#include <cstdio>    // Included but not directly used.
#include <iostream>  // Required for `std::cin`, `std::cout`.
using namespace std;

/**
 * @brief Main function to find the '1' in a 5x5 matrix and calculate moves to center.
 * Functional Utility: Reads the 5x5 matrix, identifies the position of '1',
 *                     calculates the Manhattan distance to the center (2,2), and prints it.
 */
int main() {
  int a[5][5]; // 2D array to store the 5x5 matrix.
  int x, y;    // `x`: row index of '1', `y`: column index of '1'.
  int move;    // Stores the calculated minimum number of moves.

  /**
   * Block Logic: Nested loops to iterate through the 5x5 matrix.
   * Pre-condition: Matrix dimensions are fixed at 5x5.
   * Invariant: `a[i][j]` is populated, and `x`, `y` store the coordinates of '1' once found.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      // Reads the integer value for the current cell.
      cin >> a[i][j];
      /**
       * Conditional Logic: Checks if the current cell contains the number '1'.
       * If '1' is found, its 0-indexed row and column are stored in `x` and `y`.
       */
      if (a[i][j] == 1) {
        x = i; // Stores the row index of '1'.
        y = j; // Stores the column index of '1'.
      }
    }
  }
  // Functional Utility: Calculates the Manhattan distance from the position of '1' (x, y)
  // to the center of the 5x5 matrix (0-indexed (2,2)).
  // Manhattan distance = |x1 - x2| + |y1 - y2|.
  move = abs(2 - x) + abs(2 - y);
  // Block Logic: Prints the calculated minimum number of moves followed by a newline.
  cout << move << endl;
}
