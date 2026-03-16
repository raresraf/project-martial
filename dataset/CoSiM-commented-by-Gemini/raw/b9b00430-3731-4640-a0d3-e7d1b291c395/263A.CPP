/**
 * @file 263A.CPP
 * @brief Solution to the "Beautiful Matrix" problem (Codeforces 263A).
 *
 * @details
 * This program calculates the minimum number of moves required to move a single '1'
 * in a 5x5 matrix to the center position (row 3, column 3). A move consists of
 * swapping two adjacent rows or two adjacent columns.
 *
 * The minimum number of moves is equivalent to the Manhattan distance between the
 * current position of the '1' and the center of the matrix. The center of a 5x5
 * grid is at 0-indexed coordinates (2, 2). The Manhattan distance between two
 * points (r1, c1) and (r2, c2) is calculated as `|r1 - r2| + |c1 - c2|`.
 *
 * The code first iterates through the 5x5 input to find the coordinates (row, col)
 * of the '1'. Then, it calculates `|row - 2| + |col - 2|` to get the result.
 *
 * Algorithm:
 * 1. Read the 25 elements of the 5x5 matrix.
 * 2. Find and store the 0-indexed row and column of the element that is '1'.
 * 3. Calculate the Manhattan distance from these coordinates to the center (2, 2).
 * 4. Print the result.
 *
 * Time Complexity: O(1), as the input size is fixed (5x5).
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;
int main() {
  int row, col;
  int num;

  /**
   * @brief Reads the 5x5 matrix and finds the coordinates of the '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> num;
      if (num == 1) {
        row = i;
        col = j;
      }
    }
  }

  /**
   * @brief Calculates the Manhattan distance to the center (2, 2).
   * @details The expression `(row > 2 ? row - 2 : 2 - row)` is a manual
   * implementation of `abs(row - 2)` using the ternary operator. The total
   * expression calculates `abs(row - 2) + abs(col - 2)`.
   */
  int steps = (row > 2 ? row - 2 : 2 - row) + (col > 2 ? col - 2 : 2 - col);
  cout << steps;
}