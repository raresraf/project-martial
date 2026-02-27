/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem from Codeforces.
 * @details The goal is to find the minimum number of moves to bring the single '1'
 * in a 5x5 matrix to the center, which is at position (2, 2) with 0-based indexing.
 * A move consists of swapping the '1' with an adjacent cell.
 *
 * @algorithm
 * 1. Read the 5x5 matrix from standard input.
 * 2. While reading, identify the coordinates (x, y) of the cell containing the '1'.
 * 3. The minimum number of moves is the Manhattan distance between the current
 *    position (x, y) and the target center (2, 2).
 * 4. The Manhattan distance is calculated as |2 - x| + |2 - y|.
 * 5. Print the calculated distance.
 *
 * @complexity
 * Time Complexity: O(1), since the matrix size is fixed at 5x5 (25 elements).
 * The loops run a constant number of times.
 * Space Complexity: O(1), as the matrix size is constant.
 */
#include <cmath>
#include <iostream>

using namespace std;

// Represents the 5x5 grid.
int board[5][5];

int main() {
  int x, y; // To store the coordinates of the '1'.

  /**
   * @block
   * @brief Reads the matrix and finds the position of the '1'.
   * @invariant After this block, (x, y) will hold the 0-indexed coordinates
   * of the cell containing the value 1.
   */
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++) {
      cin >> board[i][j];

      if (board[i][j] == 1) {
        x = i;
        y = j;
      }
    }

  /**
   * @brief Calculate and print the Manhattan distance to the center (2, 2).
   * @details abs(2 - x) calculates the vertical distance (rows).
   *          abs(2 - y) calculates the horizontal distance (columns).
   * The sum is the minimum number of single-step moves required.
   */
  cout << abs(2 - x) + abs(2 - y) << endl;

  return 0;
}
