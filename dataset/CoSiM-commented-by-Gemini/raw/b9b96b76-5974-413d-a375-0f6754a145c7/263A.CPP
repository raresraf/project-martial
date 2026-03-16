/**
 * @file 263A.CPP
 * @brief Solution to the "Beautiful Matrix" problem (Codeforces 263A).
 *
 * @details
 * This program finds the minimum number of moves to bring the single '1' in a
 * 5x5 matrix to the center at 0-indexed coordinates (2, 2). The minimum number
 * of moves is the Manhattan distance: `|r1 - r2| + |c1 - c2|`.
 *
 * This solution efficiently finds the location of the '1' while reading the
 * input. As soon as the '1' is encountered at coordinates (i, j), it immediately
 * calculates the Manhattan distance to the center, `|i - 2| + |j - 2|`, prints
 * the result, and breaks the inner loop.
 *
 * @note After finding the '1' and printing the result, the program continues to
 * read the remaining matrix values, which is a minor inefficiency but does not
 * affect the correctness of the output.
 *
 * Algorithm:
 * 1. Iterate through the rows and columns of a 5x5 grid.
 * 2. Read each element.
 * 3. If the element is '1', calculate `abs(2 - row) + abs(2 - col)`, print it,
 *    and stop searching.
 *
 * Time Complexity: O(1), as the matrix size is fixed at 5x5.
 * Space Complexity: O(1), as the matrix size is fixed.
 */
#include <iostream>
#include <math.h> // For abs()
#include <vector>
using namespace std;
int main() {
  vector<vector<int>> input(5, vector<int>(5, 0));

  /**
   * @brief Reads matrix elements and finds the '1'.
   * @details This loop reads the matrix element by element. As soon as the '1'
   * is found, it calculates and prints the result and breaks the inner loop.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> input[i][j];
      if (input[i][j] == 1) {
        // Calculate the Manhattan distance to the center (2,2) and print it.
        cout << abs(2 - i) + abs(2 - j);
        break; // Exit the inner loop once the '1' is found.
      }
    }
  }
  return 0;
}
