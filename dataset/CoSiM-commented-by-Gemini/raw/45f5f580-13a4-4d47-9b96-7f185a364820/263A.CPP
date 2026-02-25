/**
 * @file raw/45f5f580-13a4-4d47-9b96-7f185a364820/263A.CPP
 * @brief Solution to Codeforces problem 263A - Beautiful Matrix.
 *
 * @algorithm
 * The problem requires finding the minimum number of moves to bring the single
 * digit '1' in a 5x5 matrix to the center at (3,3) (using 1-based indexing).
 * A "move" is defined as swapping an adjacent row or column.
 *
 * The minimum number of moves required is equivalent to the Manhattan distance
 * from the current position of the '1' to the center of the matrix. The
 * Manhattan distance between two points (x1, y1) and (x2, y2) is calculated as:
 *   `|x1 - x2| + |y1 - y2|`
 *
 * In this case, the target position is (3,3).
 *
 * The algorithm is as follows:
 * 1. Read the 5x5 matrix elements from standard input.
 * 2. While reading, find the 0-based coordinates (i, j) of the element '1'.
 * 3. Convert these 0-based coordinates to 1-based coordinates (`x = i + 1`, `y = j + 1`).
 * 4. Calculate the Manhattan distance between the found coordinates (x, y) and
 *    the center (3, 3): `abs(x - 3) + abs(y - 3)`.
 * 5. Print the resulting distance.
 *
 * @complexity
 * - Time Complexity: O(1), since the matrix size is fixed at 5x5. The loops
 *   run a constant number of times (25 iterations).
 * - Space Complexity: O(1), as the matrix `a` has a fixed size.
 */
#include <iostream>
#include <math.h> // For abs(), though <cmath> is preferred in C++.
using namespace std;

int main() {
  // a: 2D array to store the 5x5 matrix.
  // x, y: to store the 1-based coordinates of the '1'.
  int a[5][5], x, y;

  /**
   * @brief Read the matrix and find the coordinates of the '1'.
   */
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      cin >> a[i][j];
      // When the '1' is found, store its 1-based coordinates.
      if (a[i][j] == 1) {
        x = i + 1;
        y = j + 1;
      }
    }
  }
  // Calculate and print the Manhattan distance to the center (3,3).
  cout << abs(x - 3) + abs(y - 3) << endl;
  
  return 0;
}
