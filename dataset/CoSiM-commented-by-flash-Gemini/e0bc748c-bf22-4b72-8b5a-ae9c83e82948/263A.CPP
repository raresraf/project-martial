/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem from Codeforces (Problem 263A).
 * @details The problem asks for the minimum number of moves to bring the single '1'
 * in a 5x5 matrix to the center cell (at row 3, column 3). A move consists of
 * swapping two adjacent rows or two adjacent columns. The minimum number of moves
 * is equivalent to the Manhattan distance from the '1's current position to the center.
 *
 * @algorithm
 * 1. Iterate through the matrix row by row (from 1 to 5).
 * 2. In each row, read the 5 column values.
 * 3. Use `if-else` statements to check which column contains the '1'.
 * 4. Once the '1' is located at `(row, col)`, calculate the Manhattan distance to
 *    the center `(3, 3)`. The distance is `abs(row - 3) + abs(col - 3)`.
 * 5. This code calculates the horizontal distance `abs(col - 3)` implicitly.
 *    - If `col` is 1 or 5, the horizontal distance is 2.
 *    - If `col` is 2 or 4, the horizontal distance is 1.
 *    - If `col` is 3, the horizontal distance is 0.
 * 6. The total distance is computed and printed, and the loop is terminated.
 *
 * @time_complexity O(1), as the matrix size is fixed at 5x5.
 * @space_complexity O(1).
 */
#include <cmath>
#include <iostream>

using namespace std;

int main() {
  int col_1, col_2, col_3, col_4, col_5;

  /**
   * @block_logic Iterate through each row of the 5x5 matrix to find the '1'.
   */
  for (int row = 1; row < 6; row++) {
    cin >> col_1 >> col_2 >> col_3 >> col_4 >> col_5;

    // Check which column contains the '1' and calculate the Manhattan distance.
    if (col_1 == 1) {
      // Horizontal distance = abs(1 - 3) = 2. Vertical distance = abs(row - 3).
      cout << 2 + abs(row - 3);
      break;
    } else if (col_2 == 1) {
      // Horizontal distance = abs(2 - 3) = 1.
      cout << 1 + abs(row - 3);
      break;
    } else if (col_3 == 1) {
      // Horizontal distance = abs(3 - 3) = 0.
      cout << abs(row - 3);
      break;
    } else if (col_4 == 1) {
      // Horizontal distance = abs(4 - 3) = 1.
      cout << 1 + abs(row - 3);
      break;
    } else if (col_5 == 1) {
      // Horizontal distance = abs(5 - 3) = 2.
      cout << 2 + abs(row - 3);
      break;
    } else {
      // The '1' is not in this row, continue to the next.
      continue;
    }
  }
  return 0;
}