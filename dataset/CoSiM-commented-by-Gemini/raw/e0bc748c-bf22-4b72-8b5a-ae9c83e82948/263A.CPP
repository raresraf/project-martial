#include <cmath>
#include <iostream>

/**
 * @file 263A.CPP
 * @brief An efficient solution to the "Beautiful Matrix" problem.
 *
 * Problem Statement:
 * Given a 5x5 matrix containing 24 zeroes and a single '1', find the minimum
 * number of moves (adjacent row/column swaps) to move the '1' to the center
 * of the matrix, which is at position (3,3).
 *
 * Algorithm:
 * The solution calculates the Manhattan distance from the '1's position to the
 * center. This implementation uses an efficient approach where it processes the
 * matrix row by row. As soon as the row containing the '1' is read, it
 * determines which column holds the '1' and immediately calculates the total
 * Manhattan distance, prints the result, and terminates.
 *
 * The calculation is broken down as:
 * - Vertical distance: `abs(row - 3)`
 * - Horizontal distance: This is pre-calculated based on the column index.
 *   - Col 1 or 5: distance is 2 (`abs(1-3)` or `abs(5-3)`)
 *   - Col 2 or 4: distance is 1 (`abs(2-3)` or `abs(4-3)`)
 *   - Col 3: distance is 0 (`abs(3-3)`)
 *
 * This avoids storing the entire matrix or even the coordinates, leading to
 * a very direct computation.
 *
 * Time Complexity: O(1), as in the worst case it reads all 25 numbers once.
 * Space Complexity: O(1).
 */

using namespace std;

int main() {
  // Variables to hold the values of a single row.
  int col_1, col_2, col_3, col_4, col_5;

  // Block Logic: Iterate through the matrix row by row, using 1-based indexing.
  for (int row = 1; row <= 5; row++) {
    cin >> col_1 >> col_2 >> col_3 >> col_4 >> col_5;
    
    // Pre-condition: Check which column contains the '1' and calculate
    // the Manhattan distance immediately.
    if (col_1 == 1) {
      // Horizontal distance from col 1 to 3 is 2.
      cout << 2 + abs(row - 3);
      break; // Exit after finding the '1' and printing the result.
    } else if (col_2 == 1) {
      // Horizontal distance from col 2 to 3 is 1.
      cout << 1 + abs(row - 3);
      break;
    } else if (col_3 == 1) {
      // Horizontal distance from col 3 to 3 is 0.
      cout << abs(row - 3);
      break;
    } else if (col_4 == 1) {
      // Horizontal distance from col 4 to 3 is 1.
      cout << 1 + abs(row - 3);
      break;
    } else if (col_5 == 1) {
      // Horizontal distance from col 5 to 3 is 2.
      cout << 2 + abs(row - 3);
      break;
    } else {
      // If '1' is not in this row, continue to the next.
      continue;
    }
  }
  return 0;
}
