/**
 * @file 263A.CPP
 * @brief Solution to a problem requiring the calculation of Manhattan distance on a 5x5 grid.
 * @details The program reads a 5x5 matrix containing a single '1' and all other '0's. It then calculates the minimum number of moves
 * (up, down, left, or right) required to move the '1' to the center of the matrix, which is at position (2,2) with 0-based indexing.
 *
 * Algorithm:
 * 1. The program iterates through a 5x5 grid, reading one element at a time.
 * 2. It searches for the element with the value '1'. When found, it stores its row and column indices (`x` and `y`).
 * 3. The center of the 5x5 grid is at index (2, 2).
 * 4. The number of moves required is the Manhattan distance between the location of '1' (`x`, `y`) and the center (2, 2).
 * 5. The Manhattan distance formula is `|x1 - x2| + |y1 - y2|`. In this case, it's `abs(x - 2) + abs(y - 2)`.
 * 6. The final result is printed.
 *
 * Time Complexity: O(1), because the size of the matrix is fixed at 5x5, so the loops run a constant number of times (25 iterations).
 * Space Complexity: O(1), as it uses a few variables to store the coordinates and the input number.
 */
#include <cmath>
#include <iostream>

using namespace std;

int main() {
  double n, y, x;

  /**
   * Block Logic: Iterate through the 5x5 matrix to find the coordinates of the '1'.
   * Pre-condition: The input stream contains 25 integer values representing the matrix.
   * Invariant: After the loops complete, 'x' and 'y' will hold the 0-indexed row and column of the element '1'.
   */
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j) {
      cin >> n;
      if (n == 1) {
        x = i; // Store the row index.
        y = j; // Store the column index.
      }
    }
  
  // Calculate the Manhattan distance from the found coordinates (x, y) to the center (2, 2).
  cout << abs(x - 2) + abs(y - 2);
}
