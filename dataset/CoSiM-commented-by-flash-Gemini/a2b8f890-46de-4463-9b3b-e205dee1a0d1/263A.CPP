/**
 * @file 263A.CPP
 * @brief Discrete simulation of matrix row/column swaps to achieve a target state.
 * 
 * Logic: Relocates a specific entity ('1') to the geometric center (2, 2) of a 5x5 grid 
 * through a sequence of adjacent row and column swaps, counting each atomic operation.
 * 
 * Algorithm: Iterative simulation of coordinate normalization.
 * Time Complexity: $O(1)$ due to fixed 5x5 grid dimensions.
 * Space Complexity: $O(1)$ for the static grid storage.
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  int counter = 0;
  int row = 2;
  int coloum = 2;
  // Matrix representing the 2D grid state.
  string matrix[5][5];

  /**
   * Block Logic: Initialization phase to locate the target entity '1'.
   * Invariant: Grid is populated with exactly one '1' and twenty-four '0's.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      char temp;
      cin >> temp;
      matrix[i][j] = temp;
      // Functional Utility: Captures the initial spatial coordinates of the target.
      if (temp == '1') {
        row = i;
        coloum = j;
      }
    }
  }

  /**
   * Block Logic: Vertical normalization via row swaps.
   * Invariant: Each loop iteration reduces the vertical distance to the target row (2).
   */
  while (row != 2) {
    if (row < 2) {
      // Logic: Performs a physical row swap to migrate the target downwards.
      for (int i = 0; i < 5; i++) {
        string temp = matrix[row][i];
        matrix[row][i] = matrix[row + 1][i];
        matrix[row + 1][i] = temp;
      }
      row++;
    } else if (row > 2) {
      // Logic: Performs a physical row swap to migrate the target upwards.
      for (int i = 0; i < 5; i++) {
        string temp = matrix[row][i];
        matrix[row][i] = matrix[row - 1][i];
        matrix[row - 1][i] = temp;
      }
      row--;
    }
    counter++;
  }

  /**
   * Block Logic: Horizontal normalization via column swaps.
   * Invariant: Each loop iteration reduces the horizontal distance to the target column (2).
   */
  while (coloum != 2) {
    if (coloum < 2) {
      // Logic: Performs a physical column swap to migrate the target rightwards.
      for (int i = 0; i < 5; i++) {
        string temp = matrix[i][coloum];
        matrix[i][coloum] = matrix[i][coloum + 1];
        matrix[i][coloum + 1] = temp;
      }
      coloum++;
    } else if (coloum > 2) {
      // Logic: Performs a physical column swap to migrate the target leftwards.
      for (int i = 0; i < 5; i++) {
        string temp = matrix[i][coloum];
        matrix[i][coloum] = matrix[i][coloum - 1];
        matrix[i][coloum - 1] = temp;
      }
      coloum--;
    }
    counter++;
  }

  // Output the total number of swaps required to reach the beautiful state.
  cout << counter;
  
  return 0;
}
