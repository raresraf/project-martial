/**
 * @file 263A.CPP
 * @brief Solution for Codeforces Problem 263A, "Beautiful Matrix".
 * @details This program finds the minimum number of moves to bring the single '1'
 * in a 5x5 matrix to the center. This version first reads the entire matrix into
 * memory and then iterates through it to find the '1' and calculate its
 * Manhattan distance from the center (2,2).
 *
 * @algorithm Manhattan Distance Calculation on a pre-filled matrix.
 * @complexity
 * Time: O(1), as the matrix size is fixed at 5x5. The loops run a constant number of times.
 * Space: O(1), as the 5x5 matrix is of a fixed size.
 */
#include <iostream>
#include <cmath> // For std::abs

using namespace std;

int main() {
  int mat[5][5];

  /**
   * @brief First block: Read the entire 5x5 matrix from standard input.
   */
  for (auto i = 0; i < 5; i++) {
    for (auto j = 0; j < 5; j++) {
      cin >> mat[i][j];
    }
  }

  /**
   * @brief Second block: Find the '1' in the matrix and calculate the distance.
   * Pre-condition: The 'mat' array is filled with the input values.
   */
  for (auto i = 0; i < 5; i++) {
    for (auto j = 0; j < 5; j++) {
      if (mat[i][j] == 1) {
        // Calculate and print the Manhattan distance to the center (2,2).
        cout << abs(2 - i) + abs(2 - j) << endl;
        // Note: The program does not break here and continues to iterate,
        // which is inefficient but correct for the given problem constraints.
      }
    }
  }

  return 0; // Indicate successful execution.
}