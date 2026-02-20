/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem.
 *
 * This program finds the minimum number of moves to make a 5x5 matrix "beautiful".
 * A beautiful matrix has the single digit '1' located at its center (row 2, column 2,
 * using 0-based indexing). The allowed moves are swapping adjacent rows or columns.
 * The minimum number of moves is equivalent to the Manhattan distance from the
 * current position of the '1' to the center.
 *
 * Algorithm:
 * 1. Read the 5x5 matrix element by element.
 * 2. While reading, find the coordinates (row 'l', column 'k') of the element '1'.
 * 3. Calculate the Manhattan distance to the center (2, 2) using the formula:
 *    `abs(l - 2) + abs(k - 2)`.
 * 4. Print the result.
 *
 * Time Complexity: O(1), because the matrix size is fixed at 5x5 (25 elements).
 * Space Complexity: O(1), for the 5x5 matrix storage.
 */
#include <iostream>
#include <stdlib.h> // For abs()

using namespace std;

int main() {
  int mat[5][5]; // The 5x5 matrix.
  int l, k; // Variables to store the row (l) and column (k) of the '1'.

  /**
   * Block Logic: Read the matrix and find the position of the '1'.
   * The nested loops iterate through each cell of the 5x5 grid.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> mat[i][j];
      // Pre-condition: Check if the current element is the '1'.
      if (mat[i][j] == 1) {
        // If found, store its 0-indexed coordinates.
        l = i;
        k = j;
      }
    }
  }

  int count = 0; // Stores the total number of moves.
  
  // Calculate the number of row moves required (vertical distance).
  count = abs(l - 2);
  // Add the number of column moves required (horizontal distance).
  count = count + abs(k - 2);
  
  // Output the total Manhattan distance.
  cout << count;
  return 0;
}
