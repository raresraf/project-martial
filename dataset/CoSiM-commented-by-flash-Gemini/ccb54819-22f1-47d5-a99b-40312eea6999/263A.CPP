/**
 * @file 263A.CPP
 * @brief Solution to a competitive programming problem: "Beautiful Matrix".
 * This program reads a 5x5 matrix to find the position of the element '1'.
 * It then calculates the minimum number of moves required to bring '1' to the
 * center of the matrix (position [3][3] for 1-indexed matrix) using Manhattan distance.
 * Algorithm: Matrix traversal, element localization, and Manhattan distance calculation.
 * Time Complexity: O(R*C) where R and C are the number of rows and columns (constant 5x5 matrix).
 * Space Complexity: O(1)
 */
#include <cmath> // Required for abs() function.
#include <iostream>

using namespace std;

/**
 * @brief Main function where the program execution begins.
 * Reads the matrix, finds the '1', and calculates moves to center.
 */
int main() {
  int x = 0; // Variable to store the current matrix element.
  /**
   * Block Logic: Traverses a 5x5 matrix to find the coordinates of the element '1'.
   * Invariant: 'i' and 'j' represent the current 1-indexed row and column.
   * Pre-condition: Input is a 5x5 matrix with exactly one '1'.
   */
  for (int i = 1; i <= 5; ++i) { // Iterates through rows (1-indexed).
    for (int j = 1; j <= 5; ++j) { // Iterates through columns (1-indexed).
      // Reads an integer element of the matrix.
      cin >> x;
      /**
       * Block Logic: Calculates and prints Manhattan distance if the current element is '1'.
       * Pre-condition: 'x' is the current matrix element.
       */
      if (x == 1) {
        // Calculates and prints the Manhattan distance from the position of '1' (i, j)
        // to the center of the 5x5 matrix (3, 3). This represents the minimum moves.
        cout << abs(i - 3) + abs(j - 3) << endl;
      }
    }
  }
  return 0;
}