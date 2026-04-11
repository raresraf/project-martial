/**
 * @file 263A.CPP
 * @brief Solution to a competitive programming problem: "Beautiful Matrix".
 * This program reads a 5x5 matrix to find the position of the element '1'.
 * It then calculates the minimum number of moves required to bring '1' to the
 * center of the matrix (position [2][2] for 0-indexed matrix) using Manhattan distance.
 * Algorithm: Matrix traversal, element localization, and Manhattan distance calculation.
 * Time Complexity: O(R*C) where R and C are the number of rows and columns (constant 5x5 matrix).
 * Space Complexity: O(1)
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function where the program execution begins.
 * Reads the matrix, finds the '1', and calculates moves to center.
 */
int main() {
  int a, ai, aj; // 'a' for current element, 'ai' and 'aj' for row and column of '1'.
  /**
   * Block Logic: Traverses a 5x5 matrix to find the coordinates of the element '1'.
   * Invariant: 'i' and 'j' represent the current row and column indices.
   * Pre-condition: Input is a 5x5 matrix with exactly one '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      // Reads an integer element of the matrix.
      cin >> a;
      /**
       * Block Logic: Stores the coordinates if the current element is '1'.
       * Pre-condition: 'a' is the current matrix element.
       */
      if (a == 1) {
        ai = i; // Stores the row index of '1'.
        aj = j; // Stores the column index of '1'.
      }
    }
  }

  // Calculates and prints the Manhattan distance from the position of '1' (ai, aj)
  // to the center of the 5x5 matrix (2, 2). This represents the minimum moves.
  cout << abs(ai - 2) + abs(aj - 2);
}
