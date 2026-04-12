/**
 * @file 263A.CPP
 * @brief Solution to the "Beautiful Matrix" problem.
 *
 * This program calculates the minimum number of moves to bring the '1' in a 5x5 matrix
 * to the center position (row 3, column 3, 1-indexed). A move involves swapping
 * adjacent rows or columns. The problem is equivalent to finding the Manhattan distance.
 *
 * Algorithm: Manhattan Distance Calculation.
 * Time Complexity: O(1) because the matrix size is fixed at 5x5, so processing involves a constant number of operations.
 * Space Complexity: O(1) as only a few variables are used.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Main function to solve the Beautiful Matrix problem.
 *
 * Functional Utility: Reads the 5x5 matrix, finds the position of '1', and calculates
 * the Manhattan distance to the center (3,3).
 */
int main() {
  int x, y, n;
  /**
   * Block Logic: Iterates through the 5x5 matrix to find the coordinates of the cell containing '1'.
   * Pre-condition: Input stream contains 25 integers representing the 5x5 matrix.
   * Invariant: Variables 'x' and 'y' will store the 1-indexed row and column of the '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      scanf("%d", &n);
      if (n) { // If the current element is 1
        x = i + 1; // Store 1-indexed row
        y = j + 1; // Store 1-indexed column
      }
    }
  }
  /**
   * Block Logic: Calculates and prints the minimum number of moves.
   * The minimum number of moves is the Manhattan distance from the found position (x,y)
   * to the center of the matrix (3,3).
   * Post-condition: The Manhattan distance is printed to standard output.
   */
  printf("%d\n", abs(x - 3) + abs(y - 3));
  return 0;
}
