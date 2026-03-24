/**
 * @file 263A.CPP
 * @brief Solution to a competitive programming problem: "Beautiful Matrix".
 *
 * Finds the position of a single '1' in a 5x5 matrix and calculates its Manhattan distance
 * to the center of the matrix (coordinate 3,3).
 *
 * Algorithm: Fixed-size grid scan to locate a specific element, followed by Manhattan distance calculation.
 * Time Complexity: O(1) due to the constant grid size (5x5).
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Main function to solve the "Beautiful Matrix" problem.
 * Scans a 5x5 matrix for a non-zero element, stores its coordinates,
 * and then computes and prints its Manhattan distance to the center (3,3).
 * @return 0 on successful execution.
 */
int main() {

  /**
   * Declares variables:
   * 'x', 'y': Stores the 1-indexed row and column coordinates of the non-zero element.
   * 'in': Temporarily holds each integer read from the input grid.
   */
  int x, y;
  int in;

  /**
   * Block Logic: Iterates through a 5x5 grid to find the coordinates of the single non-zero element.
   * The loops use 1-indexed iteration for rows (i) and columns (j).
   * Invariant: After the loops, 'x' and 'y' will hold the 1-indexed coordinates of the non-zero element.
   * Pre-condition: The input consists of a 5x5 matrix with exactly one non-zero element.
   */
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> in;
      if (in) {
        x = i;
        y = j;
      }
    }
  }

  /**
   * Block Logic: Calculates and prints the Manhattan distance from the found non-zero element
   * to the center of the 5x5 grid (coordinates 3,3).
   * Post-condition: The Manhattan distance is printed to standard output.
   */
  cout << abs(3 - x) + abs(3 - y) << endl;

  return 0;
}