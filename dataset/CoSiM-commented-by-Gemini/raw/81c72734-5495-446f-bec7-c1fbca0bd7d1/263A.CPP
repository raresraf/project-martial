/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem by calculating Manhattan distance.
 *
 * This program reads a 5x5 matrix containing a single '1' and calculates the
 * minimum number of moves (up, down, left, right) to move the '1' to the center.
 * The center of the 1-indexed 5x5 grid is at (3,3). The number of moves is
 * equivalent to the Manhattan distance.
 *
 * @algorithm
 * 1. Find the 1-indexed coordinates (row, col) of the '1'.
 * 2. Calculate the Manhattan distance to the center: |3 - row| + |3 - col|.
 *
 * @complexity
 * Time: O(1), since the matrix size is fixed at 5x5, requiring a constant
 *       number of operations.
 * Space: O(1), for storing the 5x5 matrix and a few variables.
 */
#include <iostream>
#include <cmath> // For std::abs

// The <bits/stdc++.h> header is a non-standard GNU extension often used for convenience.
// #include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to find the '1' and calculate its distance to the center.
 */
int main() {
  // i, j: loop counters for rows and columns.
  // ai, aj: store the 1-indexed row and column of the '1'.
  // a: a 6x6 array to allow for 1-based indexing (from 1 to 5) without going out of bounds.
  int i, j, ai = 0, aj = 0, a[6][6];
  
  // Pre-condition: Standard input contains a 5x5 matrix.
  // Block Logic: Use 1-based indexing to iterate through the 5x5 grid.
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> a[i][j];
      // If the current element is the '1', save its coordinates.
      if (a[i][j] != 0) {
        ai = i;
        aj = j;
        // This break only exits the inner loop; the outer loop continues.
        // This is slightly inefficient but correct as there's only one '1'.
        break;
      }
    }
  }
  
  // Invariant: 'ai' and 'aj' hold the 1-indexed coordinates of the '1'.
  // Calculate the Manhattan distance to the center (3,3).
  cout << (abs(3 - aj) + abs(3 - ai)) << "
";
  
  return 0;
}
