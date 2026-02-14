/**
 * @file 263A.CPP
 * @brief Solution for the "Beautiful Matrix" problem.
 * @details The problem involves a 5x5 matrix containing a single '1' and all other
 * elements as '0'. The goal is to find the minimum number of single-step moves
 * (swapping adjacent rows or columns) required to move the '1' to the center
 * of the matrix, which is at position (3,3) (or (2,2) using 0-based indexing).
 *
 * @section algorithm Algorithm
 * The minimum number of moves is the Manhattan distance between the '1's current
 * position and the center of the matrix.
 * 1. The program iterates through the 5x5 matrix using nested loops, with row `i`
 *    and column `j` both from 0 to 4.
 * 2. It reads each element of the matrix one by one.
 * 3. When an element with the value '1' is found at coordinates (i, j):
 *    a. The Manhattan distance to the center (2, 2) is calculated using the
 *       formula: `abs(i - 2) + abs(j - 2)`.
 *    b. This distance represents the minimum number of row and column swaps needed.
 *    c. The result is printed, and the program terminates immediately, as there's
 *       no need to read the rest of the matrix.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(1), because the matrix size is fixed at 5x5. The program
 *   will always perform at most 25 reads.
 * - Space Complexity: O(1), as it only uses a few variables to store the current
 *   element and loop indices, not the entire matrix.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to find the location of '1' and calculate moves.
 * @return 0 on successful execution.
 */
int main() {

#ifndef ONLINE_JUDGE
  // This block is for redirecting input/output to files for local testing,
  // it is ignored when compiled online.
  freopen("input", "r", stdin);
  freopen("output", "w", stdout);
#endif

  int a; // To store the value of each matrix element.
  
  // Iterate through the 5x5 matrix row by row.
  for (int i = 0; i < 5; i++) {
    // Iterate through each column in the current row.
    for (int j = 0; j < 5; j++) {
      cin >> a;
      // If the '1' is found.
      if (a == 1) {
        // Calculate the Manhattan distance to the center (2, 2) and print it.
        cout << abs(i - 2) + abs(j - 2) << endl;
        // Exit the program since the '1' has been found.
        return 0;
      }
    }
  }
}
