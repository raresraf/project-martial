/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem with an optimal single-pass approach.
 * @details This program finds the minimum number of moves (Manhattan distance) to move the '1'
 * in a 5x5 grid to the center (2,2). It iterates through the grid, reading one number at a time.
 * As soon as the '1' is found, it calculates the Manhattan distance to the center, prints the result,
 * and terminates immediately. This is highly efficient as it avoids storing the matrix and stops
 * as soon as the goal is achieved.
 *
 * @algorithm Single-pass search and immediate calculation.
 * @complexity
 * Time complexity: O(1), in the worst case it reads all 25 numbers. On average, it's faster.
 * Space complexity: O(1), as it does not store the matrix.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * @brief Conditional compilation for local testing.
   * @details This block is a common competitive programming technique. When the code is run locally
   * (where ONLINE_JUDGE is not defined), it redirects standard input and output to files,
   * making testing easier. This block is ignored by online judges.
   */
#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  int x; // Variable to store the current number being read.
  
  /**
   * @brief Loop through the 5x5 grid, reading one element at a time.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> x;
      // When the '1' is found (since it's the only non-zero input, `if (x)` is sufficient).
      if (x) {
        // Calculate the Manhattan distance from the current (0-indexed) coordinates (i, j)
        // to the center (2, 2) and print it.
        cout << abs(i - 2) + abs(j - 2);
        // Terminate the program immediately as the solution has been found.
        return 0;
      }
    }
  }

  return 0;
}
