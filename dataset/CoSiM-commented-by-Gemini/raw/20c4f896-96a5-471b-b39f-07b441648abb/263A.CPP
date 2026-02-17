/**
 * @file 263A.CPP
 * @brief Solution for Codeforces Problem 263A, "Beautiful Matrix".
 * @details This program calculates the minimum number of moves to bring a single '1' in a 5x5 matrix to the center. The number of moves is the Manhattan distance from the '1's current position to the center. This implementation uses 1-based indexing for rows and columns.
 *
 * @algorithm Direct Calculation
 * @description The program iterates through a 5x5 matrix using 1-based indexing to find the '1'. Once found, it calculates the Manhattan distance from its coordinates (r, c) to the center (3, 3) and immediately terminates, printing the result.
 * @complexity
 * - Time: O(1), as the matrix size is constant and the program exits as soon as the '1' is found.
 * - Space: O(1), using a fixed number of variables.
 */
#include <bits/stdc++.h>
using namespace std;

#define DBG(x) cout << #x << " = " << (x) << "\n"
/**
 * @brief A utility macro to print a value and terminate the program.
 * This is used for early exit once the solution is found.
 */
#define END(e)                                                                 \
  {                                                                            \
    cout << (e) << "\n";                                                       \
    return EXIT_SUCCESS;                                                       \
  }

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  int v;

  /**
   * @brief Iterates through the 5x5 matrix using 1-based indexing (rows 1-5, columns 1-5).
   */
  for (int r = 1; r < 6; ++r) {
    for (int c = 1; c < 6; ++c) {
      cin >> v;
      /**
       * @brief If the '1' is found, calculate the Manhattan distance and exit.
       * The center of a 1-indexed 5x5 matrix is at (3, 3).
       * The calculation `abs(r - 3) + abs(c - 3)` gives the Manhattan distance.
       */
      if (v == 1)
        END(abs(r - 3) + abs(c - 3))
    }
  }
}
