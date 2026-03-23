/**
 * @file 263A.CPP
 * @brief Solution for Codeforces Problem 263A, "Beautiful Matrix".
 * @details The problem involves a 5x5 matrix containing a single '1' and 24 '0's.
 * The goal is to find the minimum number of moves (swapping adjacent elements)
 * to move the '1' to the center of the matrix (at position (2,2) using 0-based indexing).
 * The minimum number of moves is equivalent to the Manhattan distance between the
 * current position of '1' and the center.
 *
 * @algorithm Manhattan Distance Calculation
 * @complexity
 * Time: O(1), as the matrix size is fixed at 5x5. The loops run at most 25 times.
 * Space: O(1), as it uses a fixed amount of memory.
 */
#include <iostream>
#include <cmath> // For std::abs

// Unused headers from the original template are removed for clarity.
// #include <algorithm>
// ... and others

#define REP(i, n) for (int i = 0; i < (int)(n); ++i)

using namespace std;

// Unused typedefs and constants from the original template are removed.

int main() {
  char c;
  int answer = 0;
  bool found = false;

  /**
   * @brief Loop to iterate through the 5x5 matrix to find the '1'.
   * The loops iterate from i=0 to 4 (rows) and j=0 to 4 (columns).
   */
  REP(i, 5) {
    REP(j, 5) {
      cin >> c;
      if (c == '1') {
        /**
         * @brief Calculate the Manhattan distance to the center (2,2).
         * The number of row moves is |2 - i| and column moves is |2 - j|.
         */
        answer = abs(2 - i) + abs(2 - j);
        found = true;
      }
      // An inner break is not strictly necessary as the input has only one '1',
      // but it's a good practice for efficiency if the rest of the row is not needed.
      // However, the original code had a break here after the if, which would break
      // after the first element of the row containing '1', which is incorrect.
      // The logic here is preserved from the original file to demonstrate the intent.
      if (found)
        break;
    }
    if (found)
      break; // Exit the outer loop once '1' is found.
  }

  // Print the minimum number of moves.
  cout << answer << endl;

  return 0; // Indicate successful execution.
}
