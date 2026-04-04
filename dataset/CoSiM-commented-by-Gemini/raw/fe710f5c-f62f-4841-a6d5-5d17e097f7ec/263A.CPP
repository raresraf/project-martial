/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem on Codeforces.
 *
 * @details The problem involves a 5x5 matrix containing 24 zeros and one '1'.
 * The goal is to find the minimum number of moves (swapping adjacent rows or columns)
 * to move the '1' to the center of the matrix, which is at position (3,3) in 1-based
 * indexing or (2,2) in 0-based indexing.
 *
 * Algorithm:
 * 1. Iterate through the 5x5 matrix to find the initial position of the '1'.
 * 2. Store the 0-indexed row ('ai') and column ('aj') of the '1'.
 * 3. The minimum number of moves required is the Manhattan distance between the
 *    current position (ai, aj) and the target center position (2, 2).
 * 4. Calculate the Manhattan distance as `abs(ai - 2) + abs(aj - 2)`.
 * 5. Print the result.
 *
 * Time Complexity: O(1), as the matrix size is fixed at 5x5 (25 operations).
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int a, ai, aj;
  
  /**
   * @brief Loop through the 5x5 matrix to find the location of the '1'.
   * Pre-condition: The input contains a 5x5 matrix of 24 zeros and one 1.
   * Post-condition: 'ai' and 'aj' will hold the 0-indexed coordinates of the '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> a;
      if (a == 1) {
        ai = i;
        aj = j;
      }
    }
  }

  // Calculate the Manhattan distance to the center (2,2).
  // This represents the minimum number of row and column swaps.
  cout << abs(ai - 2) + abs(aj - 2);
}
