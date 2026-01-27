/**
 * @file 50A.CPP
 * @brief Solution for competitive programming problem 50A "Domino piling" on Codeforces.
 *
 * This program calculates the maximum number of 2x1 dominoes that can be placed
 * on an M x N rectangular board. Each domino covers two squares.
 *
 * Algorithm: Greedy approach, directly calculating the total area divided by the domino area.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
#include <iostream>

using namespace std;

main() {
  int n, m;
  /**
   * Block Logic: Reads the dimensions of the rectangular board (M and N).
   * Pre-condition: User provides two integer inputs.
   * Invariant: `n` stores the first dimension, `m` stores the second.
   */
  cin >> n >> m;
  /**
   * Block Logic: Calculates the maximum number of dominoes that can be placed.
   * A 2x1 domino covers 2 squares. The total area of the board is M * N.
   * Thus, the maximum number of dominoes is (M * N) / 2.
   * Invariant: The result is an integer division, effectively flooring the result.
   */
  cout << m * n / 2 << endl;
}
