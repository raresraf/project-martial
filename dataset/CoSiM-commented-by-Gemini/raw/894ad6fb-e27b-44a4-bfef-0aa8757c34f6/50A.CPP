/**
 * @file 50A.CPP
 * @brief Solution for the "Domino piling" problem.
 *
 * @details
 * This program calculates the maximum number of 2x1 dominoes that can be placed
 * on a rectangular board of size M x N. The solution is found by calculating
 * the total area of the board (M * N) and then finding how many 2x1 dominoes
 * can fit, which is equivalent to the total area divided by 2.
 *
 * @section
 * Algorithm: Direct mathematical formula.
 * Time Complexity: O(1), as it involves a fixed number of arithmetic operations.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function with an implicit 'int' return type.
 */
main() {
  // n, m: the dimensions of the rectangular board.
  int n, m;
  cin >> n >> m;

  /**
   * @brief Calculates and prints the maximum number of dominoes.
   *
   * Logic: The total area of the board is n * m. Each domino covers an area of 2.
   * The maximum number of non-overlapping dominoes is the floor of the total
   * area divided by 2. The expression `(int)(n * m * 0.5)` achieves this by
   * performing floating-point multiplication and then truncating the result to an integer.
   * This is functionally equivalent to integer division: `(n * m) / 2`.
   */
  cout << (int)(n * m * 0.5);
}
