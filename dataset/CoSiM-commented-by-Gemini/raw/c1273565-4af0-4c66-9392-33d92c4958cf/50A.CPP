/**
 * @file 50A.CPP
 * @brief Solves the "Domino Piling" problem.
 * @details This program calculates the maximum number of 2x1 dominoes that can be
 *          placed on a rectangular board of size M x N.
 *
 * Algorithm:
 *   The problem is solved by calculating the total area of the board (M * N)
 *   and then finding how many 2-square dominoes can fit. This is equivalent to
 *   the integer division of the total area by 2.
 *   The code handles three cases for the dimensions n and m:
 *   1. If n is even, the result is (n/2) * m.
 *   2. If m is even, the result is (m/2) * n.
 *   3. If both are odd, it uses the formula (n-1)/2 * m + m/2.
 *   All three cases are equivalent to floor((n * m) / 2).
 *
 * Time Complexity: O(1), as the solution involves a few arithmetic operations.
 * Space Complexity: O(1), as a fixed amount of memory is used.
 */
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char const *argv[]) {
  int n, m;
  std::cin >> n >> m;

  /**
   * Block Logic: Handles the case where the board's height (n) is even.
   * In this scenario, the board can be perfectly tiled horizontally without
   * any remaining single squares in the rows, simplifying the calculation.
   * The use of exit(0) is functionally correct but `return 0` is more conventional.
   */
  if (n % 2 == 0) {
    std::cout << n / 2 * m << '
';
    exit(0);
  }

  /**
   * Block Logic: Handles the case where the board's width (m) is even.
   * Similar to the previous case, this allows for perfect vertical tiling
   * within columns.
   */
  if (m % 2 == 0) {
    std::cout << m / 2 * n << '
';
    return 0;
  }

  /**
   * Block Logic: Handles the case where both dimensions are odd.
   * The formula calculates the tiling for the largest even subgrid ((n-1) * m)
   * and then adds the additional dominoes that fit in the last remaining row.
   * This is mathematically equivalent to floor((n * m) / 2).
   */
  std::cout << (n - 1) / 2 * m + m / 2 << '
';
  return 0;
}
