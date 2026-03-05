/**
 * @file 50A.CPP
 * @brief Solution for Codeforces Problem 50A, "Domino piling".
 * @details The problem is to find the maximum number of 2x1 dominoes that can
 * be placed on a rectangular board of size M x N.
 *
 * Algorithm:
 * The total area of the board is M * N. Each domino covers an area of 2.
 * Therefore, the maximum number of dominoes that can fit on the board is
 * the total area divided by the area of a single domino, rounded down.
 * This can be calculated with integer division as `(M * N) / 2`.
 *
 * The implementation in this file reaches the correct result, although the
 * logic for the case where both M and N are odd is unnecessarily complex.
 *
 * Time Complexity: O(1).
 * Space Complexity: O(1).
 */
#include <iostream>
#include <map>
#include <set>

using namespace std;
int main() {
  int n;
  int m;
  cin >> n >> m;
  int outputm;
  int outputn;

  /**
   * @brief This block handles the case where at least one of the board's
   * dimensions is even. In this scenario, the total area (m * n) is even,
   * so the board can be perfectly tiled. The maximum number of dominoes is
   * simply the total area divided by 2.
   */
  if (m % 2 == 0) {
    cout << (m * n) / 2;
    return 0;
  } else {
    if (n % 2 == 0) {
      cout << (m * n) / 2;
      return 0;
    }
  }

  /**
   * @brief This block handles the case where both m and n are odd.
   * The logic here is overly complex but mathematically sound. It calculates
   * the result by considering tiling a board of size (m-1) x n and then the
   * remaining 1 x n strip (and symmetrically for n-1 x m).
   *
   * A simpler approach would be to recognize that if both dimensions are odd,
   * the total area (m * n) is odd. One square will always be left over.
   * The number of dominoes is `(m * n - 1) / 2`. The code's calculations for
   * `outputm` and `outputn` both simplify to this expression.
   */
  int modedm = m - (m % 2);
  int modedn = n - (n % 2);
  outputm = 0.5 * (modedm * n) + 0.5 * (modedn);
  outputn = 0.5 * (modedn * m) + 0.5 * (modedm);

  // Since outputm and outputn are mathematically equivalent, this check is redundant.
  if (outputm > outputn) {
    cout << outputn;
    return 0;
  }
  cout << outputm;
  return 0;
}
