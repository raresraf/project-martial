/**
 * @file 58A.CPP
 * @brief Solves a combinatorial pairing problem.
 * @details The filename `58A.CPP` is a misnomer, as the logic does not match the
 * "Chat room" problem (Codeforces 58A). Instead, it appears to solve a problem
 * involving forming pairs from two groups of items.
 *
 * Problem Description (Inferred):
 * Given `n` items of a red type and `m` items of a blue type, find:
 * 1. The maximum number of pairs consisting of one red and one blue item.
 * 2. From the remaining items (which will all be of the same color), find the
 *    maximum number of pairs that can be formed.
 *
 * Algorithm:
 * 1. The number of red-blue pairs is limited by the smaller of the two groups,
 *    so the result is `min(n, m)`.
 * 2. After forming these pairs, `abs(n - m)` items of one color remain.
 * 3. The number of same-color pairs that can be formed from the remainder is
 *    `abs(n - m) / 2`.
 *
 * The implementation uses an unnecessarily long and complex `if-else if`
 * cascade to handle all cases of n vs. m and their parities, but every
 * branch simplifies to the same core logic.
 *
 * Time Complexity: O(1).
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, m;
  // n red m blue
  cin >> n >> m;

  /**
   * @brief This block is a highly verbose and redundant implementation of a
   * simple calculation. All of the different conditional branches produce a
   * result equivalent to:
   *   cout << min(n, m) << " " << (abs(n - m) / 2) << endl;
   *
   * It exhaustively checks the relationship between n and m (n > m, m > n, n == m)
   * and their parities, but the underlying formula remains the same.
   */
  if (n == m) {
    cout << n << " " << 0 << endl;
  } else if (n > m && m == 0) {
    cout << 0 << " " << n / 2 << endl;
  } else if (m > n && n == 0) {
    cout << 0 << " " << m / 2 << endl;
  } else if (n > m && (n % 2 == 0) && (m % 2 == 0)) {
    cout << m << " " << (n - m) / 2 << endl;
  } else if (m > n && (n % 2 == 0) && (m % 2 == 0)) {
    cout << n << " " << (m - n) / 2 << endl;
  } else if (n > m && (n % 2 == 1) && (m % 2 == 1)) {
    cout << m << " " << (n - m) / 2 << endl;
  } else if (m > n && (n % 2 == 1) && (m % 2 == 1)) {
    cout << n << " " << (m - n) / 2 << endl;
  } else if (n > m && (n % 2 == 1) && (m % 2 == 0)) {
    int in = (n - m) / 2;
    cout << m << " " << in << endl;
  } else if (n > m && (n % 2 == 0) && (m % 2 == 1)) {
    int in = (n - m) / 2;
    cout << m << " " << in << endl;
  } else if (m > n && (n % 2 == 0) && (m % 2 == 1)) {
    int in = (m - n) / 2;
    cout << n << " " << in << endl;
  } else if (m > n && (n % 2 == 1) && (m % 2 == 0)) {
    int in = (m - n) / 2;
    cout << n << " " << in << endl;
  }
}
