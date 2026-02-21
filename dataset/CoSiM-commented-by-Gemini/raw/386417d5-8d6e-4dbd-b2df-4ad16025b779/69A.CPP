/**
 * @file 69A.CPP
 * @brief A space-optimized C++ solution for the "Young Physicist" problem (Codeforces 69A).
 *
 * @details This program checks if a system of 3D vectors is in equilibrium by summing their
 * components. It uses a memory-efficient O(1) space approach by processing vectors
 * as they are read, without storing them in an array.
 *
 * @algorithm The program reads the number of vectors, `n`. It then loops `n` times,
 * reading each vector's (x, y, z) components and adding them to three running sum
 * variables. After the loop, it checks if all three sums are zero. A chained
 * comparison `(0 == x && x == y && y == z)` is used for this check. If true, the
 * system is in equilibrium ("YES"); otherwise, it is not ("NO").
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of vectors, for a single pass.
 * Space Complexity: O(1), requiring only constant extra space for sum variables.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;

  int x = 0, y = 0, z = 0;

  /**
   * @brief Block Logic: Reads `n` vectors and accumulates their components in-place.
   * @invariant At the beginning of each iteration, `n` has been decremented, and `x`, `y`, `z`
   * hold the sum of components of vectors read so far.
   */
  while (n--) {
    int a, b, c;
    cin >> a >> b >> c;

    x += a;
    y += b;
    z += c;
  }

  /**
   * @brief Block Logic: Checks for equilibrium using a ternary operator and a chained comparison.
   * The condition `0 == x && x == y && y == z` is a concise way to verify that x, y, and z are all zero.
   */
  (0 == x && x == y && y == z) ? cout << "YES" : cout << "NO";
  cout << endl;

  return 0;
}
