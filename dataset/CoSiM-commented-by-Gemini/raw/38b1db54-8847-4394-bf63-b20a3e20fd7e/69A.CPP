/**
 * @file 69A.CPP
 * @brief A standard and space-efficient solution for the "Young Physicist" problem (Codeforces 69A).
 *
 * @details This program checks if a system of 3D vectors is in equilibrium by summing their
 * components. It uses an O(1) space complexity approach by not storing the
 * individual vectors, processing them as they are read.
 *
 * @algorithm The program reads an integer `n`, the number of vectors. It then iterates `n`
 * times, reading one vector's (x, y, z) components in each iteration and adding
 * them to three running sum variables. After the loop, it checks if the sums for all
 * three axes are zero. If they are, equilibrium is confirmed ("YES"); otherwise, it is not ("NO").
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of vectors, due to a single pass.
 * Space Complexity: O(1), as it only requires constant extra space for the sums.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, a, b, c, x = 0, y = 0, z = 0;
  cin >> n;
  /**
   * @brief Block Logic: Reads `n` vectors and accumulates their components in-place.
   * @invariant After `i` iterations, `x`, `y`, and `z` hold the component-wise
   * sum of the first `i` vectors.
   */
  for (int i = 0; i < n; ++i) {
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }
  /**
   * @brief Block Logic: Checks for the equilibrium condition.
   * The system is in equilibrium if the net force vector is (0, 0, 0).
   */
  if ((x == 0) && (y == 0) && (z == 0))
    cout << "YES\n";
  else
    cout << "NO\n";

  return 0;
}