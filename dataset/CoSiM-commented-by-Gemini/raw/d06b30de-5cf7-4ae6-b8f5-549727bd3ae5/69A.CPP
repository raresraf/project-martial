/**
 * @file 69A.CPP
 * @brief Determines if a body is in equilibrium under a set of forces.
 *
 * This program solves the "Young Physicist" problem. It is given 'n' force
 * vectors acting on a body. The body is in equilibrium if and only if the
 * vector sum of all forces is the zero vector. This means the sum of the x, y,
 * and z components of all forces must each individually be zero.
 *
 * @b Algorithm: Component-wise Vector Summation
 *   The code sums the x, y, and z components of all force vectors separately.
 *   It then checks if all three sums are zero.
 *
 * @b Time_Complexity: O(N), for reading and summing the N force vectors.
 * @b Space_Complexity: O(N), as it stores all vector components in arrays.
 *   Note: This could be optimized to O(1) by summing the components as they
 *   are read, without storing the entire arrays.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, i;
  // a, b, c will store the sum of x, y, and z components respectively.
  int a = 0, b = 0, c = 0;
  cin >> n;

  // Arrays to store the components of each of the 'n' force vectors.
  int x[100], y[100], z[100];

  // Read all n force vectors from input.
  for (i = 0; i < n; i++) {
    cin >> x[i] >> y[i] >> z[i];
  }

  /**
   * @brief Calculate the sum of each vector component.
   */
  for (i = 0; i < n; i++) {
    a += x[i];
    b += y[i];
    c += z[i];
  }

  /**
   * @brief Check for equilibrium condition.
   * The body is in equilibrium if the net force on each axis is zero.
   */
  if (a == 0 && b == 0 && c == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}