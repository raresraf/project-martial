/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A - "Young Physicist".
 *
 * This program determines if a system of 3D force vectors is in equilibrium.
 * It reads 'n' vectors, sums their components along the x, y, and z axes,
 * and then checks if the total sum for each axis is zero. If all three sums
 * are zero, the system is in equilibrium.
 *
 * @algorithm Vector component summation.
 * @time_complexity O(N), where N is the number of vectors, as the program
 *                  iterates through the input once to sum the components.
 * @space_complexity O(1), as it only stores the running sums of the vector
 *                   components.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int t, i, c, s, a = 0, b = 0, d = 0;
  // Read the number of vectors.
  cin >> t;
  /**
   * @brief Accumulates the sum of vector components.
   * This loop reads each of the 't' vectors' components (i, c, s) and
   * adds them to the running totals for each axis (a, b, d).
   */
  while (t--) {
    cin >> i;
    cin >> c;
    cin >> s;
    a = a + i;
    b = b + c;
    d = d + s;
  }
  /**
   * @brief Checks for equilibrium.
   * If the sum of components along each axis is zero, the net force is zero,
   * and the system is in equilibrium.
   */
  if (a == 0 && b == 0 && d == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
