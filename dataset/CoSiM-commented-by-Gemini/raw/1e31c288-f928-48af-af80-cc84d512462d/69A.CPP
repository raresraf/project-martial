/**
 * @file 69A.CPP
 * @brief Solution for Codeforces problem 69A, "Young Physicist".
 *
 * @description
 * This program determines if a system is in equilibrium by checking if the sum
 * of a given set of 3D vectors is the zero vector.
 *
 * @algorithm
 * 1. Initialize three variables, `x`, `y`, and `z`, to 0. These will represent
 *    the sum of the components of the resultant vector.
 * 2. Read the number of vectors, `t`.
 * 3. Loop `t` times:
 *    a. Read the three components (a, b, c) of a vector.
 *    b. Add each component to its corresponding sum variable (x, y, or z).
 * 4. After the loop, check if x, y, and z are all zero.
 * 5. If they are, print "YES"; otherwise, print "NO".
 *
 * @time_complexity O(N), where N is the number of vectors.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // a, b, c: components of an individual input vector.
  // t: the number of vectors to process.
  int a, b, c, t;
  // x, y, z: the sums of the components of all vectors along each axis.
  int x = 0, y = 0, z = 0;
  cin >> t;

  // Block Logic: Iterate through all 't' vectors and sum their components.
  while (t--) {
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }

  // Block Logic: Check if the resultant vector is the zero vector.
  // If the sum of components on each axis is 0, the system is in equilibrium.
  if (x == 0 && y == 0 && z == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
