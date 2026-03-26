/**
 * @file 69A.CPP
 * @brief Solution to Codeforces problem 69A - Young Physicist.
 * @description This program determines if a body is in equilibrium by calculating the vector sum of all forces acting upon it.
 * A body is in equilibrium if and only if the sum of all force vectors is the zero vector.
 * Algorithm: The program reads N force vectors from the input. It maintains a running sum for each of the x, y, and z components. After processing all vectors, it checks if the total sum for each component is zero. If all three sums are zero, the body is in equilibrium.
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(1), as the program uses a constant amount of extra space.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n: the number of force vectors.
  // x, y, z: variables to store the sum of the x, y, and z components of all force vectors.
  int n = 0, x = 0, y = 0, z = 0;
  cin >> n;
  /**
   * Block Logic: Iterate through each force vector and accumulate its components.
   * Invariant: Before each iteration, x, y, and z hold the sum of components of the vectors processed so far.
   */
  while (n--) {
    // xa, ya, za: components of the current force vector.
    int xa, ya, za;
    cin >> xa >> ya >> za;
    // Add the components of the current vector to the total sums.
    x += xa;
    y += ya;
    z += za;
  }
  /**
   * Block Logic: Check for equilibrium condition.
   * A body is in equilibrium if the net force in all directions is zero.
   */
  if (x == 0 and y == 0 and z == 0) {
    // If the sum of components in all three dimensions is zero, the body is in equilibrium.
    cout << "YES" << endl;
  } else {
    // Otherwise, there is a net force, and the body is not in equilibrium.
    cout << "NO" << endl;
  }
  return 0;
}
