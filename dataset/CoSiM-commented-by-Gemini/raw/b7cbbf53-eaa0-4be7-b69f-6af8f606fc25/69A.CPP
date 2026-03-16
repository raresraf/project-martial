/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" problem (Codeforces 69A).
 *
 * @details
 * This program determines if a body is in equilibrium under the influence of
 * multiple forces. According to Newton's first law, a body is in equilibrium
 * if and only if the vector sum of all forces acting upon it is zero.
 *
 * In a 3D Cartesian coordinate system, this means the sum of the x-components,
 * the sum of the y-components, and the sum of the z-components of all force
 * vectors must each individually be zero.
 *
 * The program reads the number of forces, `t`, and then iterates `t` times. In each
 * iteration, it reads a force vector's components (a, b, c) and adds them to
 * running sums for each axis (ca, cb, cc). After all vectors have been processed,
 * it checks if all three sums are zero.
 *
 * Algorithm:
 * 1. Read the number of force vectors, `t`.
 * 2. Initialize sums for x, y, and z components to zero.
 * 3. Loop `t` times, reading each force vector and adding its components to the
 *    respective sums.
 * 4. After the loop, check if all three sums are zero.
 * 5. Print "YES" for equilibrium, "NO" otherwise.
 *
 * Time Complexity: O(N), where N is the number of forces, due to a single loop.
 * Space Complexity: O(1), as only a few variables are needed for the sums.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int t, a, b, c;
  // ca, cb, cc are the running sums for the x, y, and z components respectively.
  int ca = 0, cb = 0, cc = 0;

  cin >> t;

  /**
   * @brief Main loop to process all force vectors.
   * @details This loop iterates `t` times, reading one force vector (a, b, c)
   * in each iteration and accumulating the components into the total sums
   * `ca`, `cb`, and `cc`.
   */
  while (t--) {
    cin >> a >> b >> c;
    ca += a;
    cb += b;
    cc += c;
  }

  /**
   * @brief Equilibrium check.
   * @details If the net force along each of the x, y, and z axes is zero,
   * the body is in equilibrium.
   */
  if (ca == 0 && cb == 0 && cc == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}