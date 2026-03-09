/**
 * @file 69A.CPP
 * @brief Solution to Codeforces Problem 69A: "Young Physicist".
 *        This program determines if a system of forces acting on an object is in equilibrium.
 *        Given a number of forces, each specified by its X, Y, and Z components,
 *        the program calculates the sum of components in each dimension. If all sums are zero,
 *        the system is in equilibrium.
 *
 * Algorithm: Vector summation. Reads N force vectors and sums their respective X, Y, and Z components.
 *            Checks if the resultant vector is the zero vector.
 * Time Complexity: O(N) where N is the number of force vectors, due to a single pass iteration
 *                  to sum the components.
 * Space Complexity: O(1) as only a few variables are used to store sums and temporary inputs.
 */

#include <bits/stdc++.h>
using namespace std;
#define fi(a) scanf("%d", &a);   // Macro for reading an integer into variable 'a'.
#define fli(a) scanf("%ld", &a);  // Macro for reading a long integer into variable 'a'.
#define fll(a) scanf("%lld", &a); // Macro for reading a long long integer into variable 'a'.
#define pi(a) printf("%d\n", a);  // Macro for printing an integer 'a' followed by a newline.
#define ppi(i, a) printf("Case %d: %d\n", i, a); // Macro for printing "Case i: a" followed by a newline.
#define ll long long              // Type definition for long long.
#define max 10000007              // Constant definition for a maximum value.

int main() {
  /**
   * Functional Utility: Manages the overall process of reading the number of forces,
   *                     iteratively summing their components, and determining if the system is in equilibrium.
   */

  int t, x = 0, y = 0, z = 0, a, b, c; // t: number of forces, x, y, z: sums of components, a, b, c: current force components.
  cin >> t; // Reads the number of forces from standard input.
  /**
   * Block Logic: Iterates 't' times to read each force vector and accumulate its components.
   * Pre-condition: `t` contains the number of force vectors to process.
   * Invariant: `x`, `y`, `z` accumulate the sums of the X, Y, and Z components, respectively, of all processed forces.
   */
  while (t--) {
    cin >> a >> b >> c; // Reads the X, Y, and Z components of the current force.
    x += a;             // Adds the current X component to the total X sum.
    y += b;             // Adds the current Y component to the total Y sum.
    z += c;             // Adds the current Z component to the total Z sum.
  }
  /**
   * Block Logic: Checks if the sum of all force components is zero, indicating equilibrium.
   * Pre-condition: `x`, `y`, `z` contain the total sums of X, Y, and Z components respectively.
   * Invariant: Prints "YES" if all sums are zero, indicating equilibrium; otherwise, prints "NO".
   */
  if (x == 0 && y == 0 && z == 0)
    cout << "YES\n"; // Output "YES" if the system is in equilibrium.
  else
    cout << "NO\n"; // Output "NO" if the system is not in equilibrium.
  return 0;         // Indicate successful program termination.
}
