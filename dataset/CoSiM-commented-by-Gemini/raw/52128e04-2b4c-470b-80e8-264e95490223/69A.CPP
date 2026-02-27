/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces.
 * @details The problem asks whether a body is in equilibrium given a set of
 * 3D force vectors. A body is in equilibrium if and only if the vector sum of
 * all forces acting on it is the zero vector.
 *
 * @algorithm
 * The principle of equilibrium states that the net force must be zero. This is
 * checked by summing the components of all force vectors along each axis (X, Y, Z)
 * and verifying that each sum is zero.
 * 1. Read the number of force vectors, t.
 * 2. Initialize variables `x`, `y`, and `z` to 0. These will store the sum of
 *    components for each axis.
 * 3. Loop t times:
 *    a. Read the three components (a, b, c) of a single force vector.
 *    b. Add these components to the running sums `x`, `y`, and `z`.
 * 4. After the loop, if `x`, `y`, and `z` are all zero, print "YES".
 * 5. Otherwise, the body is not in equilibrium, so print "NO".
 *
 * @complexity
 * Time Complexity: O(t), where t is the number of force vectors, as the code
 * iterates through the vectors once.
 * Space Complexity: O(1), using only a few variables for the sums.
 */
#include <bits/stdc++.h>
using namespace std;

// Note: The following preprocessor macros for C-style I/O are defined
// but are not used in the main function, which uses C++ streams (cin/cout).
#define fi(a) scanf("%d", &a);
#define fli(a) scanf("%ld", &a);
#define fll(a) scanf("%lld", &a);
#define pi(a) printf("%d\n", a);
#define ppi(i, a) printf("Case %d: %d\n", i, a);
#define ll long long
#define max 10000007

int main() {

  int t; // The number of force vectors.
  // x, y, z will store the sum of the components along each axis.
  int x = 0, y = 0, z = 0;
  // a, b, c will store the components of a single force vector.
  int a, b, c;
  
  cin >> t;
  
  /**
   * @block
   * @brief Sum the components of all force vectors.
   */
  while (t--) {
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }

  /**
   * @brief Check if the net force vector is zero.
   * If the sum of components on each axis is 0, the body is in equilibrium.
   */
  if (x == 0 && y == 0 && z == 0)
    cout << "YES\n";
  else
    cout << "NO\n";
    
  return 0;
}
