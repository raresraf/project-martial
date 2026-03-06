/**
 * @file 69A.CPP
 * @brief Determines if a body is in equilibrium under a set of forces.
 *
 * This program solves the "Young Physicist" problem. It reads a set of 3D force
 * vectors applied to a body and determines if the body is in equilibrium.
 * A body is in equilibrium if and only if the vector sum of all forces acting
 * upon it is zero. This means the sum of the x, y, and z components of all
 * force vectors must each be zero.
 *
 * @algorithm Vector Component Summation
 * @complexity
 * Time: O(N), where N is the number of force vectors, as we iterate through
 *       the vectors once to sum their components.
 * Space: O(1), as we only use a few variables to store the running sums.
 */
#include <iostream>
using namespace std;

// Typedefs for various integer types, common in competitive programming.
typedef long long int lli;
typedef long long ll;
typedef unsigned long long int ulli;
typedef double dbl;

/**
 * @brief Main function to read force vectors and check for equilibrium.
 */
int main(int argc, char const *argv[]) {
  // Fast I/O optimization for C++ streams.
  ios_base::sync_with_stdio(false);
  
  // n: the number of force vectors.
  ulli n;
  cin >> n;
  
  // a, b, c: components of a single force vector (x, y, z).
  // d, e, f: running sums of the x, y, and z components of all vectors.
  lli a, b, c, d = 0, e = 0, f = 0;
  
  // Pre-condition: Component sums d, e, f are all zero.
  // Block Logic: Loop 'n' times, once for each force vector.
  while (n--) {
    cin >> a >> b >> c;
    // Accumulate the components into their respective sums.
    d += a;
    e += b;
    f += c;
  }
  
  // Invariant: d, e, f hold the total force components along each axis.
  // Block Logic: Check if the net force along all three axes is zero.
  // The expression !d is equivalent to d == 0 for integers.
  if (!d && !e && !f) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  
  return 0;
}
