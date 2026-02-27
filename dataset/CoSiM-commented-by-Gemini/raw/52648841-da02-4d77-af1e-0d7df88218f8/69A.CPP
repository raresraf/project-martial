/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces.
 * @details The problem is to determine if a body is in equilibrium under the
 * influence of `n` given force vectors. A body is in equilibrium if the
 * vector sum of all forces is zero.
 *
 * @algorithm
 * To check for equilibrium, we sum the components of all force vectors along
 * each of the three axes (X, Y, Z). If the sum of components for each axis
 * is zero, the net force is the zero vector, and the body is in equilibrium.
 * 1. Read the number of force vectors, n.
 * 2. Initialize sum variables `x`, `y`, and `z` to 0.
 * 3. Loop n times:
 *    a. Read the x, y, and z components of a force vector.
 *    b. Add these components to the respective sum variables.
 * 4. After the loop, check if `x`, `y`, and `z` are all zero.
 * 5. Print "YES" if they are, "NO" otherwise.
 *
 * @complexity
 * Time Complexity: O(n), where n is the number of force vectors, due to the
 * single loop over the input vectors.
 * Space Complexity: O(1), as a constant number of variables are used.
 */
#include <iostream>
using namespace std;

int main() {
  int n; // The number of force vectors.
  cin >> n;
  
  int a, b, c; // Variables to store components of a single vector.
  int x, y, z; // Variables to store the sum of components on each axis.
  
  // Initialize component sums to zero.
  x = y = z = 0;

  /**
   * @block
   * @brief Iterate n times to read all vectors and sum their components.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }
  
  /**
   * @brief Check for the equilibrium condition.
   * @details If the sum of forces along each axis is zero, the net force
   * is zero, and the body is in equilibrium.
   */
  if ((x == 0) && (y == 0) && (z == 0))
    cout << "YES\n";
  else
    cout << "NO\n";
}
