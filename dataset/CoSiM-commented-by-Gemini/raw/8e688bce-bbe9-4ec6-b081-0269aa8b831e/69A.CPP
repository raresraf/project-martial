/**
 * @file 69A.CPP
 * @brief Solution for a vector equilibrium problem (e.g., Codeforces 69A "Young Physicist").
 * @details This program determines if a body is in equilibrium by checking if the net force
 * acting upon it is zero. It does this by summing the x, y, and z components of all
 * given force vectors and verifying that the resultant vector is (0, 0, 0).
 *
 * @algorithm Vector Summation
 * @complexity
 * - Time: O(n), where 'n' is the number of force vectors, as the program iterates
 *   through each vector once.
 * - Space: O(1), as it uses a fixed number of variables to store the component sums,
 *   regardless of the number of input vectors.
 */
#include <iostream>
using namespace std;

int main() {
  // n: the number of force vectors.
  // x, y, z: the components of a single force vector.
  // a, b, c: the sums of the x, y, and z components of all vectors, respectively.
  int n, x, y, z, a, b, c;
  
  // Initialize the sum of components to zero.
  a = b = c = 0;
  cin >> n;

  /**
   * @block Vector Summation Loop
   * @logic Reads each of the 'n' force vectors and adds their components to the running totals.
   */
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    a += x; // Sum of x-components.
    b += y; // Sum of y-components.
    c += z; // Sum of z-components.
  }

  /**
   * @block Equilibrium Check
   * @logic A body is in equilibrium if the net force on all axes is zero.
   */
  if (a == 0 && b == 0 && c == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
