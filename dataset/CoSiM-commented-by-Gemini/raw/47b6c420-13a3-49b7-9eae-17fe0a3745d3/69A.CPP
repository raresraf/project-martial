/**
 * @file 69A.CPP
 * @brief Solution for a physics equilibrium problem ("Young Physicist").
 * @details The program determines if a body is in equilibrium given 'n' forces
 * acting upon it. A body is in equilibrium if the vector sum of all forces is zero.
 *
 * Algorithm:
 * The condition for equilibrium is that the sum of the components of all force
 * vectors along each of the three axes (x, y, z) must be zero.
 * 1. Initialize three variables (is, js, ks) to store the sum of components for
 *    the x, y, and z axes, respectively.
 * 2. Read the total number of forces, 'n'.
 * 3. Loop 'n' times to process each force vector.
 * 4. In each iteration, read the three components of the force.
 * 5. Add each component to its corresponding sum (is, js, ks).
 * 6. After the loop, check if all three sums are zero.
 * 7. If `is == 0`, `js == 0`, and `ks == 0`, the net force is zero, and the
 *    body is in equilibrium. Print "YES". Otherwise, print "NO".
 *
 * Time Complexity: O(n), where 'n' is the number of force vectors.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;
  int is = 0, js = 0, ks = 0;
  int l, p, q;
  
  /**
   * @block
   * @description This loop reads 'n' force vectors and accumulates their
   * x, y, and z components into the `is`, `js`, and `ks` sums.
   */
  for (int i = 0; i < n; i++) {
    cin >> l >> p >> q;
    is += l;
    js += p;
    ks += q;
  }

  /**
   * @block
   * @description Checks the equilibrium condition. In C++, the '!' operator on an
   * integer evaluates to true if the integer is 0. This line concisely checks
   * if all three force component sums are zero.
   */
  if (!is && !js && !ks) {
    cout << "YES\n";
  } else {
    cout << "NO\n";
  }
}
