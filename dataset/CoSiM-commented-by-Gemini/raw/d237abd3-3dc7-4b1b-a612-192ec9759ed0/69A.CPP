/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem (Codeforces 69A).
 * @details This program determines if a body is in equilibrium under the influence of N forces.
 * Equilibrium is achieved if the vector sum of all forces is zero.
 *
 * @section algorithm
 * Algorithm:
 * 1. Read the number of force vectors, N.
 * 2. Initialize three variables (x, y, z) to store the sum of the components of all vectors, starting at 0.
 * 3. Loop N times:
 *    a. Read the x, y, and z components of a force vector.
 *    b. Add these components to the corresponding sum variables.
 * 4. After the loop, check if the sums of all x, y, and z components are zero.
 * 5. If all sums are zero, the net force is zero, and the body is in equilibrium.
 *
 * @section complexity
 * - Time Complexity: O(N), because it iterates through N vectors once.
 * - Space Complexity: O(1), as it uses a fixed amount of space for variables.
 */
#include <iostream>

using namespace std;

int main() {
  int n, x = 0, y = 0, z = 0;
  cin >> n;

  /**
   * @brief This loop reads N vectors and calculates the resultant vector.
   * @invariant At each step, x, y, and z hold the sum of components of the
   * vectors read so far.
   */
  for (int i = 0; i < n; ++i) {
    int a, b, c;
    cin >> a >> b >> c;
    x += a;
    y += b;
    z += c;
  }

  /**
   * @brief Checks if the body is in equilibrium.
   * The condition for equilibrium in physics is that the net force acting on
   * the body is zero. This translates to the sum of the components of all
   * force vectors along each axis being zero.
   */
  if (x == 0 && y == 0 && z == 0)
    // Output: "YES" indicates the system is in equilibrium.
    cout << "YES";
  else
    // Output: "NO" indicates the system is not in equilibrium.
    cout << "NO";
}