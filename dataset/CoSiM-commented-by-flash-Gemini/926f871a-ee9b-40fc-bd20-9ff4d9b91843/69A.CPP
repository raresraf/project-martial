/**
 * @file 69A.CPP
 * @brief Determines if a system of forces is in equilibrium.
 *
 * This program reads an integer 'n' representing the number of forces acting on an object.
 * For each force, it reads three integer components (x, y, z). It then calculates
 * the net force along each axis (sum of x, sum of y, sum of z). If all three net
 * force components are zero, the system is in equilibrium, and the program outputs "YES".
 * Otherwise, it outputs "NO".
 *
 * Algorithm: Vector summation. Reads 'n' 3D vectors and sums their components.
 * Time Complexity: O(N), where N is the number of forces, as each force is processed once.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h> // Includes most standard library headers.
using namespace std;     // Uses the standard namespace.
int main() {
  /**
   * @brief Main function: Determines if a system of 'n' forces is in equilibrium.
   * @return 0 implied by function end, but explicitly outputs "YES" or "NO".
   */
  int n, x, y, z;     // Declare variables for number of forces (n), and individual force components (x, y, z).
  int x1 = 0;         // Accumulator for sum of x-components.
  int y1 = 0;         // Accumulator for sum of y-components.
  int z1 = 0;         // Accumulator for sum of z-components.
  int total;          // Declared but unused.
  int sum = 0;        // Declared but unused.
  cin >> n;           // Read the number of forces 'n'.
  /**
   * @brief Loop to read 'n' sets of force components and accumulate their sums.
   * @pre 'n' is a positive integer. 'x1', 'y1', 'z1' are initialized to 0.
   * @post 'x1', 'y1', 'z1' contain the total sum of force components along x, y, and z axes respectively.
   * @invariant 'i' iterates from 1 to n, processing one force vector in each iteration.
   */
  for (int i = 1; i <= n; i++) {
    cin >> x >> y >> z; // Read the x, y, and z components of the current force.
    x1 = x1 + x;         // Add x-component to total x-sum.
    y1 = y1 + y;         // Add y-component to total y-sum.
    z1 = z1 + z;         // Add z-component to total z-sum.
  }
  /**
   * @brief Conditional block to check if the net force in all directions is zero.
   * @pre 'x1', 'y1', 'z1' hold the accumulated sums of force components.
   * @invariant If all three sums are zero, the object is in equilibrium.
   */
  if (x1 == 0 && y1 == 0 && z1 == 0) { // Check for equilibrium condition.
    cout << "YES"; // Output "YES" if the system is in equilibrium.
  } else {
    cout << "NO"; // Output "NO" otherwise.
  }
}
