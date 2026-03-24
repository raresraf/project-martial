/**
 * @file 69A.CPP
 * @brief Solution to a competitive programming problem: "Young Physicist".
 *
 * Determines if a system of 'n' 3D force vectors is in equilibrium.
 * A system is in equilibrium if the sum of all x-components, y-components,
 * and z-components of the forces are each zero.
 *
 * Algorithm: Iterative summation of vector components followed by a zero-sum check.
 * Time Complexity: O(N), where N is the number of force vectors.
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function to solve the "Young Physicist" problem.
 * Reads 'n' 3D force vectors, calculates the sum of their components,
 * and determines if the system is in equilibrium.
 * @return 0 on successful execution.
 */
int main() {
  /**
   * Declares variables:
   * 'n': The number of force vectors.
   * 'x', 'y', 'z': Components of each individual force vector.
   * 'a', 'b', 'c': Accumulators for the sum of x, y, and z components, respectively.
   * Initializes accumulators 'a', 'b', and 'c' to zero.
   */
  int n, x, y, z, a, b, c;
  a = b = c = 0;
  /**
   * Reads the total number of force vectors 'n' from standard input.
   * Pre-condition: Standard input provides an integer 'n'.
   *
   * Block Logic: Iterates 'n' times to read each 3D force vector and accumulate its components.
   * Invariant: After each iteration, 'a', 'b', 'c' hold the cumulative sum of components
   *            for vectors processed so far.
   * Pre-condition: 'n' has been read and is a non-negative integer.
   */
  cin >> n;

  for (int i = 0; i < n; i++) {
    /**
     * Block Logic: Reads the x, y, and z components of the current force vector.
     * Then, adds these components to their respective accumulators.
     * Pre-condition: Standard input provides three integers (x, y, z).
     * Invariant: 'a', 'b', 'c' are updated with the current vector's contributions.
     */
    cin >> x >> y >> z;
    a += x;
    b += y;
    c += z;
  }
  /**
   * Block Logic: Checks if the system of forces is in equilibrium.
   * If the sum of all x, y, and z components are all zero, the system is in equilibrium.
   * Outputs "YES" if in equilibrium, "NO" otherwise.
   * Post-condition: "YES" or "NO" is printed to standard output.
   */
  if (a == 0 && b == 0 && c == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}