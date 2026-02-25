/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces (Problem 69A).
 *        This program determines if a system of 'n' force vectors acting on an
 *        object results in equilibrium. Equilibrium is achieved if the sum of
 *        all force vectors (component-wise) is the zero vector.
 * Algorithm: Vector Summation
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(1).
 */

#include <bits/stdc++.h> // Includes various standard libraries, e.g., iostream.
using namespace std;

/**
 * @brief Main function to read force vectors, calculate their sum,
 *        and determine if the object is in equilibrium.
 * @return 0 if the program executes successfully.
 */
int main() {
  // Declare variables:
  // n: total number of force vectors.
  // x, y, z: accumulators for the sum of x, y, and z components, respectively.
  // xa, ya, za: components of a single force vector.
  int n = 0, x = 0, y = 0, z = 0;
  
  // Read the number of force vectors from standard input.
  cin >> n;
  
  /**
   * Block Logic: Reads 'n' force vectors and sums their components.
   * Invariant: 'n' decrements after each vector, 'x', 'y', 'z' accumulate the sums.
   */
  while (n--) { // Loop 'n' times, once for each force vector.
    int xa, ya, za; // Declare variables for the components of the current vector.
    // Read the x, y, and z components of the current vector.
    cin >> xa >> ya >> za;
    // Accumulate the components.
    x += xa;
    y += ya;
    z += za;
  }
  
  /**
   * Block Logic: Checks if the resultant force vector is the zero vector.
   * Pre-condition: 'x', 'y', and 'z' contain the sums of all vector components.
   * Post-condition: Prints "YES" if the object is in equilibrium, "NO" otherwise.
   */
  if (x == 0 and y == 0 and z == 0) {
    cout << "YES" << endl; // Output "YES" if the sum of forces is zero (equilibrium).
  } else {
    cout << "NO" << endl; // Output "NO" otherwise.
  }
  return 0; // Indicate successful program termination.
}