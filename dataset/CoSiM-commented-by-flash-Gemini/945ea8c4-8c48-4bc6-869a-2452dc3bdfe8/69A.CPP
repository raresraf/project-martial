/**
 * @945ea8c4-8c48-4bc6-869a-2452dc3bdfe8/69A.CPP
 * @brief Solution for a competitive programming problem determining if a body is in equilibrium.
 *
 * This program takes a series of 3D force vectors and checks if their sum along each
 * of the X, Y, and Z axes is zero. If all three sums are zero, the body is considered
 * to be in equilibrium. This is a direct implementation of Newton's first law applied
 * to a system of forces.
 *
 * Algorithm:
 * 1. Read the total number of forces 'n'.
 * 2. Initialize three accumulators (result, result1, result2) to zero for X, Y, Z components respectively.
 * 3. For each of the 'n' forces:
 *    a. Read the three integer components (x_i, y_i, z_i) of the force.
 *    b. Add x_i to result, y_i to result1, and z_i to result2.
 * 4. After summing all components, check if all three accumulators (result, result1, result2) are zero.
 * 5. Print "YES" if all sums are zero, otherwise print "NO".
 *
 * Time Complexity: O(N) where N is the number of forces, as each force involves reading 3 integers and 3 additions.
 * Space Complexity: O(1) as only a few variables are used to store sums and temporary force components.
 */

#include <bits/stdc++.h> // Functional Utility: Includes most standard C++ libraries, common in competitive programming.
#include <iostream>      // Functional Utility: Specifically includes std::iostream for cin/cout.

using namespace std;

int main() {
  // Functional Utility: An array to temporarily store force components, though only the first 3 elements are used.
  int x[100];
  int n;
  // Functional Utility: Reads the total number of forces 'n'.
  cin >> n;

  // Functional Utility: Accumulators for the sum of force components along X, Y, and Z axes.
  int result = 0, result1 = 0, result2 = 0;

  // Block Logic: Loops 'n' times to read and sum the components of each force.
  // Pre-condition: 'n' is a positive integer.
  // Invariant: 'result', 'result1', 'result2' will hold the cumulative sum of X, Y, Z components respectively.
  while (n--) {
    // Block Logic: Reads the three components of a single force vector.
    // Pre-condition: Three integers representing force components are available in input.
    // Invariant: 'x[0]', 'x[1]', 'x[2]' will store the current force's components.
    for (int i = 0; i < 3; i++) {
      cin >> x[i];
    }
    // Functional Utility: Adds the X-component of the current force to the total X-sum.
    result = result + x[0];
    // Functional Utility: Adds the Y-component of the current force to the total Y-sum.
    result1 = result1 + x[1];
    // Functional Utility: Adds the Z-component of the current force to the total Z-sum.
    result2 = result2 + x[2];
  }

  // Conditional Logic: Checks if the sum of all components along X, Y, and Z axes are all zero.
  // This condition determines if the body is in equilibrium.
  if (result == 0 && result1 == 0 && result2 == 0) {
    cout << "YES"; // Functional Utility: Prints "YES" if the body is in equilibrium.
  } else {
    cout << "NO"; // Functional Utility: Prints "NO" if the body is not in equilibrium.
  }

  return 0; // Functional Utility: Indicates successful program execution.
}
