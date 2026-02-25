/**
 * @file 69A.CPP
 * @brief Solves the "Young Physicist" problem from Codeforces (Problem 69A).
 *        This program determines if a system of 'm' force vectors acting on an
 *        object results in equilibrium. Equilibrium is achieved if the sum of
 *        all force vectors (component-wise) is the zero vector.
 * Algorithm: Vector Summation
 * Time Complexity: O(M), where M is the number of force vectors.
 * Space Complexity: O(1).
 */

#include <bits/stdc++.h> // Includes various standard libraries, e.g., iostream.
using namespace std;

typedef long long ll; // Defines 'll' as a shorthand for long long.

/**
 * @brief Main function to read force vectors, calculate their sum,
 *        and determine if the object is in equilibrium.
 * @return 0 if the program executes successfully.
 */
int main() {
  // Declare variables:
  // x, y, z: components of a single force vector.
  // a, b, c: accumulators for the sum of x, y, and z components, respectively.
  // m: total number of force vectors.
  // (n, i, j, k are declared but unused in this snippet).
  int x, y, z, a = 0, b = 0, c = 0, m, n, i, j, k;
  
  // Read the number of force vectors from standard input.
  cin >> m;
  
  /**
   * Block Logic: Reads 'm' force vectors and sums their components.
   * Invariant: 'm' decrements after each vector, 'a', 'b', 'c' accumulate the sums.
   */
  while (m--) { // Loop 'm' times, once for each force vector.
    // Read the x, y, and z components of the current vector.
    cin >> x >> y >> z;
    // Accumulate the components.
    a += x, b += y, c += z;
  }
  
  /**
   * Block Logic: Checks if the resultant force vector is the zero vector.
   * Pre-condition: 'a', 'b', and 'c' contain the sums of all vector components.
   * Post-condition: Prints "YES" if the object is in equilibrium, "NO" otherwise.
   */
  if (a == 0 && b == 0 && c == 0)
    cout << "YES" << endl; // Output "YES" if the sum of forces is zero (equilibrium).
  else
    puts("NO"); // Output "NO" otherwise. `puts` prints a string followed by a newline.

  return 0; // Indicate successful program termination.
}
