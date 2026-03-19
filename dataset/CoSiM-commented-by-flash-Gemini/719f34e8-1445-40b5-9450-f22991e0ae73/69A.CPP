/**
 * @file 69A.CPP
 * @brief Solution to Codeforces Problem 69A: "Young Physicist".
 * Determines if a body is in equilibrium. A body is in equilibrium if the sum of all
 * force vectors acting on it is the zero vector. Given 't' force vectors, each
 * with three components (x, y, z), the task is to check if their resultant sum is zero.
 * Algorithm: Accumulate the sum of x, y, and z components separately across all force vectors.
 * If all three accumulated sums are zero, the body is in equilibrium.
 * Time Complexity: O(N), where N is the number of force vectors.
 * Space Complexity: O(1) as only a few variables are used to store sums.
 */
#include <bits/stdc++.h> // Includes various standard libraries for C++, including iostream.

using namespace std;

/**
 * @brief Main function to solve the "Young Physicist" problem.
 * Reads the number of force vectors and their components, then calculates
 * if the body is in equilibrium.
 */
int main() {
  // Declare variables:
  // 't': total number of force vectors.
  // 'i', 'c', 's': temporary variables to read the x, y, and z components of each force vector.
  // 'a', 'b', 'd': accumulators for the sum of x, y, and z components across all vectors, respectively.
  int t, i, c, s, a = 0, b = 0, d = 0;
  cin >> t; // Reads the total number of force vectors.

  /**
   * Block Logic: Iterates through each force vector, summing its components.
   * Pre-condition: 't' specifies the number of force vectors to process.
   *                'a', 'b', 'd' are initialized to 0.
   * Invariant: After each iteration, 'a', 'b', 'd' hold the accumulated sums of
   *            x, y, z components for the forces processed so far.
   */
  while (t--) { // Loop 't' times.
    cin >> i; // Reads the x-component of the current force vector.
    cin >> c; // Reads the y-component of the current force vector.
    cin >> s; // Reads the z-component of the current force vector.
    
    a = a + i; // Accumulates x-components.
    b = b + c; // Accumulates y-components.
    d = d + s; // Accumulates z-components.
  }
  
  /**
   * Block Logic: Checks if the body is in equilibrium.
   * The body is in equilibrium if and only if the sum of all x, y, and z
   * components are each equal to zero.
   */
  if (a == 0 && b == 0 && d == 0) {
    cout << "YES"; // Body is in equilibrium.
  } else {
    cout << "NO";  // Body is not in equilibrium.
  }
  return 0;
}