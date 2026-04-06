/**
 * @file 69A.CPP
 * @brief Solution for Codeforces Problem 69A - "Young Physicist".
 * This program determines if a system of N three-dimensional vectors is in equilibrium.
 * A system of vectors is in equilibrium if the sum of all x-components,
 * the sum of all y-components, and the sum of all z-components are each equal to zero.
 *
 * @problem_statement
 * Given N force vectors, each with x, y, and z components, determine if the body
 * is in equilibrium.
 *
 * @algorithm
 * 1. Read the number of vectors `n`.
 * 2. Initialize `xsum`, `ysum`, and `zsum` to zero.
 * 3. Loop `n` times:
 *    a. Read the x, y, and z components of the current vector.
 *    b. Add the read components to their respective sums (`xsum`, `ysum`, `zsum`).
 * 4. After the loop, check if `xsum`, `ysum`, and `zsum` are all zero.
 * 5. Print "YES" if all sums are zero, otherwise print "NO".
 *
 * Time Complexity: O(N), as the program iterates through the N vectors once.
 * Space Complexity: O(1), as only a few fixed-size variables are used.
 */
#include <iostream>
using namespace std;
int main() {
  /**
   * @brief Main function to determine if a system of vectors is in equilibrium.
   * Reads the number of vectors, then iteratively reads each vector's components,
   * accumulates their sums, and finally checks if all three component sums are zero.
   * @return 0 on successful execution.
   */
  int n; ///< @var n The number of force vectors to be processed.
  cin >> n; ///< Read the total number of vectors.

  int xsum = 0; ///< @var xsum Accumulator for the sum of all x-components.
  int ysum = 0; ///< @var ysum Accumulator for the sum of all y-components.
  int zsum = 0; ///< @var zsum Accumulator for the sum of all z-components.

  /**
   * @brief Loop to read and accumulate components of N vectors.
   * @invariant `n` decrements with each iteration, counting down the remaining vectors to process.
   * `xsum`, `ysum`, `zsum` accumulate the sum of components read so far.
   * @post `xsum`, `ysum`, `zsum` hold the total sum of x, y, and z components, respectively,
   * for all N input vectors.
   */
  while (n--) {
    int x, y, z; ///< @var x, y, z Temporary variables to store components of the current vector.
    cin >> x >> y >> z; // Read the components of the current vector.
    xsum += x; // Add x-component to xsum.
    ysum += y; // Add y-component to ysum.
    zsum += z; // Add z-component to zsum.
  }

  /**
   * @brief Checks if the system of vectors is in equilibrium.
   * @details A system is in equilibrium if the sum of all x, y, and z components
   * are simultaneously zero.
   */
  if (xsum == 0 && ysum == 0 && zsum == 0) {
    cout << "YES" << '\n'; // Print "YES" if in equilibrium.
  } else
    cout << "NO" << '\n'; // Print "NO" otherwise.
}
