/**
 * @file 231A.CPP
 * @brief Solution to a problem that counts the number of problems for which
 *        at least two out of three participants agree on a solution.
 *
 * This program reads an integer 'n' representing the total number of problems
 * (or test cases) to evaluate. For each problem, it reads three binary inputs
 * (0 or 1) representing the decision of three different participants. If the
 * sum of these three inputs is 2 or more, it signifies that at least two
 * participants agree, and a counter for solved problems is incremented.
 * Finally, the total count of problems with sufficient agreement is printed.
 *
 * Algorithm: Iterative processing of 'n' problems, checking a simple summation
 *            condition for agreement.
 * Time Complexity: O(N), where N is the number of problems/test cases.
 * Space Complexity: O(1) for storing variables.
 */
#include <iostream>

int main() {
  // Functional Utility: Disables synchronization between C-style I/O and C++ streams,
  // speeding up input/output operations, common in competitive programming.
  std::ios::sync_with_stdio(false);
  // Declare 'n' for the number of problems/test cases.
  int n;
  // Read the total number of problems 'n'.
  std::cin >> n;
  // Declare 'a', 'b', 'c' for participant inputs and 'count' initialized to 0.
  int a, b, c, count = 0;
  /**
   * Block Logic: Loop through each problem (test case) to evaluate participant agreement.
   * Invariant: In each iteration, decisions for one problem are read, and 'count'
   *            is incremented if the agreement condition is met. 'n' decrements.
   */
  while (n--) {
    // Read the three binary inputs representing participant decisions for the current problem.
    std::cin >> a >> b >> c;
    /**
     * Block Logic: Checks if at least two participants agree on the current problem.
     * Logic: If the sum of 'a', 'b', and 'c' is 2 or greater, it indicates agreement.
     */
    if ((a + b + c) >= 2) {
      // Increment the 'count' if the agreement condition is met.
      count++;
    }
  }
  // Output the final count of problems for which at least two participants agreed.
  std::cout << count << std::endl;
  return 0; // Indicate successful program execution.
}