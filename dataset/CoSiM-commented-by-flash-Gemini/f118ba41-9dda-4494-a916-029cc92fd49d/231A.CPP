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
#include <bits/stdc++.h>

using namespace std;

int main() {
  // Declare 'n' for the number of problems/test cases.
  int n;
  // Read the total number of problems 'n'.
  cin >> n;

  // Declare 'a', 'b', 'c' for participant inputs and 'solved' initialized to 0.
  int a, b, c;
  int solved = 0;

  /**
   * Block Logic: Loop through each problem (test case) to evaluate participant agreement.
   * Invariant: In each iteration, decisions for one problem are read, and 'solved'
   *            is incremented if the agreement condition is met.
   */
  for (int i = 0; i < n; i++) {
    // Read the three binary inputs representing participant decisions for the current problem.
    cin >> a >> b >> c;
    /**
     * Block Logic: Checks if at least two participants agree on the current problem.
     * Logic: If the sum of 'a', 'b', and 'c' is 2 or greater, it indicates agreement.
     */
    if (a + b + c >= 2)
      // Increment the 'solved' counter if the agreement condition is met.
      solved++;
  }

  // Output the final count of problems for which at least two participants agreed.
  cout << solved;
  return 0; // Indicate successful program execution.
}
