/**
 * @file 231A.CPP
 * @brief Solution to a problem that counts the number of problems for which
 *        at least two out of three participants agree on a solution.
 *
 * This program reads an integer 'n' representing the total number of problems
 * (or test cases) to evaluate. For each problem, it reads three binary inputs
 * (0 or 1) representing the decision of three different participants. If the
 * sum of these three inputs is greater than 1, it signifies that at least two
 * participants agree, and a counter for solved problems is incremented.
 * Finally, the total count of problems with sufficient agreement is printed.
 *
 * Algorithm: Iterative processing of 'n' problems, checking a simple summation
 *            condition for agreement.
 * Time Complexity: O(N), where N is the number of problems/test cases.
 * Space Complexity: O(1) for storing variables.
 */
#include <cstdio> // Required for scanf and printf.

using namespace std; // Using the standard namespace.

int main() {
  // Declare 'n' for the number of problems/test cases.
  int n;
  // Declare 'count' initialized to 0 to store the number of problems with agreement.
  int count = 0;
  // Read the total number of problems 'n'.
  scanf("%d", &n);
  /**
   * Block Logic: Loop through each problem (test case) to evaluate participant agreement.
   * Invariant: 'i' decrements, processing each problem; 'count' is incremented if
   *            the agreement condition is met. The loop runs 'n' times.
   */
  for (int i = n; i > 0; i--) {
    // Declare 'a', 'b', 'c' for participant inputs.
    int a, b, c;
    // Read the three binary inputs representing participant decisions for the current problem.
    scanf("%d %d %d", &a, &b, &c);
    /**
     * Block Logic: Checks if at least two participants agree on the current problem.
     * Logic: If the sum of 'a', 'b', and 'c' is greater than 1, it indicates agreement.
     */
    if (a + b + c > 1) {
      // Increment the 'count' if the agreement condition is met.
      count++;
    }
  }
  // Print the final count of problems for which at least two participants agreed.
  printf("%d\n", count);
  return 0; // Indicate successful program execution.
}