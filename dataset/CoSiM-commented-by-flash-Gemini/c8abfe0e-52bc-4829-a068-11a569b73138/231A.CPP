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
#include <math.h>   // Included but not used in this specific snippet.
#include <stdio.h>  // Required for scanf and printf.
#include <string.h> // Included but not used in this specific snippet.

int main() {
  // Declare 'n' for the number of test cases, 'a', 'b', 'c' for participant inputs,
  // 'sum' for the sum of inputs, and 'cnt' initialized to 0 for problems with agreement.
  int n, a, b, c, sum, cnt = 0;

  // Read the total number of test cases 'n'.
  scanf("%d", &n);

  /**
   * Block Logic: Loop through each test case or problem.
   * Invariant: 'n' decreases with each iteration; 'cnt' is incremented if
   *            the agreement condition is met for the current problem.
   */
  while (n--) {
    sum = 0; // Reset sum for each new test case.
    // Read the three binary inputs representing participant decisions for the current problem.
    scanf("%d %d %d", &a, &b, &c);
    // Calculate the sum of the three inputs.
    sum = a + b + c;
    /**
     * Block Logic: Checks if at least two participants agree on the current problem.
     * Logic: If the sum of 'a', 'b', and 'c' is greater than 1, it indicates agreement.
     */
    if (sum > 1)
      // Increment the 'cnt' if the agreement condition is met.
      cnt++;
  }

  // Print the final count of problems for which at least two participants agreed.
  printf("%d\n", cnt);

  return 0; // Indicate successful program execution.
}
