/**
 * @file 231A.CPP
 * @brief Solution to a problem involving counting agreed-upon problems among participants.
 *
 * This program reads an integer 't' representing the number of problems.
 * For each problem, it reads three integers (a, b, c) which likely represent
 * whether three different participants agree to solve the problem (1 for agree, 0 for disagree).
 * If at least two participants agree (i.e., their sum is > 1), a counter is incremented.
 * Finally, the total count of problems with at least two agreements is printed.
 *
 * Algorithm: Iterative processing of 't' test cases with a conditional sum check.
 * Time Complexity: O(T), where T is the number of test cases.
 * Space Complexity: O(1) for variable storage.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // Declare variables: 't' for the number of test cases, 'a', 'b', 'c' for participant inputs,
  // and 'sum' initialized to 0 to count problems with agreement.
  int t, a, b, c, sum = 0;
  /**
   * Block Logic: Loop through each test case.
   * Invariant: In each iteration, 't' is decremented, and inputs for one problem are processed.
   * Pre-condition: 't' is read from input, determining the number of iterations.
   */
  for (scanf("%d", &t); t--;) {
    // Read the three participant inputs for the current problem.
    scanf("%d %d %d", &a, &b, &c);
    /**
     * Block Logic: Check if at least two participants agree on the current problem.
     * Logic: If the sum of 'a', 'b', and 'c' is greater than 1, it means at least two are 1.
     */
    if (a + b + c > 1)
      // If the condition is met, increment the counter 'sum'.
      sum++;
  }
  // Print the final count of problems where at least two participants agreed.
  printf("%d\n", sum);
  return 0;
}
