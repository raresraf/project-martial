/**
 * @file 231A.CPP
 * @brief Solution to a problem involving counting agreed-upon problems among participants.
 *
 * This program reads an integer 'n' representing the number of test cases or problems.
 * For each problem, it reads three integers (a, b, c), which typically represent
 * binary decisions (e.g., 1 for "yes", 0 for "no") from three different participants.
 * The program then determines if at least two participants agreed (i.e., if the sum
 * of a, b, and c is 2 or more). A counter 'result' is incremented for each such case.
 * Finally, the total count of problems with at least two agreements is printed.
 *
 * Algorithm: Iterative processing of 'n' test cases with a conditional sum check.
 * Time Complexity: O(N), where N is the number of test cases.
 * Space Complexity: O(1) for variable storage.
 */
#include <iostream>
using namespace std;

int main() {
  // Declare 'n' for the number of test cases.
  int n;
  // Read the total number of test cases 'n'.
  cin >> n;
  // Declare 'a', 'b', 'c' for participant inputs and 'result' initialized to 0.
  int a, b, c;
  int result = 0;

  /**
   * Block Logic: Loop through each test case or problem.
   * Invariant: In each iteration, inputs for one problem are read, and the agreement
   *            condition is checked, potentially incrementing 'result'.
   */
  for (int i = 0; i < n; i++) {
    // Read the three participant inputs for the current problem.
    cin >> a >> b >> c;

    /**
     * Block Logic: Checks if at least two participants agree on the current problem.
     * Logic: If the sum of 'a', 'b', and 'c' is 2 or greater, it means at least two inputs were 1.
     */
    if ((a + b + c) >= 2) {
      // If the condition is met, increment the 'result' counter.
      result++;
    }
  }

  // Output the final count of problems where at least two participants agreed.
  cout << result << endl;

  return 0; // Indicate successful program execution.
}
