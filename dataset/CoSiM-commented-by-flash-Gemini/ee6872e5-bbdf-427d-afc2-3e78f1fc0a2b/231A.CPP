/**
 * @file 231A.CPP
 * @brief Solution to a competitive programming problem.
 * This program reads an integer 'n', then iterates 'n' times. In each iteration,
 * it reads three integers (a, b, c) and checks if their sum is greater than 1.
 * If the sum is greater than 1, a counter 'ans' is incremented. Finally, it
 * prints the total count.
 * Algorithm: Iterative summation and conditional counting.
 * Time Complexity: O(N) where N is the input integer.
 * Space Complexity: O(1)
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function where the program execution begins.
 * Orchestrates reading input, processing, and printing the result.
 */
int main() {
  int n;
  // Reads the number of test cases or iterations.
  cin >> n;
  int ans = 0;
  /**
   * Block Logic: Processes each test case or iteration.
   * Invariant: 'n' decreases with each iteration, moving towards the termination condition.
   * Pre-condition: 'n' is a non-negative integer representing the number of cases to process.
   */
  while (n != 0) {
    int a, b, c;
    // Reads three integer inputs for the current case.
    cin >> a >> b >> c;
    /**
     * Block Logic: Checks if the sum of the three integers exceeds 1.
     * Pre-condition: 'a', 'b', and 'c' are valid integers.
     * Invariant: 'ans' accumulates the count of cases where the condition is met.
     */
    if (a + b + c > 1)
      // Increments the counter if the sum is greater than 1.
      ans += 1;
    // Decrements the counter for remaining cases.
    n--;
  }
  // Prints the final accumulated count.
  cout << ans << endl;
  return 0;
}