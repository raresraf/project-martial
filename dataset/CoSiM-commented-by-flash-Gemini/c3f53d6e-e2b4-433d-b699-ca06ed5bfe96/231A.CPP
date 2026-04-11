/**
 * @file 231A.CPP
 * @brief Solution to a competitive programming problem.
 * This program reads an integer 'n', then iterates 'n' times. In each iteration,
 * it reads three integers (a, b, c) using scanf and calculates their sum.
 * If the sum is greater than or equal to 2, a counter 'total' is incremented.
 * Finally, it prints the total count.
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

  int total = 0;

  /**
   * Block Logic: Processes each test case or iteration.
   * Invariant: 'n' decreases with each iteration, moving towards the termination condition.
   * Pre-condition: 'n' is a non-negative integer representing the number of cases to process.
   */
  while (n--) {
    int a, b, c;
    // Reads three integer inputs for the current case using scanf.
    scanf("%d %d %d", &a, &b, &c);

    // Calculates the sum of the three integers.
    int sum = a + b + c;

    /**
     * Block Logic: Checks if the sum of the three integers is greater than or equal to 2.
     * Pre-condition: 'sum' holds the sum of three valid integers.
     * Invariant: 'total' accumulates the count of cases where the condition is met.
     */
    if (sum >= 2) {
      // Increments the counter if the sum is greater than or equal to 2.
      total++;
    }
  }
  // Prints the final accumulated count.
  cout << total << endl;

  return 0;
}
