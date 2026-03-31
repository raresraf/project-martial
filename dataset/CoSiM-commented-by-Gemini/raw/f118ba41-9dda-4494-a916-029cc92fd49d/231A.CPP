/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem by counting how many sub-problems meet a threshold.
 *
 * This program reads an integer `n`, representing the total number of sub-problems.
 * For each sub-problem, it takes three integer inputs, which can be thought of as
 * votes or confidence levels (typically 0 or 1). If the sum of these three inputs
 * is 2 or greater, it means the sub-problem is "solved". The program's goal is to
 * count and print the total number of solved sub-problems.
 *
 * @author f118ba41-9dda-4494-a916-029cc92fd49d
 *
 * @algorithm Simple Iterative Counting
 * @complexity
 * Time: O(N), where N is the number of sub-problems, because of the single loop.
 * Space: O(1), as the memory usage is constant and does not scale with N.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // `n` stores the total number of sub-problems to evaluate.
  int n;
  cin >> n;

  // `a`, `b`, `c` will store the three integer inputs for each sub-problem.
  int a, b, c;
  // `solved` is a counter for the number of sub-problems that satisfy the condition.
  int solved = 0;

  /**
   * @brief This loop processes each of the `n` sub-problems.
   * Pre-condition: `solved` contains the count of solved problems from previous iterations.
   * Invariant: At the end of each iteration, `solved` is updated if the current
   *            sub-problem's inputs sum to 2 or more.
   */
  for (int i = 0; i < n; i++) {
    // Read the three integer values for the current sub-problem.
    cin >> a >> b >> c;

    /**
     * @brief Checks if the current sub-problem is "solved".
     * The condition `a + b + c >= 2` checks if the sum of the inputs is at least 2.
     * This is the core logic for determining if a sub-problem is counted.
     */
    if (a + b + c >= 2)
      solved++;
  }

  // Print the total number of solved sub-problems.
  cout << solved;
}