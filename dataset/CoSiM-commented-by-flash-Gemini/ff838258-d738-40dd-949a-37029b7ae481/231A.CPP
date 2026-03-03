/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - "Team".
 * This program determines the number of problems a team of three friends will implement
 * in a programming contest. A problem is implemented if at least two friends are
 * sure about its solution.
 * Algorithm: Iterates through problems, sums certainty votes, and counts problems with >= 2 votes.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1) - uses a constant amount of memory for variables.
 */
/*
Link: https://codeforces.com/problemset/problem/231/A
@uthor: chavodon
*/

#include <iostream> // Required for standard input/output operations (cin, cout).

using namespace std; // Uses the standard namespace.

int main() {
  // n: total number of problems.
  // i: loop counter.
  // n1, n2, n3: represent the certainty of each of the three friends (1 if sure, 0 if not).
  // sum: accumulates the number of problems that will be implemented.
  int n, i, n1, n2, n3, sum = 0;

  cin >> n; // Reads the total number of problems from standard input.

  /**
   * Block Logic: Processes each problem one by one, determining if it will be implemented.
   * The loop runs 'n' times, once for each problem.
   * Invariant: `sum` holds the number of problems implemented so far.
   */
  for (i = 0; i < n; i++) { // Loop 'n' times, once for each problem.
    // Reads the certainty votes of the three friends for the current problem.
    cin >> n1 >> n2 >> n3;

    // Checks if the sum of certainties is 2 or more, meaning at least two friends are sure.
    if ((n1 + n2 + n3) >= 2)
      sum++; // If so, increment the counter for implemented problems.
  }

  cout << sum; // Prints the total number of problems that will be implemented.

  return 0; // Indicates successful program execution.
}