/**
 * @file 231A.CPP
 * @brief Counts the number of problems that will be solved based on team confidence.
 *
 * This program solves a classic competitive programming problem. It reads an integer `n`,
 * representing the number of problems. For each problem, it reads three boolean-like
 * integers (0 or 1) which represent the confidence of three team members in solving
 * that problem. If at least two members are confident (i.e., the sum of the three
 * integers is 2 or more), the team will solve the problem. The program outputs the
 * total number of problems that will be solved.
 *
 * @author f112314b-6c49-4203-847e-008ba36d832b
 *
 * @algorithm Simple Counting
 * @complexity
 * Time: O(N), where N is the number of problems, as it processes each problem in a single pass.
 * Space: O(1) for essential data. The unused VLA `arr` would be O(N) if used, but it's optimized away.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  // `n` is the total number of problems to evaluate.
  int n;
  cin >> n;

  // This 2D array is declared but never used in the logic.
  // It's a Variable Length Array (VLA), a non-standard C++ extension.
  int arr[3][n], x, y, z;

  // `count` will store the total number of problems the team decides to solve.
  int count = 0;

  /**
   * @brief Loop to iterate through each of the `n` problems.
   * Pre-condition: `count` is the number of problems solved so far.
   * Invariant: After each iteration, `count` is incremented if the current
   *            problem meets the criteria.
   */
  for (int j = 0; j < n; j++) {
    // Read the confidence levels of the three members for the current problem.
    cin >> x >> y >> z;

    /**
     * @brief Condition to check if the team will solve the problem.
     * The team solves the problem if the sum of their confidence levels (x, y, z)
     * is greater than 1, meaning at least two members are confident.
     */
    if (x + y + z > 1) {
      count++;
    }
  }

  // Output the final count of problems to be solved.
  cout << count << endl;

  return 0;
}