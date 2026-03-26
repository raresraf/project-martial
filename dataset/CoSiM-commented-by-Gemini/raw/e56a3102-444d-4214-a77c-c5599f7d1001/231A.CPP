/**
 * @file 231A.CPP
 * @brief Solution for a counting problem based on confidence levels.
 * @details The program determines how many problems a team of three will solve.
 * It reads an integer 'n', the total number of problems. For each problem, it
 * reads three boolean-like integers (0 or 1) representing the confidence of each team member.
 * If at least two members are confident (the sum is >= 2), the problem is considered solvable.
 * The program outputs the total count of solvable problems.
 *
 * @algorithm Simple Counting
 * @complexity
 * - Time: O(N), where N is the number of problems, as the outer loop runs N times and the inner loop is constant.
 * - Space: O(1), as only a few variables are used for storage regardless of input size.
 */

#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  int n;
  // Read the total number of problems.
  cin >> n;
  // 'res' will store the count of problems to be implemented.
  int res = 0;
  int t; // Temporary variable to read each confidence level.
  
  /**
   * @block Pre-condition: 'n' is the number of problems to evaluate.
   * This loop iterates through each problem.
   * @invariant 'res' holds the count of solvable problems from the first i-1 iterations.
   */
  for (int i = 0; i < n; i++) {
    int a = 0; // 'a' will sum the confidence levels for the current problem.
    /**
     * @block This loop reads the three confidence levels for the current problem.
     */
    for (int j = 0; j < 3; j++) {
      cin >> t;
      a += t;
    }
    /**
     * @block Logic: If the sum of confidence levels is 2 or more, it means at least two
     * members are sure. The problem will be implemented.
     */
    if (a > 1)
      res++;
  }
  
  // Output the total count of solvable problems.
  cout << res << "
";
  return 0;
}
