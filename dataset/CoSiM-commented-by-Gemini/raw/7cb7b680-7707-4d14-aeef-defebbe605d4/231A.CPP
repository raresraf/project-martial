/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A, "Team".
 * @details This program calculates how many problems a team of three will
 * implement based on their confidence. A problem is implemented if at least
 * two of the three members are sure about the solution.
 *
 * Algorithm:
 * 1. Read an integer `n`, the total number of problems.
 * 2. Initialize a counter `c` to 0. This counter will track the number of
 *    problems to be implemented.
 * 3. Loop `n` times to evaluate each problem.
 * 4. Inside the loop, read the three members' opinions (`p`, `v`, `t`),
 *    represented as 1 (sure) or 0 (not sure).
 * 5. Sum the three opinions. If the sum is greater than 1 (i.e., 2 or 3),
 *    increment the counter `c`.
 * 6. After the loop, print the final value of `c`.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k, c = 0; // The variable `k` is declared but not used.
  cin >> n;
  /**
   * @brief This loop iterates through each of the `n` problems to be evaluated.
   */
  for (int i = 0; i < n; i++) {
    int p, v, t;
    cin >> p >> v >> t;
    /**
     * @brief Checks if at least two members are sure about the solution.
     * If the sum of their opinions is greater than 1, the condition is met.
     */
    if ((p + v + t) > 1)
      c++;
  }
  cout << c;
  return 0;
}
