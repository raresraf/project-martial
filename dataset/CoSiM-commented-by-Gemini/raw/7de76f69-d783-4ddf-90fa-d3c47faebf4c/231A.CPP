/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A, "Team".
 * @details This program counts the number of problems a team will solve.
 * The condition for solving a problem is that at least two of the three team
 * members are sure about the solution.
 *
 * Algorithm:
 * 1. Read the number of problems, `n`.
 * 2. Initialize a counter to 0.
 * 3. Loop `n` times, once for each problem.
 * 4. In each iteration, read the three integer opinions (1 for sure, 0 for not).
 * 5. A ternary operator checks if the sum of the three opinions is 2 or greater.
 * 6. If the condition is true, 1 is added to the counter; otherwise, 0 is added.
 * 7. After the loop, the final count is printed.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, count = 0, p, v, t;
  cin >> n;
  /**
   * @brief This loop iterates through each of the `n` problems.
   */
  while (n--) {
    cin >> p >> v >> t;
    /**
     * @brief This line uses a ternary operator to concisely implement the logic.
     * It increments `count` by 1 if `(p + v + t)` is 2 or more, and by 0
     * otherwise.
     */
    count += ((p + v + t) >= 2) ? 1 : 0;
  }
  cout << count;
  return 0;
}