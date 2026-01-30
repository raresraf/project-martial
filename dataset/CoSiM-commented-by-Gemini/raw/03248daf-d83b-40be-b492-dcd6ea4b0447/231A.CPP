/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Round #143 (Div. 2) - A. Team
 * @see https://codeforces.com/problemset/problem/231/A
 *
 * @brief This program solves a simple counting problem. It determines how many
 * problems a three-person team will implement based on their individual confidence
 * levels. The team will only solve a problem if at least two members are sure of
 * the solution.
 *
 * Algorithm:
 * - Read the total number of problems `n`.
 * - Iterate `n` times, once for each problem.
 * - In each iteration, read the confidence levels of the three members (1 for sure, 0 for not sure).
 * - Sum the confidence levels. If the sum is 2 or greater, it means at least two members
 *   are sure, so increment a counter for the number of problems to be implemented.
 * - After the loop, print the total count.
 *
 * Time Complexity: O(N), where N is the number of problems. The program iterates
 * through each problem once.
 * Space Complexity: O(1), as the program uses a constant amount of extra space
 * regardless of the input size.
 */
#include <bits/stdc++.h>
using namespace std;
/**
 * @def FAST_INP
 * @brief A macro to speed up C++ I/O operations.
 *
 * This macro unties `cin` from `cout` and disables synchronization with C-style
 * I/O, which is a common optimization in competitive programming to reduce
 * runtime, especially for large inputs.
 */
#define FAST_INP                                                               \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL)

int main() {
  FAST_INP;
  int n, ans = 0, a, b, c;
  // Read the total number of problems.
  cin >> n;
  /**
   * @brief Main loop to process each problem.
   * Invariant: `ans` holds the count of problems that will be implemented
   * among the problems processed so far.
   */
  while (n--) {
    // Read the confidence levels for the current problem.
    cin >> a >> b >> c;
    /**
     * @brief Check if at least two team members are sure.
     * Pre-condition: a, b, and c are either 0 or 1.
     * Logic: The sum of the confidence levels directly corresponds to the
     * number of members who are sure.
     */
    if (a + b + c >= 2)
      ans++;
  }
  // Output the final count of problems to be implemented.
  cout << ans << "\n";
  return 0;
}