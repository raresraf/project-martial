/**
 * @file 266B.CPP
 * @brief Simulates a queue of boys and girls swapping positions over time.
 * @details This program solves the "Queue at the School" problem from Codeforces.
 *          It simulates a queue where if a boy ('B') is standing in front of a
 *          girl ('G'), they swap places in one second. This process is repeated
 *          for a given amount of time.
 *
 * @algorithm Discrete-time simulation with a single-pass swap.
 * @complexity
 * - Time: O(N * T), where N is the number of children in the queue and T is the
 *   total time in seconds. The simulation runs for T seconds, and each second involves
 *   a linear scan of the queue.
 * - Space: O(N) to store the string representing the queue.
 *
 * @problem_link https://codeforces.com/problemset/problem/266/B
 * @contest_link https://codeforces.com/contest/266/problem/B
 *
 * @author Nikhil Nagrale
 * @codeforces_profile https://codeforces.com/profile/nikhilnagrale2
 * @codechef_profile https://www.codechef.com/users/nikhilnagrale2
 * @github_profile https://github.com/nikhilnagrale2
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: number of children, t: time in seconds.
  int n, t;
  cin >> n >> t;
  // a: string representing the queue of children.
  string a;
  cin >> a;

  /**
   * @brief Block Logic: Simulates the swapping process over `t` seconds.
   * Invariant: At the beginning of each iteration, `a` represents the state of the
   *            queue at the current time step.
   */
  while (t--) {
    /**
     * @brief Block Logic: Performs a single pass to swap all 'B'/'G' pairs for the current second.
     * Invariant: Within a single time step, each child can participate in at most one swap.
     */
    for (int i = 0; i < n - 1; i++)
      // Pre-condition: Check if a boy ('B') is directly in front of a girl ('G').
      if (a[i] == 'B' && a[i + 1] == 'G') {
        swap(a[i], a[i + 1]);
        /**
         * @brief Inline: Skips the next position after a swap.
         * Functional Utility: This ensures that a boy who has just moved to position `i+1`
         * does not immediately move again in the same time step. It correctly models
         * the constraint that each swap takes one second.
         */
        i++;
      }
  }
  // Output the final arrangement of the queue after t seconds.
  cout << a << endl;
  return 0;
}
