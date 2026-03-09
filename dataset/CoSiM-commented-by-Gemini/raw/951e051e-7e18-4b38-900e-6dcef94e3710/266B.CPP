/**
 * @file 266B.CPP
 * @brief Simulates a queue of boys and girls swapping positions over time.
 * @details This program solves the "Queue at the School" problem from Codeforces.
 *          It simulates a queue where if a boy ('B') is standing in front of a
 *          girl ('G'), they swap places in one second. This process is repeated
 *          for a given amount of time `t`.
 *
 * @algorithm Discrete-time simulation with a single-pass swap.
 * @complexity
 * - Time: O(N * T), where N is the length of the queue and T is the total time.
 *   The simulation involves T iterations, each performing a linear scan of the queue.
 * - Space: O(N) to store the string representing the queue.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // I/O optimization for speed.
  ios::sync_with_stdio(false);

  // n: number of children in the queue, t: time in seconds.
  int n, t;
  cin >> n >> t;
  // s: the string representing the queue.
  string s;
  cin >> s;

  /**
   * @brief Block Logic: Simulates the process for `t` seconds.
   * This is the outer loop representing the passage of time.
   */
  for (int j = 0; j < t; ++j) {
    /**
     * @brief Block Logic: A single pass over the queue to perform swaps for one second.
     */
    for (int i = 0; i < n - 1; ++i) {
      // Pre-condition: Check if a boy is immediately followed by a girl.
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        /**
         * @brief Inline: Advance the index to prevent multiple moves by the same boy in one second.
         * Functional Utility: After a boy at `i` moves to `i+1`, we should start the next
         * check from `i+2`. Incrementing `i` here achieves this, ensuring that each
         * swap is an atomic one-second action.
         */
        i++;
      }
    }
  }

  // Print the final state of the queue.
  cout << s << "\n";
  return 0;
}