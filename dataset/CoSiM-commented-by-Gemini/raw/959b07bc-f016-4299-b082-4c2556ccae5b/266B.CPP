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
 * - Space: O(N) to store the string representing the queue.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // n: number of children, t: time in seconds, s: the queue.
  int n, t;
  string s;
  cin >> n >> t >> s;

  /**
   * @brief Block Logic: Simulates the process for `t` seconds.
   */
  while (t--) {
    /**
     * @brief Block Logic: A single pass over the queue to perform swaps for one second.
     * This loop starts from the second element to facilitate comparing with the previous one.
     */
    for (int i = 1; i < n; i++) {
      // Pre-condition: Check if a girl is preceded by a boy.
      if (s[i] == 'G' && s[i - 1] == 'B') {
        // Perform the swap manually.
        s[i] = 'B';
        s[i - 1] = 'G';
        /**
         * @brief Inline: Advance the index to prevent multiple moves in one time step.
         * Functional Utility: After swapping `s[i]` and `s[i-1]`, the boy is now at `i`.
         * By incrementing `i`, the next check will be at `i+2` (relative to original `i`),
         * preventing this same boy from moving again in the current time step.
         */
        ++i;
      }
    }
  }

  // Print the final state of the queue.
  cout << s << endl;
  return 0;
}
