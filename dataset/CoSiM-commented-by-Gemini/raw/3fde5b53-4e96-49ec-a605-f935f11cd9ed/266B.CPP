/**
 * @file 266B.CPP
 * @brief Solution to Codeforces problem 266B - Queue at the School.
 *
 * @algorithm
 * This program simulates a queue of boys ('B') and girls ('G'). At each time
 * step, if a boy is standing directly in front of a girl, they swap places.
 * This process is repeated for a given number of time steps.
 *
 * The algorithm is as follows:
 * 1. Read the queue length `n`, the total time `t`, and the initial queue string `s`.
 * 2. Loop `t` times to simulate each second passing.
 * 3. Inside the time loop, iterate through the queue string from left to right.
 * 4. If a 'B' is found at index `i` and a 'G' at `i+1`, swap them.
 * 5. After a swap, increment the loop counter `i` an extra time. This is a
 *    critical step to ensure that a person only moves once per time step.
 *    For example, in "BBG", the first 'B' swaps with 'G' to become "BGB".
 *    Without the extra `i++`, the new 'B' at index 1 would immediately swap
 *    with the 'G' at index 0, resulting in "GBB" in one time step, which is incorrect.
 * 6. After all time steps are simulated, print the final state of the queue.
 *
 * @complexity
 * - Time Complexity: O(t * n), where `t` is the number of time steps and `n` is the length of the queue.
 * - Space Complexity: O(n) to store the queue string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  int t;
  cin >> n >> t;
  string s;
  cin >> s;

  /**
   * @brief This loop simulates the passage of time for `t` seconds.
   */
  while (t--) {
    /**
     * @brief In each second, this loop performs a single pass over the queue
     * to swap adjacent 'B' and 'G' pairs.
     */
    for (int i = 0; i < n - 1; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';
        s[i + 1] = 'B';
        /**
         * @brief Skip the next position to ensure each person moves at most once
         * per time step. After swapping s[i] and s[i+1], the person at the new
         * s[i+1] (the boy) has completed their move for this time step.
         */
        i++;
      }
    }
  }
  // Print the final state of the queue.
  cout << s;
}
