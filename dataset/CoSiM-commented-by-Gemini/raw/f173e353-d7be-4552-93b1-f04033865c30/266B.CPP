/**
 * @file 266B.CPP
 * @brief Simulates the swapping of positions in a queue over a discrete time interval.
 *
 * This program models a common simulation problem, often "Queue at the School".
 * It reads the length of a queue `n`, a time duration `t`, and a string `s`
 * representing the initial arrangement of 'B' (Boys) and 'G' (Girls).
 * The simulation runs for `t` seconds. In each second, any adjacent pair of a Boy
 * followed by a Girl ('BG') swaps their positions. The final arrangement of the
 * queue is then printed.
 *
 * @author f173e353-d7be-4552-93b1-f04033865c30
 *
 * @algorithm Discrete Time Simulation with Local Swaps
 * @complexity
 * Time: O(T * N), where T is the time duration and N is the length of the queue.
 *       The outer loop runs T times and the inner loop runs N times.
 * Space: O(N), to store the string representing the queue.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // `n`: the number of children in the queue.
  // `t`: the total time in seconds for the simulation.
  int n, t;
  cin >> n >> t;
  // `s`: the string representing the initial state of the queue.
  string s;
  cin >> s;

  /**
   * @brief Outer loop representing the passage of time, from 0 to `t` seconds.
   * Invariant: `s` holds the state of the queue at the beginning of each second.
   */
  for (int i = 0; i < t; i++) {
    /**
     * @brief Inner loop to perform swaps for the current second.
     * It iterates through the queue to find adjacent 'BG' pairs.
     */
    for (int j = 0; j < n - 1; j++) {
      /**
       * @brief Condition to find a Boy followed by a Girl.
       * If this pattern is found, a swap is performed.
       */
      if (s[j] == 'B' and s[j + 1] == 'G') {
        // Swap the 'B' and 'G'.
        s[j] = 'G';
        s[j + 1] = 'B';
        /**
         * Inline Logic: Increment `j` to advance past the swapped pair.
         * This is crucial to ensure that a single element does not move more than
         * one position in a single time step, correctly modeling the simultaneous
         * nature of the swaps.
         */
        j++;
      }
    }
  }
  // Print the final state of the queue after `t` seconds.
  cout << s << endl;
}