/**
 * @file 266B.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Queue at the School").
 * This program simulates the process of boys and girls swapping positions in a
 * queue over a given number of time steps.
 *
 * @algorithm Iterative single-pass swap simulation.
 * @time_complexity O(T * N), where T is the number of time steps and N is the length of the queue.
 * @space_complexity O(N) to store the queue sequence as a string.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int n, t;
  string s;
  cin >> n >> t >> s;

  /**
   * @block
   * @brief Orchestrates the temporal progression of the simulation for `t` seconds.
   * Invariant: At the start of each `while` loop iteration, `s` represents the
   * state of the queue at the beginning of that time step.
   */
  while (t--) {
    /**
     * @block
     * @brief Performs a single-pass sweep to resolve 'BG' inversions.
     * This loop iterates through the queue, identifying any instance where a boy ('B')
     * is immediately followed by a girl ('G'), and swaps them.
     */
    for (int i = 1; i < n; ++i) {
      if (s[i] == 'G' && s[i - 1] == 'B') {
        s[i] = 'B';
        s[i - 1] = 'G';
        /**
         * @inline
         * Advances the loop counter by an extra step after a swap. This ensures
         * that the just-swapped boy does not participate in another swap in the
         * same time step, correctly modeling that each person can only move one
         * position per second.
         */
        ++i;
      }
    }
  }
  cout << s << endl;
  return 0;
}