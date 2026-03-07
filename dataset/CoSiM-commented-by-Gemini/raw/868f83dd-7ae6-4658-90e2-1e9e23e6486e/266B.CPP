/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem (Codeforces 266B).
 *
 * @details
 * This program simulates a discrete-time process of reordering a queue of boys ('B')
 * and girls ('G'). In each time step, any boy standing immediately in front of a
 * girl will swap places with her. This simulation is run for a specified number
 * of seconds.
 *
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: O(T * N), where T is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) to store the queue as a string.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, t; // n: number of children, t: time in seconds
  cin >> n >> t;
  string s;
  cin >> s;

  /**
   * @brief Block Logic: This outer loop simulates the progression of time for 't' seconds.
   * Invariant: At the start of each iteration, 's' represents the queue's state
   *            at the beginning of the j-th second.
   */
  for (int j = 0; j < t; j++) {
    /**
     * @brief Block Logic: A single-pass sweep to perform all swaps for the current second.
     * Logic: Identifies all adjacent ('B', 'G') pairs and swaps them.
     */
    for (int i = 0; i < n; i++) {
      // Pre-condition for a swap: A boy 'B' is directly ahead of a girl 'G'.
      // The loop condition `i < n` is technically loose, but works because s[n] will be '\0'.
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';
        s[i + 1] = 'B';
        // Inline: Skips the next index. This is crucial as it ensures that a boy
        // who has just swapped places cannot participate in another swap within the
        // same time step, correctly modeling the simultaneous nature of the swaps.
        i++;
      }
    }
  }

  // Output the final state of the queue after 't' seconds.
  cout << s << endl;
}
