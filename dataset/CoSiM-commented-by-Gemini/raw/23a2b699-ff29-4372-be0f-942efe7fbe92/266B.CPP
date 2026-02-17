/**
 * @file 266B.CPP
 * @brief Simulates the rearrangement of a queue over a fixed time duration.
 * @details This program models a queue of boys ('B') and girls ('G'). Over 't'
 * seconds, any boy standing immediately in front of a girl swaps places with her.
 * This process is repeated for the specified number of seconds to determine the
 * final arrangement of the queue.
 *
 * @algorithm Iterative single-pass swap simulation.
 * @time_complexity O(t * n), where 't' is the number of time steps and 'n' is the
 * length of the queue. For each time step, the algorithm performs a single pass
 * over the string.
 * @space_complexity O(n) to store the string representing the queue.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, t, a; // n: length of queue, t: time steps. 'a' is unused.
  cin >> n >> t;
  string s;
  cin >> s;

  int i, j, k; // Loop counters. 'k' is unused.

  /**
   * Block Logic: Simulates the passage of time for 't' seconds.
   * Invariant: At the beginning of each outer loop iteration, 's' represents the
   * state of the queue at the current time step.
   */
  for (i = 0; i < t; i++) {
    /**
     * Block Logic: A single pass through the queue to perform swaps for one time step.
     */
    for (j = 1; j < n; j++) {
      // Pre-condition: Check if a 'B' is immediately followed by a 'G'.
      if (s[j - 1] == 'B' && s[j] == 'G') {
        swap(s[j - 1], s[j]);
        // Inline: The counter 'j' is incremented to move past the swapped pair.
        // This ensures that the newly positioned 'G' does not participate in another
        // swap within the same time step, correctly modeling one swap per second.
        j++;
      }
    }
  }
  // Output the final state of the queue after 't' seconds.
  cout << s;

  return 0;
}
