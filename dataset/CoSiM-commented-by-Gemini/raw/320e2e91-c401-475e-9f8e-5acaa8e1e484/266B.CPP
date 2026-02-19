/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a state-reordering process based on local adjacency rules.
 * This program simulates the changes in a queue of boys ('B') and girls ('G') at a school
 * over 't' seconds. In each second, any boy standing immediately in front of a girl swaps
 * places with her.
 *
 * @algorithm Iterative single-pass swap simulation.
 *
 * @time_complexity O(T * N), where T is the time in seconds and N is the number of children in the queue.
 * @space_complexity O(N) to store the state of the queue as a string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  char z; // Unused variable from original code.

  /**
   * Block Logic: Orchestrates the temporal progression of the simulation over 't' seconds.
   * Invariant: At the start of each iteration, the string 's' represents the queue's
   *            state at the beginning of the current time step.
   */
  for (; t--;) {
    /**
     * Block Logic: Performs a single-pass sweep across the queue to resolve all possible swaps for the current time step.
     * Logic: Identifies adjacent 'B' (Boy) and 'G' (Girl) pairs and performs a local state
     *        transition (swap) to move the girl ahead of the boy.
     */
    for (int i = 0; i < n; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        // Inline: Skips the next index to prevent a single 'B' from moving
        // more than one position per time step, ensuring atomic movement.
        i++;
      }
    }
  }
  // Output the final state of the queue after 't' seconds.
  cout << s;
}
