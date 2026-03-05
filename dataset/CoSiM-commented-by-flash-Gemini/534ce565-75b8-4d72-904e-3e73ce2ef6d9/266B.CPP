/**
 * @file 266B.CPP
 * @brief Solution to a queue reordering problem where 'B' (Boy) and 'G' (Girl)
 *        elements swap positions if 'B' is followed by 'G'. This simulates
 *        a discrete-time process of reordering.
 *
 * Algorithm: Iterative single-pass swap simulation.
 *            The simulation progresses through discrete time steps. In each time step,
 *            the queue is scanned from left to right, and any 'BG' pair is swapped
 *            to 'GB'. To ensure each 'B' moves at most one position per time step,
 *            the index `i` is incremented after a swap.
 * Time Complexity: O(T * N), where T is the number of time steps and N is the length of the queue.
 *                  In each of T iterations, the inner loop iterates N times.
 * Space Complexity: O(N) to store the queue string.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output operations,
  // which is common practice in competitive programming to meet time limits.
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);

  string s;
  int n, t;
  // Block Logic: Reads the queue length (n), number of time steps (t),
  // and the initial configuration of the queue (s) from standard input.
  cin >> n >> t >> s;

  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: At the start of each iteration, the string `s` represents the
   *            queue state after `(initial_t - t)` time steps have passed.
   */
  while (t--) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve priority inversions.
     * Logic: Iterates through the queue to identify adjacent 'B' (Boy) and 'G' (Girl)
     *        pairs where 'B' precedes 'G', and performs a local state transition (swap).
     * Invariant: For each `i`, `s[i]` and `s[i+1]` are considered for swapping.
     *            After processing `s[i]` and `s[i+1]`, `i` is advanced to `i+2` if a swap occurred,
     *            ensuring that a boy does not move more than one position per second.
     */
    for (int i = 0; i < n - 1; i++) { // Loop up to n-1 to prevent out-of-bounds access for s[i+1]
      // Pre-condition: Checks for an adjacent 'B' followed by 'G'.
      if (s[i] == 'B' && s[i + 1] == 'G') {
        // Functional Utility: Swaps the positions of the 'B' and 'G' characters.
        swap(s[i], s[i + 1]);
        // Inline: Increments `i` to skip the newly swapped 'G', ensuring that
        // this 'G' (which was originally 'B') is not considered again in the same time step,
        // effectively moving the boy only one step.
        i++;
      }
    }
  }
  // Post-condition: Prints the final configuration of the queue after all time steps have elapsed.
  cout << s << endl;
  return 0;
}
