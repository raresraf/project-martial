/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a state-reordering process based on local adjacency rules.
 * * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: $O(T \times N)$ where T is the time duration and N is the queue length.
 * Space Complexity: $O(N)$ to store the entity sequence.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and flushes 
   * to optimize high-throughput I/O required for competitive programming constraints.
   */
  ios::sync_with_stdio(false);
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: At the start of each iteration, the string represents the queue state at time T-k.
   */
  for (int j = 0; j < t; ++j) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve priority inversions.
     * Logic: Identifies 'B' (Boy) and 'G' (Girl) pairs where 'B' precedes 'G' 
     * and performs a local state transition (swap).
     */
    for (int i = 0; i < n - 1; ++i) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        // Inline: Skips the next index to prevent a single 'B' from migrating 
        // more than one position per time step (atomic movement).
        i++;
      }
    }
  }
  // Output the final converged or time-bounded state of the queue.
  cout << s << "\n";
  return 0;
}
