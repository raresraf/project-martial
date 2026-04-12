/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a state-reordering process based on local adjacency rules.
 * 
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: $O(T \times N)$ where T is the time duration and N is the queue length.
 * Space Complexity: $O(N)$ to store the entity sequence.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, t;
  string s;

  // Reads total entities (n) and the number of simulation time-steps (t).
  scanf("%d%d", &n, &t);
  cin >> s;

  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: At the start of each iteration, the string represents the queue state 
   * at time T=k.
   */
  for (int i = 1; i <= t; ++i) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve priority inversions.
     * Logic: Identifies 'B' (Boy) and 'G' (Girl) pairs where 'B' precedes 'G' 
     * and performs a local state transition (swap).
     */
    for (int j = 0; j < n - 1;) {
      if (s[j] == 'B' && s[j + 1] == 'G') {
        swap(s[j], s[j + 1]);
        // Inline: Skips the next index to prevent a single 'B' from migrating 
        // more than one position per time step (atomic movement).
        j += 2;
      } else
        // Normal progression if no swap criteria are met.
        j++;
    }
  }

  // Output the final converged or time-bounded state of the queue.
  cout << s << endl;
}
