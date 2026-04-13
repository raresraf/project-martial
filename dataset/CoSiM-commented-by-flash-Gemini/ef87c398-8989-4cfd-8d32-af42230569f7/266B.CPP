/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a state-reordering process based on local adjacency rules.
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: O(T * N) where T is the time duration and N is the queue length.
 * Space Complexity: O(N) to store the entity sequence.
 */

#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience.

#define ll long long int // Defines a macro 'll' as a shorthand for 'long long int'.
#define MAX (ll)(10e6 + 5) // Defines a constant for maximum value, often used in array sizing or limits.

using namespace std;

int main() {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and flushes
   * to optimize high-throughput I/O required for competitive programming constraints.
   */
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);
  // Optional: For local debugging, redirects standard input from "266B.in".
  // freopen("266B.in","r",stdin);
  int n, t; // 'n' represents the length of the queue, 't' represents the number of time steps.
  string s; // Stores the current state of the queue as a string.
  // Reads the queue length (n) and the number of time steps (t).
  cin >> n >> t;
  // Reads the initial configuration of the queue.
  cin >> s;
  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: At the start of each iteration, the string 's' represents the queue state at time T-k.
   */
  while (t--) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve priority inversions.
     * Logic: Iterates through the queue to find adjacent 'B' (Boy) and 'G' (Girl) pairs.
     */
    for (int i = 0; i < n; i++) {
      /**
       * Conditional Logic: Identifies a 'B' followed by a 'G' and performs a local state transition (swap).
       */
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';     // Swap 'B' with 'G'.
        s[i + 1] = 'B'; // Swap 'G' with 'B'.
        // Inline: Skips the next index to prevent a single 'B' from migrating
        // more than one position per time step (ensures atomic movement).
        i++;
      }
    }
  }
  // Output the final converged or time-bounded state of the queue.
  cout << s << endl;
  return 0;
}