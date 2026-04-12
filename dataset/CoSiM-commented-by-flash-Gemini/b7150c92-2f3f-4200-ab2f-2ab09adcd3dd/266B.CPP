/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of queue reordering based on local adjacency rules.
 * 
 * Logic: Simulates a queue where entities of type 'B' move behind entities of 
 * type 'G' if they are adjacent. Each time-step allows each 'B' to migrate at 
 * most one position towards the back.
 * 
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: $O(T \times N)$ where T is the time duration and N is the queue length.
 * Space Complexity: $O(N)$ for string storage.
 */

#include <iostream>

using namespace std;

int main() {
  int n, t;
  // Functional Utility: Reads total entities (n) and the simulation time duration (t).
  if (!(cin >> n >> t)) return 0;
  
  string s;
  cin >> s;

  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: Each outer loop iteration represents one discrete unit of time.
   */
  for (int j = 0; j < t; ++j) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve priority inversions.
     * Logic: Identifies 'B' preceding 'G' and performs a local state transition (swap).
     */
    for (int i = 0; i < (int)s.size() - 1;) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        /**
         * Inline: Skips the next index to prevent a single 'B' from migrating 
         * more than one position per time step (atomic movement).
         */
        i += 2;
      } else {
        // Normal progression if no swap conditions are met.
        ++i;
      }
    }
  }
  
  // Output the final state of the queue after the time bound is reached.
  cout << s;
  
  return 0;
}
