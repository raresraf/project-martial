/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of queue reordering based on local adjacency rules.
 * 
 * Logic: Simulates a queue where entities of type 'B' (Boys) move behind entities 
 * of type 'G' (Girls) if they are adjacent. Each time-step 'j' allows each 'B' 
 * to migrate at most one position towards the back.
 * 
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: $O(J \times N)$ where J is the simulation duration and N is the queue length.
 * Space Complexity: $O(N)$ for string storage.
 */

#include <iostream>
#include <string>
using namespace std;

/**
 * @brief Functional Utility: Executes the queue reordering simulation.
 * @param a The current state of the queue.
 * @param n The total number of entities in the queue.
 * @param j The number of discrete time steps to simulate.
 * @return The final state of the queue after J time steps.
 */
string findOrder(string a, int n, int j) {
  /**
   * Block Logic: Orchestrates the temporal progression.
   * Invariant: Each while(j--) iteration represents one discrete unit of time.
   */
  while (j--) {
    int i = 0;
    /**
     * Block Logic: Performs a single-pass sweep to resolve priority inversions.
     * Logic: Identifies 'B' preceding 'G' and performs a local state transition (swap).
     */
    while (i < n - 1) {
      if (a[i] == 'B' && a[i + 1] == 'G') {
        // Atomic state transition.
        a[i] = 'G';
        a[i + 1] = 'B';
        /**
         * Inline: Skips the next index to prevent a single 'B' from migrating 
         * more than one position per time step (atomic movement constraint).
         */
        i += 2;
      } else {
        // Normal progression if no swap criteria are met.
        i++;
      }
    }
  }
  return a;
}

int main(int argc, char const *argv[]) {
  int n, j;
  // Functional Utility: Reads total entities (n) and the temporal limit (j).
  if (!(cin >> n >> j)) return 0;
  
  string str;
  cin >> str;
  
  // Output the final converged or time-bounded state of the queue.
  cout << findOrder(str, n, j);
  
  return 0;
}
