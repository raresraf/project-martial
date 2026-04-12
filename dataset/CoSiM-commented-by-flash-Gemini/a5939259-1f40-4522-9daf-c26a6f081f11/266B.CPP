/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of queue reordering based on local adjacency rules.
 * 
 * Algorithm: Iterative single-pass swap simulation.
 * Logic: Simulates a queue where entities 'B' move behind entities 'G' if they are 
 * adjacent. Each time-step 't' allows each 'B' to move at most one position.
 * 
 * Time Complexity: $O(T \times N)$ where T is simulation duration and N is queue length.
 * Space Complexity: $O(N)$ for the character sequence buffer.
 */

#include <algorithm>
#include <stdio.h>

using namespace std;

int main() {
  int n, t;
  // Functional Utility: Reads total headcount (n) and the temporal limit (t).
  scanf("%d %d", &n, &t);
  
  // Buffer for the queue state.
  char queue[n + 1];
  scanf("%s", &queue);

  /**
   * Block Logic: Orchestrates the temporal progression of the queue.
   * Invariant: Each iteration of the while loop represents one discrete unit of time.
   */
  while (t--) {
    /**
     * Block Logic: Resolves priority inversions in a single sweep.
     * Logic: Identifies 'B' preceding 'G' and performs a local state transition (swap).
     */
    for (int i = 0; i < n - 1; i++) {
      if (queue[i] == 'B' && queue[i + 1] == 'G') {
        swap(queue[i], queue[i + 1]);
        /**
         * Inline: Increments index 'i' to prevent a single 'B' from migrating 
         * through multiple 'G's in a single time step (atomic movement).
         */
        i++;
      }
    }
  }
  
  // Output the final state of the queue after T time steps.
  printf("%s\n", queue);
  
  return 0;
}
