/**
 * @file 266B.CPP
 * @brief Solution for CodeForces Problem 266B - "Queue at the School".
 * 
 * @details
 * This program simulates a queue of children, represented by 'B' for boys and 'G' for girls.
 * In each time step, any boy ('B') standing directly in front of a girl ('G') swaps places with her.
 * The simulation runs for a specified number of time steps (`t`).
 *
 * @algorithm Iterative single-pass swap simulation.
 * @time_complexity O(T * N), where T is the number of time steps and N is the length of the queue.
 * @space_complexity O(N) to store the queue sequence.
 */
#include <algorithm>
#include <stdio.h>

using namespace std;

int main() {
  int n; // Length of the queue.
  int t; // Number of time steps to simulate.
  scanf("%d %d", &n, &t);
  char queue[n + 1]; // Character array to store the queue, +1 for null terminator.
  scanf("%s", queue);

  /**
   * @brief Block that orchestrates the temporal progression of the simulation.
   * This outer loop runs the simulation for `t` discrete time steps.
   * Invariant: At the start of each iteration, `queue` represents the state at time `k`.
   * After the inner loop, it represents the state at time `k+1`.
   */
  while (t--) {
    /**
     * @brief Block that performs a single-pass sweep to resolve inversions ('BG' pairs).
     * This inner loop iterates through the queue to find adjacent 'B' and 'G' pairs
     * and swaps them.
     */
    for (int i = 0; i < n - 1; i++) {
      // Pre-condition: Check if a boy is immediately followed by a girl.
      if (queue[i] == 'B' && queue[i + 1] == 'G') {
        // Perform the local state transition (swap).
        swap(queue[i], queue[i + 1]);
        /**
         * @brief Inline Logic: Skips the next index.
         * This is crucial to ensure that a single boy does not move more than one position
         * to the right in a single time step, preserving the atomic nature of the movement.
         */
        i++;
      }
    }
  }

  // Output the final state of the queue after `t` time steps.
  printf("%s
", queue);
  return 0;
}
