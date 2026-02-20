/**
 * @file 266B.CPP
 * @brief Simulates a queue of boys and girls swapping positions over time.
 *
 * This program solves the "Queue at the School" problem. It models a queue
 * of boys ('B') and girls ('G'). In each time step, any boy standing
 * immediately in front of a girl will swap places with her. This simulation
 * runs for a specified number of time steps 't'.
 *
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: O(T * N), where T is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) to store the queue string.
 */
#include <iostream>
#include <string> // For std::string
#include <utility> // For std::swap

int main() {
  int n; // Length of the queue
  int t; // Number of time steps
  std::string queue;
  std::cin >> n >> t;
  std::cin >> queue;

  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * This outer loop iterates for 't' time steps.
   * Invariant: At the start of each iteration, `queue` represents the state at time `i`.
   */
  for (int i = 0; i < t; i++) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve swaps for one time step.
     * This inner loop identifies adjacent 'B' and 'G' pairs and swaps them.
     */
    for (int j = 0; j < n; j++) {
      // Pre-condition: Check for a 'B' followed by a 'G'.
      if (queue[j] == 'B' && queue[j + 1] == 'G') {
        std::swap(queue[j], queue[j + 1]);
        // Inline: Skips the next index. After a swap (B,G) -> (G,B), the new 'B' at j+1
        // should not be considered for another swap in the same time step. This ensures
        // that each boy moves at most one position to the right per time step.
        j++;
      }
    }
  }

  // Output the final state of the queue after 't' seconds.
  std::cout << queue;

  return 0;
}
