/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a state-reordering process based on local adjacency rules.
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the queue length.
 * Space Complexity: O(N) to store the entity sequence.
 */
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main(void) {
  int n, t;
  string queue, result;
  // Reads the queue length (n) and the number of time steps (t).
  cin >> n >> t;
  // Reads the initial configuration of the queue.
  cin >> queue;
  result = queue;

  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: At the start of each iteration, 'queue' represents the queue state at time T-k.
   */
  for (int i = 0; i < t; i++) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve priority inversions.
     * Logic: Iterates through the queue to find adjacent 'B' (Boy) and 'G' (Girl) pairs.
     * Invariant: Elements up to the current index 'i' have been considered for a swap.
     */
    for (int i = 0; i < queue.size() - 1; i++) {
      /**
       * Conditional Logic: Identifies a 'B' followed by a 'G' to initiate a swap.
       * If a 'B' is at index 'i' and a 'G' is at index 'i+1', they swap positions.
       */
      if (queue.at(i) == 'B' && queue.at(i + 1) == 'G') {
        result.at(i) = queue.at(i + 1);
        result.at(i + 1) = queue.at(i);
      }
    }

    // Updates the 'queue' with the new state from 'result' for the next time step.
    queue = result;
  }

  // Output the final converged or time-bounded state of the queue.
  cout << result << endl;
  return 0;
}