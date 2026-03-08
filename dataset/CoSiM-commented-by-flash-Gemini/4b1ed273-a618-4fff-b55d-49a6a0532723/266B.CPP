/**
 * @4b1ed273-a618-4fff-b55d-49a6a0532723/266B.CPP
 * @brief Discrete-time simulation of a state-reordering process based on local adjacency rules.
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: $O(T \times N)$ where T is the time duration and N is the queue length.
 * Space Complexity: $O(N)$ to store the entity sequence.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, t;    // n: length of the queue, t: number of time steps (iterations).
  string A;    // A: Represents the queue sequence of 'B' and 'G'.
  // Functional Utility: Read the queue length 'n', number of time steps 't', and the initial queue sequence 'A'.
  cin >> n >> t;
  cin >> A;

  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: At the start of each iteration, the string represents the queue state at time T-k.
   */
  while (t--) {
    // i: Loop counter for iterating through the queue.
    int i = 0;
    /**
     * Block Logic: Performs a single-pass sweep to resolve priority inversions.
     * Logic: Identifies 'B' (Boy) and 'G' (Girl) pairs where 'B' precedes 'G'
     * and performs a local state transition (swap).
     * Invariant: At the start of each inner loop iteration, the substring `A[0...i-1]` is processed for current time step.
     */
    while (i < (n - 1)) { // Iterate up to the second-to-last element to check pairs.
      // Pre-condition: Check for a 'B' followed by a 'G'.
      if (A[i] == 'B' && A[i + 1] == 'G') {
        A[i] = 'G';     // Swap 'B' to 'G'.
        A[i + 1] = 'B'; // Swap 'G' to 'B'.
        // Inline: Skips the next index to prevent a single 'B' from migrating
        // more than one position per time step (atomic movement).
        i += 2;
      } else {
        i++; // Move to the next element if no swap occurred.
      }
    }
  }
  // Functional Utility: Output the final converged or time-bounded state of the queue.
  cout << A;
  return 0; // Indicate successful program termination.
}