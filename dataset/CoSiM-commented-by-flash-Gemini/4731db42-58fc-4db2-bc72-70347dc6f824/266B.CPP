/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a state-reordering process based on local adjacency rules.
 *
 * This program simulates the reordering of 'B' (boy) and 'G' (girl) characters
 * in a queue. In each time step, if a 'B' is followed by a 'G', they swap positions.
 * The simulation runs for a specified number of time steps.
 *
 * Algorithm: Iterative single-pass swap simulation. The loop scans from left to right,
 * swapping 'B' and 'G' if 'B' is immediately followed by 'G'. To prevent a single 'B'
 * from moving multiple positions in one time step, the inner loop index is incremented
 * after a swap.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the queue length.
 * Space Complexity: O(N) to store the character array.
 */
// input
// 5 1
// BGGBG
// output
// GBGGB

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to execute the queue reordering simulation.
 *
 * Reads the queue length (n), number of time steps (t), and the initial queue
 * configuration from standard input. It then simulates the reordering process
 * and prints the final state of the queue.
 *
 * @return 0 on successful execution.
 */
int main() {

// Conditional compilation for local debugging.
// Pre-condition: `ONLINE_JUDGE` is not defined.
// Block Logic: Redirects standard input from "input.txt" and standard output to "output.txt".
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  // Declare integers for queue length 'n' and number of time steps 't'.
  int n, t;
  // Read 'n' and 't' from standard input.
  cin >> n >> t;
  // Declare a string 'a' to store the queue configuration.
  string a;
  // Read the initial queue configuration string into 'a'.
  cin >> a;

  /**
   * @brief Block Logic: Orchestrates the temporal progression of the simulation.
   *
   * This loop iterates 't' times, with each iteration representing one unit of time.
   * Invariant: At the start of each iteration, the string 'a' represents the queue
   * state at time T-k, where k is the number of completed iterations.
   */
  while (t-- > 0) {
    /**
     * @brief Block Logic: Performs a single-pass sweep to resolve priority inversions.
     *
     * This loop iterates through the queue from the second element (index 1) to the end.
     * It identifies 'G' (Girl) followed by 'B' (Boy) where 'B' was originally
     * at `i-1` and 'G' at `i`, effectively simulating a 'B' moving right.
     * Invariant: At the start of each inner iteration, characters up to 'i-1' have
     * been considered for swapping in the current time step.
     */
    for (int i = 1; i < n; i++) {
      // Pre-condition: Check if the current character is 'G' and the previous is 'B'.
      // This identifies a 'B' that needs to move past a 'G'.
      if (a[i] == 'G' && a[i - 1] == 'B') {
        // Perform the swap: 'B' moves to the right, 'G' moves to the left.
        a[i] = 'B';
        a[i - 1] = 'G';
        // Inline: Increment 'i' to skip the character that was just moved to the
        // current position. This ensures that a 'B' does not migrate more than one
        // position per time step (atomic movement) and prevents double-processing.
        ++i;
      }
    }
  }
  // Print the final converged or time-bounded state of the queue.
  cout << a;
  return 0;
}
