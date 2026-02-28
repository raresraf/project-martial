/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a state-reordering process based on local adjacency rules.
 *
 * This program simulates a queue of boys ('B') and girls ('G'). Over a series of
 * time steps, if a boy is immediately followed by a girl ("BG"), they swap positions
 * ("GB"). This process is repeated for a specified number of time steps.
 *
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: $O(B \times N)$ where B is the number of time steps and N is the queue length.
 * Space Complexity: $O(N)$ to store the entity sequence.
 */

#include <algorithm>
#include <iostream>
using namespace std;
int n, b;
string a;
int main() {
  /**
   * @brief Main function to execute the queue reordering simulation.
   *
   * Reads the queue length (n), number of time steps (b), and initial queue
   * configuration (string a). It then applies the swapping rule for each
   * time step and prints the final state of the queue.
   */
  // @param n The length of the queue (number of people).
  // @param b The number of time steps for the simulation.
  // @param a The string representing the current state of the queue ('B' for boy, 'G' for girl).
  cin >> n >> b >> a;
  /**
   * @brief Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: At the start of each iteration, the string `a` represents the queue state at time `t`,
   *            and the loop processes the state transition to time `t+1` for `b` steps.
   */
  for (int i = 0; i < b; i++) {
    /**
     * @brief Block Logic: Performs a single-pass sweep to resolve priority inversions.
     * Logic: Identifies 'B' (Boy) and 'G' (Girl) pairs where 'B' precedes 'G'
     *        and performs a local state transition (swap).
     * Invariant: At the start of each inner iteration, characters `a[0...j-1]` have been checked for swaps in this pass.
     */
    for (int j = 0; j < n; j++) {
      // @brief Condition to check for a 'B' followed by a 'G'.
      if (a[j] == 'B' && a[j + 1] == 'G') {
        // @brief Performs the swap operation for the 'B' and 'G' pair.
        swap(a[j], a[j + 1]);
        // Inline: Skips the next index to prevent a single 'B' from migrating
        //         more than one position per time step (atomic movement).
        j += 1;
      }
    }
  }
  // @brief Prints the final converged or time-bounded state of the queue.
  cout << a;
}