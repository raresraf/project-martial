/**
 * @file 266B.CPP
 * @brief Simulates a queue reordering process based on adjacent character swaps.
 *   This program models the movement of 'B' (boys) and 'G' (girls) in a queue,
 *   where 'B' characters attempt to move past 'G' characters.
 * * Algorithm: Iterative Single-Pass Swap Simulation.
 *   In each time step (iteration), the queue is scanned. Every adjacent pair
 *   'BG' is swapped to 'GB'. To simulate an atomic movement where a 'B' moves
 *   at most one position per time step, the index is incremented an extra time
 *   after a swap.
 * Time Complexity: O(T * S) where T is the number of time steps and S is the length of the string.
 * Space Complexity: O(S) for storing the string.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to simulate the queue reordering process.
 * Functional Utility: Handles input, orchestrates the queue simulation logic, and outputs the final queue state.
 */
int main() {
  string s;       // The queue represented as a string of 'B' and 'G' characters.
  // Variables for loop control and string size. 'f', 'l', 'j', 'm' are unused.
  long int i, x, t, n;

  /**
   * Block Logic: Continuously reads input until End-Of-File (EOF).
   * Functional Utility: Allows processing multiple test cases from a single input stream.
   * Invariant: At the start of each iteration, 'n' is the string length and 't' is the number of time steps.
   */
  while (cin >> n >> t) {
    cin >> s; // Reads the initial state of the queue string.

    /**
     * Block Logic: Orchestrates the temporal progression of the simulation.
     * Invariant: 'x' represents the current time step, iterating up to 't'.
     */
    for (x = 1; x <= t; x++) {
      /**
       * Block Logic: Performs a single-pass sweep to resolve priority inversions ('BG' pairs).
       * Pre-condition: 's' represents the queue state at the beginning of the current time step.
       * Invariant: After each full pass, all immediately adjacent 'BG' pairs (where 'B' precedes 'G')
       *   have been swapped to 'GB', ensuring each 'B' moves at most one position forward.
       */
      for (i = 1; i < s.size(); i++) {
        /**
         * Condition: Checks for an adjacent 'G' followed by a 'B' (which means 'B' is at s[i-1] and 'G' at s[i]).
         * If the pattern 'BG' is found, it indicates a boy needs to move past a girl.
         */
        if (s[i] == 'G' && s[i - 1] == 'B') {
          s[i] = 'B';     // Boy moves to the current position.
          s[i - 1] = 'G'; // Girl moves to the previous position.
          i++;            // Increments 'i' to skip the newly moved 'G' and ensure
                          // that each 'B' moves at most one position per time step.
        }
      }
    }
    cout << s << endl; // Outputs the final state of the queue after 't' time steps.
  }

  return 0; // Indicates successful program execution.
}
