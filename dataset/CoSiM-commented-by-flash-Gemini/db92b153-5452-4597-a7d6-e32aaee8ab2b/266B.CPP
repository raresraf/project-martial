/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a state-reordering process based on local adjacency rules,
 * specifically swapping 'B' (Boy) and 'G' (Girl) in a queue when a 'G' is immediately
 * followed by a 'B'. This simulates 'G' moving ahead of 'B'.
 *
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the queue length.
 * Space Complexity: O(N) to store the entity sequence.
 */

#include <iostream>
#include <string> // Required for std::string

using namespace std;

int main() {
  // Functional Utility: 'n' represents the length of the queue (string).
  int n;
  // Functional Utility: 't' represents the number of time steps (iterations) for the simulation.
  int t;
  // Functional Utility: 's' stores the current state of the queue as a string of characters.
  string s;

  // Block Logic: Reads the queue length 'n', number of time steps 't', and the initial queue string 's'.
  cin >> n >> t >> s;

  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: At the start of each iteration, 't' represents the remaining time steps to simulate.
   */
  while (t--) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve priority inversions where 'G' is after 'B'.
     * Invariant: After each iteration of this inner loop, characters `s[i-1]` and `s[i]` are considered for swapping.
     */
    for (int i = 1; i < n; ++i) {
      /**
       * Block Logic: Checks for the specific condition where a 'G' is immediately followed by a 'B'.
       * Pre-condition: 's[i]' and 's[i-1]' are valid characters within the string 's'.
       * Invariant: If 's[i]' is 'G' and 's[i-1]' is 'B', they will be swapped.
       */
      if (s[i] == 'G' && s[i - 1] == 'B') {
        // Functional Utility: Swaps the 'G' and 'B' characters.
        s[i] = 'B';
        s[i - 1] = 'G';
        // Inline: Skips the next index to prevent re-processing the newly swapped 'B' in the same pass.
        // This ensures the 'B' moves only one position forward per time step.
        i++;
      }
    }
  }
  // Output the final state of the queue after 't' time steps.
  cout << s << endl;
  return 0;
}
