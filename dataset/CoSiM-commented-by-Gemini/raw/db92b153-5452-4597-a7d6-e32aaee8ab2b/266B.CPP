/**
 * @file 266B.CPP
 * @brief Solution for Codeforces Problem 266B - "Queue at the School".
 *
 * @details
 * This program simulates a queue of children (Boys 'B' and Girls 'G').
 * In each time step (second), if a boy is standing directly in front of a girl,
 * they swap places. This simulation is run for 't' seconds.
 *
 * Algorithm: Iterative single-pass swap simulation.
 * For each second, the code iterates through the queue and performs all
 * adjacent 'BG' swaps. A crucial detail is incrementing the loop counter
 * after a swap to ensure that a child moves at most one position per time step.
 *
 * Time Complexity: O(T * N), where T is the number of seconds and N is the number of children.
 * Space Complexity: O(N), to store the queue configuration as a string.
 */
#include <iostream>
using namespace std;
int main() {
  // n: number of children, t: time in seconds
  int n, t;
  // s: the string representing the queue
  string s;
  cin >> n >> t >> s;
  
  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * This loop runs for 't' seconds (time steps).
   */
  while (t--) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve adjacent 'BG' pairs.
     * This represents all swaps happening within one second.
     */
    for (int i = 1; i < n; ++i) {
      // Pre-condition: Find a boy 'B' standing in front of a girl 'G'.
      if (s[i] == 'G' && s[i - 1] == 'B') {
        // Perform the swap.
        s[i] = 'B';
        s[i - 1] = 'G';
        // Inline: Crucially, increment 'i' to skip past the just-swapped girl.
        // This ensures that a single child moves at most one position per time step,
        // correctly modeling the simultaneous nature of the swaps.
        i++;
      }
    }
  }
  // Output the final state of the queue after 't' seconds.
  cout << s << endl;
}
