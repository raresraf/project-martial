/**
 * @file 266B.CPP
 * @brief Simulates a queue of boys and girls swapping positions over time.
 *
 * This program solves the "Queue at the School" problem. It simulates a queue
 * of boys ('B') and girls ('G'). In each time step, any boy standing immediately
 * in front of a girl swaps places with her. This is repeated for 't' seconds.
 *
 * @b Algorithm: Direct Time-Step Simulation
 *   The simulation proceeds second by second. In each second, a single pass is
 *   made through the queue to perform all possible 'BG' -> 'GB' swaps.
 *
 * @b Time_Complexity: O(N * T), where N is the queue length and T is the time in seconds.
 * @b Space_Complexity: O(N) to store the queue as a string.
 */
#include <iostream>

using namespace std;

int main() {
  // n: number of children in the queue.
  // t: the total time in seconds for the simulation.
  int n, t;
  // s: the string representing the queue.
  string s;
  cin >> n >> t;
  cin >> s;

  /**
   * @brief Outer loop simulates the passage of time for 't' seconds.
   */
  for (int i = 0; i < t; i++) {
    /**
     * @brief Inner loop performs a single pass over the queue for one second.
     * It identifies and performs all possible adjacent 'BG' swaps.
     */
    for (int j = 1; j < n; j++) {
      // Check for a boy 'B' standing before a girl 'G'.
      if (s[j] == 'G' && s[j - 1] == 'B') {
        // Swap their positions.
        s[j] = 'B';
        s[j - 1] = 'G';
        /**
         * @brief Advance the index 'j' to prevent a double-swap in one time step.
         * After a swap, the boy is at position 'j' and the girl at 'j-1'.
         * The next check should start after the newly moved boy. This `j++`
         * correctly skips the next adjacent pair for this time step.
         */
        j++;
      }
    }
  }

  // Print the final arrangement of the queue.
  cout << s;
  return 0;
}
