/**
 * @file 266B.CPP
 * @brief Simulates positional swaps in a queue over time.
 *
 * This program solves the "Queue at the School" problem. It simulates a queue
 * of boys ('B') and girls ('G') for a given time `t`. In each second, any boy
 * standing immediately in front of a girl swaps places with her. The program
 * calculates the final arrangement of the queue after `t` seconds.
 *
 * @author f2e26bec-a427-47a7-b5eb-98971a21083a
 *
 * @algorithm Discrete Time Simulation with Conditional Index Hopping
 * @complexity
 * Time: O(T * N), where T is the time duration and N is the queue length.
 * Space: O(N), for storing the queue string.
 */
#include <cstring>
#include <iostream>
// `using namespace std;` is used for brevity.
using namespace std;
int main() {
  // `n`: the number of children (length of the queue).
  // `t`: the total simulation time in seconds.
  int n;
  int t;
  cin >> n >> t;
  // `s`: the string representing the queue's initial state.
  string s;
  cin >> s;

  /**
   * @brief Outer loop simulates the passage of time.
   * This loop runs `t` times, once for each second.
   */
  while (t--) {
    int i = 0;
    /**
     * @brief Inner loop processes the queue for the current second.
     * It scans the queue to find adjacent 'BG' pairs.
     */
    while (i < n - 1) { // Loop until the second to last element
      /**
       * @brief Checks for a 'B' followed by a 'G'.
       */
      if (s[i] == 'B' and s[i + 1] == 'G') {
        // If the pattern is found, swap the elements.
        swap(s[i], s[i + 1]);
        /**
         * @brief Advance index by 2.
         * This is a crucial step. By jumping two positions, we ensure that
         * the swapped pair is not re-evaluated in the same time step,
         * correctly modeling a single swap per element per second.
         */
        i += 2;
      } else {
        // If no swap occurs, move to the next position.
        i += 1;
      }
    }
  }
  // Print the final state of the queue.
  cout << s << '
';
  return 0;
}