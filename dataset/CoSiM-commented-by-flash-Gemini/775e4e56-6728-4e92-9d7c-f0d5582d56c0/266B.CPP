/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem from Codeforces (266B).
 * @details This program simulates the process of boys and girls rearranging themselves
 * in a queue. Over 't' seconds, any boy standing directly in front of a girl
 * will swap places with her.
 *
 * @section Algorithm
 * The simulation is performed for 't' time steps. In each step, the algorithm
 * iterates through the queue string from right to left. This direction is crucial.
 * 1. An outer loop runs 't' times, representing each second of the simulation.
 * 2. An inner loop scans the queue from `n-1` down to `0`.
 * 3. If a 'G' (girl) is found at position `i` and a 'B' (boy) is at `i-1`, they are swapped.
 * 4. After a swap, the loop counter `i` is decremented an additional time (`i--`). This
 *    is a critical step to ensure that a boy who has just moved forward does not
 *    participate in another swap within the same time step, correctly modeling
 *    the simultaneous nature of the swaps.
 * After 't' seconds, the final arrangement of the queue is printed.
 *
 * @section Complexity
 * - Time Complexity: O(n * t), where 'n' is the number of children in the queue and
 *   't' is the number of seconds.
 * - Space Complexity: O(n) to store the string representing the queue.
 */
#include <iostream>
#include <memory.h>
#include <string>

using namespace std;

int main() {
  /**
   * @var n The number of children in the queue.
   * @var t The time in seconds for the simulation.
   * @var s The string representing the queue of children.
   */
  int n, t;
  string s;
  cin >> n >> t >> s;

  /**
   * @brief Block Logic: This outer loop simulates the passage of time, with each
   * iteration representing one second.
   */
  while (t--)
    /**
     * @brief Block Logic: This inner loop performs all swaps for a single time step.
     * It iterates from right to left to ensure that swaps are based on the state at
     * the beginning of the time step.
     */
    for (int i = n - 1; i >= 0; i--)
      /**
       * @brief Logic: Identifies a boy ('B') at `i-1` followed by a girl ('G') at `i`.
       */
      if (s[i] == 'G' && i - 1 >= 0 && s[i - 1] == 'B') {
        // Perform the swap.
        s[i] = 'B';
        s[i - 1] = 'G';
        /**
         * @brief Inline: Decrement 'i' again to skip the new position of the girl.
         * This prevents the boy from moving more than one spot per time step,
         * correctly simulating simultaneous movement.
         */
        i--;
      }

  // Output the final state of the queue after 't' seconds.
  cout << s << endl;

  return 0;
}